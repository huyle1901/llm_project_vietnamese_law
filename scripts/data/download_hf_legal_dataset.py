from __future__ import annotations

import argparse
import json
from pathlib import Path

def iter_parquet_rows(parquet_paths: list[Path], batch_size: int):
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to read parquet shards. Install it first, e.g. "
            "`uv pip install pyarrow`."
        ) from e
    for p in parquet_paths:
        pf = pq.ParquetFile(str(p))
        print(f"reading {p.name} rows={pf.metadata.num_rows}")
        for batch in pf.iter_batches(batch_size=batch_size):
            cols = batch.to_pydict()
            if not cols:
                continue
            keys = list(cols.keys())
            n = len(cols[keys[0]])
            for i in range(n):
                yield {k: cols[k][i] for k in keys}


def write_jsonl(path: Path, rows, progress_every: int = 50000) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            c += 1
            if c % progress_every == 0:
                print(f"written={c} -> {path.name}")
    return c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a Hugging Face legal dataset and export full JSONL files.")
    p.add_argument("--dataset", type=str, default="vohuutridung/vietnamese-legal-documents")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--raw-dir", type=Path, default=Path("data/datahuggingface/hf_legal_dataset_raw"))
    p.add_argument("--content-out", type=Path, default=Path("data/datahuggingface/content_full.jsonl"))
    p.add_argument("--metadata-out", type=Path, default=Path("data/datahuggingface/metadata_full.jsonl"))
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to download the dataset snapshot. Install it first, e.g. "
            "`uv pip install huggingface_hub`."
        ) from e

    print("downloading snapshot...")
    snapshot_path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["content/*.parquet", "metadata/*.parquet", "README.md"],
        local_dir=str(args.raw_dir),
    )
    print("snapshot:", snapshot_path)

    content_parquets = sorted((args.raw_dir / "content").glob("*.parquet"))
    metadata_parquets = sorted((args.raw_dir / "metadata").glob("*.parquet"))

    if not content_parquets:
        raise SystemExit("No content parquet files found in raw_dir/content")
    if not metadata_parquets:
        raise SystemExit("No metadata parquet files found in raw_dir/metadata")

    print(f"content shards={len(content_parquets)}")
    content_rows = write_jsonl(
        args.content_out,
        iter_parquet_rows(content_parquets, batch_size=args.batch_size),
    )

    print(f"metadata shards={len(metadata_parquets)}")
    metadata_rows = write_jsonl(
        args.metadata_out,
        iter_parquet_rows(metadata_parquets, batch_size=args.batch_size),
    )

    summary = {
        "dataset": args.dataset,
        "revision": args.revision,
        "raw_dir": str(args.raw_dir),
        "content_out": str(args.content_out),
        "metadata_out": str(args.metadata_out),
        "content_rows": content_rows,
        "metadata_rows": metadata_rows,
        "content_shards": [p.name for p in content_parquets],
        "metadata_shards": [p.name for p in metadata_parquets],
    }
    summary_path = args.raw_dir / "download_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
