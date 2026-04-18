#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm


DEFAULT_INPUT = Path("data/datahuggingface/corpus_important_docs_chunks.jsonl")
DEFAULT_OUTPUT = Path("/kaggle/working/emb_out")
DEFAULT_META_KEYS = (
    "chunk_id,id,document_number,title,url,legal_type,legal_sectors,"
    "issuing_authority,issuance_date,signers,section_type,article_no,clause_no,"
    "word_count,token_estimate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed full chunk JSONL on Kaggle and export shards for FAISS/OpenSearch pipelines."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="Embedding model on Hugging Face",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = auto",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--prefix-mode",
        type=str,
        default="auto",
        choices=["auto", "query_passage", "none"],
        help="For E5 models, corpus text should usually be prefixed with 'passage: '",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize embeddings for cosine / inner-product retrieval",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization",
    )
    parser.add_argument(
        "--emb-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for exported .npy",
    )
    parser.add_argument("--text-key", type=str, default="chunk_text")
    parser.add_argument("--id-key", type=str, default="chunk_id")
    parser.add_argument(
        "--meta-keys",
        type=str,
        default=DEFAULT_META_KEYS,
        help="Comma-separated metadata keys to save",
    )
    parser.add_argument(
        "--metadata-format",
        type=str,
        default="parquet",
        choices=["parquet", "jsonl", "csv"],
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Rows per shard",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all rows; otherwise process at most N rows",
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def infer_batch_size(model_name: str, user_batch_size: int) -> int:
    if user_batch_size > 0:
        return user_batch_size
    lower = model_name.lower()
    if "e5-small" in lower:
        return 128
    if "e5-base" in lower:
        return 96
    return 64


def infer_prefix_mode(model_name: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "query_passage" if "e5" in model_name.lower() else "none"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_metadata(rows: list[dict[str, Any]], out_path: Path, fmt: str) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out_path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path.with_suffix(".jsonl").name

    if fmt == "csv":
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
        return out_path.with_suffix(".csv").name

    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(out_path.with_suffix(".parquet"), index=False)
        return out_path.with_suffix(".parquet").name
    except Exception:
        fallback = out_path.with_suffix(".jsonl")
        with fallback.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return fallback.name


def flush_shard(
    shard_id: int,
    emb_parts: list[np.ndarray],
    meta_rows: list[dict[str, Any]],
    output_dir: Path,
    emb_dtype: np.dtype,
    metadata_format: str,
) -> dict[str, Any]:
    arr = np.concatenate(emb_parts, axis=0)
    if arr.dtype != emb_dtype:
        arr = arr.astype(emb_dtype, copy=False)

    emb_name = f"embeddings_{shard_id:05d}.npy"
    np.save(output_dir / emb_name, arr)

    meta_stub = output_dir / f"metadata_{shard_id:05d}"
    meta_name = write_metadata(meta_rows, meta_stub, metadata_format)

    return {
        "shard_id": shard_id,
        "rows": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "embedding_file": emb_name,
        "metadata_file": meta_name,
        "embedding_dtype": str(arr.dtype),
    }


def main() -> int:
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Missing dependency: sentence-transformers") from e

    if not args.input_jsonl.exists():
        raise FileNotFoundError(
            "Input JSONL not found. Pass --input-jsonl explicitly, e.g. "
            "--input-jsonl /kaggle/input/<dataset>/corpus_important_docs_chunks.jsonl"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix_mode = infer_prefix_mode(args.model_name, args.prefix_mode)
    batch_size = infer_batch_size(args.model_name, args.batch_size)
    emb_dtype = np.float16 if args.emb_dtype == "float16" else np.float32

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=args.device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length

    total_rows = count_jsonl(args.input_jsonl)
    if args.limit > 0:
        total_rows = min(total_rows, args.limit)

    print(f"Rows to process: {total_rows}")
    print(f"Device={args.device} | batch_size={batch_size} | max_length={args.max_length}")
    print(f"prefix_mode={prefix_mode} | normalize={args.normalize} | emb_dtype={args.emb_dtype}")

    meta_keys = [k.strip() for k in args.meta_keys.split(",") if k.strip()]
    if args.id_key not in meta_keys:
        meta_keys = [args.id_key] + meta_keys

    emb_parts: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []

    shard_size = max(1, args.shard_size)
    shard_id = 0
    processed = 0
    started = time.time()

    pbar = tqdm(total=total_rows, desc="Embedding", unit="rows")

    batch_texts: list[str] = []
    batch_meta: list[dict[str, Any]] = []

    def process_batch() -> None:
        nonlocal batch_texts, batch_meta, emb_parts, meta_rows
        if not batch_texts:
            return
        embs = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=args.normalize,
            convert_to_numpy=True,
        )
        emb_parts.append(embs)
        meta_rows.extend(batch_meta)
        batch_texts = []
        batch_meta = []

    for row in iter_jsonl(args.input_jsonl):
        if args.limit > 0 and processed >= args.limit:
            break

        raw_text = str(row.get(args.text_key, "")).strip()
        if not raw_text:
            continue

        text = f"passage: {raw_text}" if prefix_mode == "query_passage" else raw_text
        batch_texts.append(text)
        batch_meta.append({k: row.get(k) for k in meta_keys})

        if len(batch_texts) >= batch_size:
            process_batch()

        processed += 1
        pbar.update(1)

        current_rows = sum(x.shape[0] for x in emb_parts)
        if current_rows >= shard_size:
            shard_info = flush_shard(
                shard_id=shard_id,
                emb_parts=emb_parts,
                meta_rows=meta_rows,
                output_dir=args.output_dir,
                emb_dtype=emb_dtype,
                metadata_format=args.metadata_format,
            )
            shard_manifest.append(shard_info)
            shard_id += 1
            emb_parts = []
            meta_rows = []

            if args.log_every > 0 and shard_id % args.log_every == 0:
                elapsed = time.time() - started
                speed = processed / max(elapsed, 1e-6)
                print(f"Shards={shard_id} | processed={processed} | speed={speed:.1f} rows/s")

    process_batch()

    if emb_parts:
        shard_info = flush_shard(
            shard_id=shard_id,
            emb_parts=emb_parts,
            meta_rows=meta_rows,
            output_dir=args.output_dir,
            emb_dtype=emb_dtype,
            metadata_format=args.metadata_format,
        )
        shard_manifest.append(shard_info)

    pbar.close()

    total_emb_rows = int(sum(s["rows"] for s in shard_manifest))
    dim = int(shard_manifest[0]["dim"]) if shard_manifest else 0

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "model_name": args.model_name,
        "device": args.device,
        "max_length": args.max_length,
        "prefix_mode": prefix_mode,
        "normalize": args.normalize,
        "embedding_dtype": args.emb_dtype,
        "text_key": args.text_key,
        "id_key": args.id_key,
        "meta_keys": meta_keys,
        "metadata_format": args.metadata_format,
        "total_rows": total_emb_rows,
        "embedding_dim": dim,
        "shard_size_target": shard_size,
        "shards": shard_manifest,
        "created_at_unix": int(time.time()),
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - started
    print("Done")
    print(f"- total_rows: {total_emb_rows}")
    print(f"- embedding_dim: {dim}")
    print(f"- shards: {len(shard_manifest)}")
    print(f"- elapsed_sec: {elapsed:.1f}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
