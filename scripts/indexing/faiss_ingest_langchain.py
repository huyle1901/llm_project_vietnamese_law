#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from langchain_core.embeddings import Embeddings
except Exception:
    class Embeddings:  # type: ignore[no-redef]
        pass


class NoOpEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("NoOpEmbeddings cannot embed documents")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("NoOpEmbeddings cannot embed query")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_text_map(path: Path, id_key: str = "chunk_id", text_key: str = "chunk_text") -> dict[str, str]:
    out: dict[str, str] = {}
    for row in iter_jsonl(path):
        cid = str(row.get(id_key, "")).strip()
        if not cid:
            continue
        out[cid] = str(row.get(text_key, "") or "")
    return out


def load_metadata(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".jsonl":
        return list(iter_jsonl(path))
    raise ValueError(f"Unsupported metadata format: {path}")


def build_index(args: argparse.Namespace) -> int:
    try:
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.faiss import DistanceStrategy
        from langchain_core.documents import Document
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for FAISS ingest. Install faiss-cpu/faiss-gpu and langchain-community."
        ) from e

    manifest = load_manifest(args.manifest)
    emb_root = args.emb_root.resolve()

    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError("manifest.json has no shards")

    dim = int(manifest.get("embedding_dim", 0))
    if dim <= 0:
        raise ValueError("Invalid embedding_dim in manifest")

    normalize = bool(manifest.get("normalize", True))
    if normalize:
        index = faiss.IndexFlatIP(dim)
        distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
    else:
        index = faiss.IndexFlatL2(dim)
        distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE

    text_map: dict[str, str] = {}
    if args.text_jsonl:
        text_jsonl = args.text_jsonl.resolve()
        if text_jsonl.exists():
            print(f"Loading text map from: {text_jsonl}")
            text_map = load_text_map(text_jsonl, id_key=args.id_key, text_key=args.text_key)
            print(f"Loaded text rows: {len(text_map)}")
        else:
            print(f"Warning: text_jsonl not found: {text_jsonl}")

    docs: dict[str, Document] = {}
    index_to_docstore_id: dict[int, str] = {}
    running = 0

    for shard in shards:
        emb_path = emb_root / shard["embedding_file"]
        meta_path = emb_root / shard["metadata_file"]
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        embs = np.load(emb_path)
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)
        metas = load_metadata(meta_path)

        if embs.shape[0] != len(metas):
            raise ValueError(
                f"Row mismatch in shard {shard['shard_id']}: vectors={embs.shape[0]} metadata={len(metas)}"
            )

        index.add(embs)

        for i, meta in enumerate(metas):
            chunk_id = str(meta.get(args.id_key, "")).strip()
            if not chunk_id:
                chunk_id = f"row_{running + i}"

            page_content = text_map.get(chunk_id, "")
            if not page_content:
                page_content = str(meta.get("title", "") or "")

            safe_meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}
            docs[chunk_id] = Document(page_content=page_content, metadata=safe_meta)
            index_to_docstore_id[running + i] = chunk_id

        running += embs.shape[0]
        print(f"Added shard {shard['shard_id']}: rows={embs.shape[0]} | total={running}")

    if args.index_dir.exists() and any(args.index_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Index dir is not empty: {args.index_dir}. Use --overwrite.")

    args.index_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS(
        embedding_function=NoOpEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(docs),
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=distance_strategy,
        normalize_L2=False,
    )
    vs.save_local(str(args.index_dir))

    info = {
        "source_manifest": str(args.manifest),
        "index_dir": str(args.index_dir),
        "rows": running,
        "embedding_dim": dim,
        "normalize": normalize,
        "distance": "ip" if normalize else "l2",
        "text_jsonl": str(args.text_jsonl) if args.text_jsonl else "",
    }
    with (args.index_dir / "index_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved LangChain FAISS at: {args.index_dir}")
    print(f"Rows: {running} | dim: {dim}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LangChain FAISS from Kaggle embedding exports")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--emb-root", type=Path, default=Path("."))
    parser.add_argument("--text-jsonl", type=Path, default=Path("data/datahuggingface/corpus_important_docs_chunks.jsonl"))
    parser.add_argument("--id-key", type=str, default="chunk_id")
    parser.add_argument("--text-key", type=str, default="chunk_text")
    parser.add_argument("--index-dir", type=Path, default=Path("data/faiss/law_chunks_e5_base"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_index(args)


if __name__ == "__main__":
    raise SystemExit(main())
