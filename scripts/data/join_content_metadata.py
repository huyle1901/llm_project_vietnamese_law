#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


DEFAULT_CONTENT = Path("data/datahuggingface/content_important_docs.jsonl")
DEFAULT_METADATA = Path("data/datahuggingface/metadata_important_docs.jsonl")
DEFAULT_OUT = Path("data/datahuggingface/corpus_important_docs.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join legal content JSONL with metadata JSONL by id.")
    parser.add_argument("--content", type=Path, default=DEFAULT_CONTENT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    if not args.content.exists():
        raise FileNotFoundError(f"Content JSONL not found: {args.content}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {args.metadata}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    metadata_map: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(args.metadata):
        rid = int(row["id"])
        metadata_map[rid] = row

    matched = 0
    missing_meta = 0

    with args.out.open("w", encoding="utf-8") as out:
        for row in read_jsonl(args.content):
            rid = int(row["id"])
            meta = metadata_map.get(rid)

            if meta is None:
                missing_meta += 1
                continue

            merged = {
                "id": rid,
                "content": row.get("content", ""),
                "document_number": meta.get("document_number") or meta.get("so_ky_hieu"),
                "title": meta.get("title"),
                "url": meta.get("url") or meta.get("nguon_thu_thap"),
                "legal_type": meta.get("legal_type") or meta.get("loai_van_ban"),
                "legal_sectors": meta.get("legal_sectors") or meta.get("linh_vuc"),
                "issuing_authority": meta.get("issuing_authority") or meta.get("co_quan_ban_hanh"),
                "issuance_date": meta.get("issuance_date") or meta.get("ngay_ban_hanh"),
                "signers": meta.get("signers") or meta.get("nguoi_ky"),
            }

            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
            matched += 1

    print(f"matched={matched}, missing_meta={missing_meta}, out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
