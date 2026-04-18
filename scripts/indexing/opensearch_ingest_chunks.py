#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk ingest chunk JSONL into OpenSearch.")
    parser.add_argument(
        "--input",
        "--input-jsonl",
        dest="input_jsonl",
        type=Path,
        default=Path("data/datahuggingface/corpus_important_docs_chunks.jsonl"),
        help="Path to chunk JSONL file",
    )
    parser.add_argument(
        "--opensearch-url",
        type=str,
        default="http://localhost:9200",
        help="OpenSearch endpoint",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="law_chunks_bm25",
        help="Target OpenSearch index",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="OpenSearch username (optional)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="OpenSearch password (optional)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate index before ingest",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Bulk index batch size",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="For testing: index at most N docs (0 = all)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="OpenSearch request timeout (seconds)",
    )
    parser.add_argument(
        "--verify-certs",
        action="store_true",
        default=False,
        help="Verify TLS certs (set true for HTTPS prod)",
    )
    return parser.parse_args()


def get_client(args: argparse.Namespace):
    from opensearchpy import OpenSearch

    kwargs: dict[str, Any] = {
        "hosts": [args.opensearch_url],
        "verify_certs": args.verify_certs,
        "ssl_show_warn": False,
        "timeout": args.request_timeout,
    }
    if args.username and args.password:
        kwargs["http_auth"] = (args.username, args.password)
    return OpenSearch(**kwargs)


def build_index_body() -> dict[str, Any]:
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "long"},
                "chunk_id": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "section_type": {"type": "keyword"},
                "article_no": {"type": "integer"},
                "clause_no": {"type": "integer"},
                "document_number": {"type": "keyword"},
                "title": {"type": "text"},
                "url": {"type": "keyword"},
                "legal_type": {"type": "keyword"},
                "legal_sectors": {"type": "keyword"},
                "issuing_authority": {"type": "keyword"},
                "issuance_date": {
                    "type": "date",
                    "format": "dd/MM/yyyy||strict_date_optional_time",
                },
                "signers": {"type": "keyword"},
                "word_count": {"type": "integer"},
                "token_estimate": {"type": "integer"},
            }
        },
    }


def ensure_index(client: Any, index_name: str, recreate: bool) -> None:
    exists = client.indices.exists(index=index_name)
    if recreate and exists:
        client.indices.delete(index=index_name)
        exists = False
    if not exists:
        client.indices.create(index=index_name, body=build_index_body())


def iter_jsonl(path: Path, max_docs: int = 0) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield row
            count += 1
            if max_docs > 0 and count >= max_docs:
                return


def iter_actions(index_name: str, rows: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for row in rows:
        doc_id = row.get("chunk_id")
        if not doc_id:
            continue
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc_id,
            "_source": row,
        }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    args = parse_args()
    try:
        from opensearchpy import helpers
    except Exception as e:
        raise RuntimeError("Missing dependency: opensearchpy") from e

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    print(f"Input JSONL: {args.input_jsonl}")
    print(f"OpenSearch: {args.opensearch_url}")
    print(f"Index: {args.index_name}")
    print(f"Recreate index: {args.recreate_index}")
    print(f"Batch size: {args.batch_size}")
    if args.max_docs > 0:
        print(f"Max docs: {args.max_docs}")

    client = get_client(args)
    ensure_index(client, args.index_name, recreate=args.recreate_index)

    rows = iter_jsonl(args.input_jsonl, max_docs=args.max_docs)
    actions = iter_actions(args.index_name, rows)
    ok, errors = helpers.bulk(
        client,
        actions,
        chunk_size=args.batch_size,
        stats_only=True,
        request_timeout=args.request_timeout,
    )

    print(f"Indexed docs: {ok}")
    print(f"Bulk errors: {errors}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
