#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

q = "Kể từ 1/11/2025, có đúng là những giao dịch nộp/rút/chuyển tiền trong một ngày đạt từ 400 triệu đồng trở lên—kể cả khi người thực hiện không có tài khoản—vẫn phải được tổ chức thực hiện thủ tục nhận biết khách hàng không?"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick test search against OpenSearch index (BM25)."
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
        help="OpenSearch index name",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=q,
        help="Query text",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Top-k results",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="chunk_text^3,title^2,document_number^4,filename",
        help="Comma-separated fields for multi_match",
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
        "--verify-certs",
        action="store_true",
        default=False,
        help="Verify TLS certs (for HTTPS production)",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=280,
        help="Number of characters to print for chunk_text snippet",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default="",
        help="Optional path to dump full JSON response",
    )
    return parser.parse_args()


def get_client(args: argparse.Namespace):
    from opensearchpy import OpenSearch

    kwargs: dict[str, Any] = {
        "hosts": [args.opensearch_url],
        "verify_certs": args.verify_certs,
        "ssl_show_warn": False,
        "timeout": 60,
    }
    if args.username and args.password:
        kwargs["http_auth"] = (args.username, args.password)
    return OpenSearch(**kwargs)


def build_query_body(args: argparse.Namespace) -> dict[str, Any]:
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    return {
        "size": args.size,
        "query": {
            "multi_match": {
                "query": args.query,
                "fields": fields,
                "type": "best_fields",
            }
        },
        "_source": [
            "chunk_id",
            "chunk_text",
            "filename",
            "source_file",
            "title",
            "document_number",
            "section_type",
            "article_no",
            "clause_no",
            "legal_type",
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    args = parse_args()
    client = get_client(args)

    body = build_query_body(args)
    res = client.search(index=args.index_name, body=body)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"Saved full response to: {args.dump_json}")

    total = res.get("hits", {}).get("total", {})
    total_value = total.get("value", 0) if isinstance(total, dict) else total
    hits = res.get("hits", {}).get("hits", [])

    print(f"OpenSearch: {args.opensearch_url}")
    print(f"Index: {args.index_name}")
    print(f"Query: {args.query}")
    print(f"Total hits: {total_value}")
    print(f"Returned: {len(hits)}")

    for i, h in enumerate(hits, 1):
        src = h.get("_source", {})
        score = h.get("_score")
        text = str(src.get("chunk_text") or "")
        text = " ".join(text.split())
        if len(text) > args.snippet_chars:
            text = text[: args.snippet_chars] + "..."

        print("\n" + "=" * 90)
        print(f"#{i} | score={score}")
        print(f"chunk_id={src.get('chunk_id')} | file={src.get('filename')} | source={src.get('source_file')}")
        print(
            "doc_no="
            f"{src.get('document_number')} | section={src.get('section_type')} "
            f"| article={src.get('article_no')} | clause={src.get('clause_no')}"
        )
        print(f"title={src.get('title')}")
        print(f"snippet={text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
