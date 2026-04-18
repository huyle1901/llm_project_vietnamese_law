from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def maybe_fix_mojibake(s: str) -> str:
    s = (s or '').strip()
    if not s:
        return s

    suspicious = any(ch in s for ch in ('Ã', 'Ä', 'Â', 'Ê', 'Ô', 'Ð', '\x90')) or ('Ä\x90' in s)
    if not suspicious:
        return s

    try:
        fixed = s.encode('latin1', errors='strict').decode('utf-8', errors='strict')
        return fixed.strip() if fixed.strip() else s
    except Exception:
        return s


def canonical_doc_no(raw: str) -> str:
    s = maybe_fix_mojibake(raw)
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('–', '-').replace('—', '-').replace('_', '-')
    s = re.sub(r'\s*([/\-])\s*', r'\1', s)
    s = re.sub(r'\s+', '', s)

    s = s.replace('Đ', 'D').replace('đ', 'd')
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = unicodedata.normalize('NFC', s)

    s = s.upper()
    s = re.sub(r'[^A-Z0-9/\-]', '', s)
    return s


def compact_doc_no(canonical: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', canonical)


def load_important_docs(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            raw = re.sub(r'^\s*\d+[\.)\-:]\s*', '', raw)
            canonical = canonical_doc_no(raw)
            if not canonical:
                continue

            if canonical in seen:
                continue
            seen.add(canonical)

            out.append(
                {
                    'raw': raw,
                    'fixed': maybe_fix_mojibake(raw),
                    'canonical': canonical,
                    'compact': compact_doc_no(canonical),
                }
            )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Filter metadata/content by important document numbers.')
    p.add_argument('--important-docs', type=Path, default=Path('important_docs.txt'))
    p.add_argument('--metadata', type=Path, default=Path('data/datahuggingface/metadata_full.jsonl'))
    p.add_argument('--content', type=Path, default=Path('data/datahuggingface/content_full.jsonl'))
    p.add_argument('--out-metadata', type=Path, default=Path('data/datahuggingface/metadata_important_docs.jsonl'))
    p.add_argument('--out-content', type=Path, default=Path('data/datahuggingface/content_important_docs.jsonl'))
    p.add_argument('--out-missing', type=Path, default=Path('data/datahuggingface/important_docs_missing.txt'))
    p.add_argument('--out-summary', type=Path, default=Path('data/datahuggingface/important_docs_summary.json'))
    p.add_argument('--dedup-content-by-id', action='store_true', default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    wanted = load_important_docs(args.important_docs)
    wanted_canonical = {x['canonical'] for x in wanted}
    wanted_compact = {x['compact'] for x in wanted}

    matched_wanted: set[str] = set()
    filtered_meta: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    meta_rows = 0
    for row in read_jsonl(args.metadata):
        meta_rows += 1
        doc_no = str(row.get('so_ky_hieu') or row.get('document_number') or '').strip()
        can = canonical_doc_no(doc_no)
        cmp_ = compact_doc_no(can)

        is_match = can in wanted_canonical or cmp_ in wanted_compact
        if not is_match:
            continue

        filtered_meta.append(row)
        try:
            selected_ids.add(int(row['id']))
        except Exception:
            pass

        if can in wanted_canonical:
            matched_wanted.add(can)

        if cmp_ in wanted_compact:
            for w in wanted:
                if w['compact'] == cmp_:
                    matched_wanted.add(w['canonical'])

    write_jsonl(args.out_metadata, filtered_meta)

    filtered_content: list[dict[str, Any]] = []
    content_rows = 0
    content_kept = 0
    dup_skipped = 0
    seen_content_ids: set[int] = set()

    for row in read_jsonl(args.content):
        content_rows += 1
        try:
            rid = int(row['id'])
        except Exception:
            continue

        if rid not in selected_ids:
            continue

        if args.dedup_content_by_id and rid in seen_content_ids:
            dup_skipped += 1
            continue

        seen_content_ids.add(rid)
        filtered_content.append(row)
        content_kept += 1

    write_jsonl(args.out_content, filtered_content)

    ids_in_meta = {int(r['id']) for r in filtered_meta if 'id' in r}
    ids_in_content = {int(r['id']) for r in filtered_content if 'id' in r}
    ids_missing_content = sorted(ids_in_meta - ids_in_content)

    missing_docs = [w for w in wanted if w['canonical'] not in matched_wanted]

    args.out_missing.parent.mkdir(parents=True, exist_ok=True)
    with args.out_missing.open('w', encoding='utf-8') as f:
        f.write('=== Docs in important_docs but NOT found in metadata ===\n')
        for m in missing_docs:
            f.write(f"- raw={m['raw']} | fixed={m['fixed']} | canonical={m['canonical']}\n")

        f.write('\n=== IDs found in metadata_important but missing in content_important ===\n')
        for rid in ids_missing_content:
            f.write(f'- {rid}\n')

    summary = {
        'important_total': len(wanted),
        'important_matched': len(matched_wanted),
        'important_missing': len(missing_docs),
        'metadata_total_rows': meta_rows,
        'metadata_filtered_rows': len(filtered_meta),
        'metadata_filtered_unique_ids': len(ids_in_meta),
        'content_total_rows': content_rows,
        'content_filtered_rows': content_kept,
        'content_filtered_unique_ids': len(ids_in_content),
        'content_duplicate_rows_skipped': dup_skipped,
        'ids_missing_in_content_after_meta_match': len(ids_missing_content),
        'out_metadata': str(args.out_metadata),
        'out_content': str(args.out_content),
        'out_missing': str(args.out_missing),
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
