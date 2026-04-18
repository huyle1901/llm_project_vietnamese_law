# scripts/data/chunk_legal_corpus.py
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ARTICLE_RE = re.compile(r"(?im)^Điều\s+(\d+)\s*[\.:]?\s*(.*)$")
APPENDIX_RE = re.compile(r"(?im)^PHỤ\s+LỤC(?:\s+[IVXLC]+)?\b.*$")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_tokens(text: str) -> int:
    words = text.split()
    return int(math.ceil(len(words) * 1.33))


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    out: list[str] = []
    for b in blocks:
        parts = re.split(r"(?<=[\.\!\?;:])\s+", b)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out


def overlap_tail_words(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    overlap_words = max(1, int(overlap_tokens / 1.33))
    return " ".join(words[-overlap_words:])


def chunk_text_with_overlap(
    text: str, target_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if approx_tokens(text) <= max_tokens:
        return [text]

    pieces = split_sentences(text)
    if not pieces:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            chunks.append(" ".join(current).strip())

    for piece in pieces:
        if not current:
            current = [piece]
            continue

        candidate = " ".join(current + [piece]).strip()
        if approx_tokens(candidate) <= target_tokens:
            current.append(piece)
        else:
            flush()
            prev = chunks[-1] if chunks else ""
            ov = overlap_tail_words(prev, overlap_tokens)
            current = [ov, piece] if ov else [piece]

    flush()

    # hard-guard max_tokens
    final_chunks: list[str] = []
    for ch in chunks:
        if approx_tokens(ch) <= max_tokens:
            final_chunks.append(ch)
        else:
            words = ch.split()
            win = max(300, int(max_tokens / 1.33))
            step = max(200, int((max_tokens - overlap_tokens) / 1.33))
            i = 0
            while i < len(words):
                part = " ".join(words[i : i + win]).strip()
                if part:
                    final_chunks.append(part)
                if i + win >= len(words):
                    break
                i += step

    return [c for c in final_chunks if c.strip()]


def split_appendix_table_chunks(text: str, max_lines: int = 35) -> list[str]:
    lines = [ln.rstrip() for ln in text.split("\n")]
    if not lines:
        return []

    table_lines = [ln for ln in lines if "|" in ln]
    if len(table_lines) < 10:
        return [text.strip()]

    appendix_title = lines[0].strip() if lines and "PHỤ LỤC" in lines[0].upper() else ""

    header = []
    for ln in lines:
        if "|" in ln:
            header.append(ln)
            if len(header) >= 3:
                break
    header_text = "\n".join(header).strip()

    prefix = "\n".join([x for x in [appendix_title, header_text] if x]).strip()

    chunks = []
    current = []
    for ln in lines:
        if "|" not in ln:
            continue
        current.append(ln)
        if len(current) >= max_lines:
            body = "\n".join(current).strip()
            chunks.append(f"{prefix}\n{body}".strip() if prefix else body)
            current = []

    if current:
        body = "\n".join(current).strip()
        chunks.append(f"{prefix}\n{body}".strip() if prefix else body)

    return [c for c in chunks if c.strip()]

def merge_short_tail(chunks: list[str], min_tokens: int = 80) -> list[str]:
    if not chunks:
        return chunks
    if len(chunks) == 1:
        return chunks

    last = chunks[-1]
    if approx_tokens(last) < min_tokens:
        chunks[-2] = (chunks[-2] + "\n" + last).strip()
        chunks.pop()
    return chunks

def split_doc_sections(content: str) -> list[dict[str, Any]]:
    markers = []

    for m in ARTICLE_RE.finditer(content):
        markers.append(
            {
                "start": m.start(),
                "kind": "article",
                "article_no": int(m.group(1)),
                "heading": m.group(0).strip(),
            }
        )

    for m in APPENDIX_RE.finditer(content):
        markers.append(
            {
                "start": m.start(),
                "kind": "appendix",
                "article_no": None,
                "heading": m.group(0).strip(),
            }
        )

    markers.sort(key=lambda x: x["start"])

    if not markers:
        return [{"section_type": "body", "article_no": None, "heading": "", "text": content.strip()}]

    sections = []
    if markers[0]["start"] > 0:
        pre = content[: markers[0]["start"]].strip()
        if pre:
            sections.append({"section_type": "header", "article_no": None, "heading": "", "text": pre})

    for i, mk in enumerate(markers):
        s = mk["start"]
        e = markers[i + 1]["start"] if i + 1 < len(markers) else len(content)
        sec_text = content[s:e].strip()
        if not sec_text:
            continue
        sections.append(
            {
                "section_type": mk["kind"],
                "article_no": mk["article_no"],
                "heading": mk["heading"],
                "text": sec_text,
            }
        )

    return sections


def split_article_into_clauses(article_text: str, article_heading: str) -> list[dict[str, Any]]:
    first_nl = article_text.find("\n")
    body = article_text[first_nl + 1 :].strip() if first_nl != -1 else ""

    clause_re = re.compile(r"(?m)^\s*(\d+)\.\s+")
    ms = list(clause_re.finditer(body))
    if not ms:
        return [{"clause_no": None, "text": article_text.strip()}]

    opening = body[: ms[0].start()].strip()
    parts = []

    for i, m in enumerate(ms):
        cno = int(m.group(1))
        s = m.start()
        e = ms[i + 1].start() if i + 1 < len(ms) else len(body)
        clause_text = body[s:e].strip()

        composed = [article_heading]
        if i == 0 and opening:
            composed.append(opening)
        composed.append(clause_text)

        parts.append({"clause_no": cno, "text": "\n".join(composed).strip()})

    return parts


def make_chunk_record(
    doc: dict[str, Any],
    chunk_text: str,
    chunk_idx: int,
    section_type: str,
    article_no: int | None,
    clause_no: int | None,
) -> dict[str, Any]:
    doc_id = int(doc["id"])
    return {
        "id": doc_id,
        "chunk_id": f"{doc_id}_{chunk_idx:04d}",
        "chunk_text": chunk_text,
        "word_count": len(chunk_text.split()),
        "token_estimate": approx_tokens(chunk_text),
        "section_type": section_type,
        "article_no": article_no,
        "clause_no": clause_no,
        "document_number": doc.get("document_number"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "legal_type": doc.get("legal_type"),
        "legal_sectors": doc.get("legal_sectors"),
        "issuing_authority": doc.get("issuing_authority"),
        "issuance_date": doc.get("issuance_date"),
        "signers": doc.get("signers"),
    }


def chunk_one_document(
    doc: dict[str, Any],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    content = normalize_text(str(doc.get("content", "")))
    if not content or len(content) < min_chars:
        return []

    sections = split_doc_sections(content)
    out = []
    idx = 0

    for sec in sections:
        sec_type = sec["section_type"]
        article_no = sec.get("article_no")
        heading = sec.get("heading", "")
        text = sec["text"]

        if sec_type == "article":
            clause_parts = split_article_into_clauses(text, heading)
            for part in clause_parts:
                chunks = chunk_text_with_overlap(
                    part["text"], target_tokens, max_tokens, overlap_tokens
                )
                for ct in chunks:
                    if len(ct.strip()) < min_chars:
                        continue
                    out.append(
                        make_chunk_record(
                            doc, ct, idx, sec_type, article_no, part.get("clause_no")
                        )
                    )
                    idx += 1

        elif sec_type == "appendix":
            chunks = split_appendix_table_chunks(text, max_lines=35)
            if len(chunks) == 1 and approx_tokens(chunks[0]) > max_tokens:
                chunks = chunk_text_with_overlap(
                    chunks[0], target_tokens=target_tokens, max_tokens=max_tokens, overlap_tokens=0
                )

            chunks = merge_short_tail(chunks, min_tokens=80)
            for ct in chunks:
                if len(ct.strip()) < min_chars:
                    continue
                out.append(make_chunk_record(doc, ct, idx, sec_type, article_no, None))
                idx += 1

        else:
            chunks = chunk_text_with_overlap(text, target_tokens, max_tokens, overlap_tokens)
            for ct in chunks:
                if len(ct.strip()) < min_chars:
                    continue
                out.append(make_chunk_record(doc, ct, idx, sec_type, article_no, None))
                idx += 1

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/datahuggingface/content_important_docs.jsonl")
    parser.add_argument("--output", type=str, default="data/datahuggingface/content_important_docs_chunks.jsonl")
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=40)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    all_chunks = []
    docs = 0
    skipped = 0

    for doc in read_jsonl(in_path):
        docs += 1
        chunks = chunk_one_document(
            doc,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            min_chars=args.min_chars,
        )
        if not chunks:
            skipped += 1
            continue
        all_chunks.extend(chunks)

    write_jsonl(out_path, all_chunks)
    print(f"docs={docs}")
    print(f"skipped={skipped}")
    print(f"chunks={len(all_chunks)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
