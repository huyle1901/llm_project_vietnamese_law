#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Regex strings are ASCII-only with unicode escapes to avoid terminal/file encoding issues on Windows.
LEGAL_SPLIT_REGEX = [
    r"(?im)^\s*(?:phan|ph\u1ea7n)\s+[ivxlcdm]+",
    r"(?im)^\s*(?:chuong|ch\u01b0\u01a1ng)\s+[ivxlcdm]+",
    r"(?im)^\s*(?:muc|m\u1ee5c)\s+\d+",
    r"(?im)^\s*(?:dieu|\u0111ieu|\u0111i\u1ec1u)\s+\d+",
]
ARTICLE_SPLIT_RE = re.compile(r"(?im)(?=^\s*(?:dieu|\u0111ieu|\u0111i\u1ec1u)\s+\d+)")


@dataclass
class DocStats:
    path: str
    chars: int
    words: int
    non_empty_lines: int
    avg_non_empty_line_len: float
    article_segments: int
    article_seg_mean_chars: float


def q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    weight = idx - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "cp1258", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> tuple[int, int]:
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        txt_count = sum(1 for n in names if n.lower().endswith(".txt"))
        existing_txt = sum(1 for p in extract_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
        if existing_txt == txt_count and txt_count > 0 and not force:
            return txt_count, 0

        # Rebuild cleanly so the directory reflects exactly the archive content.
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        collision_count = 0
        used_casefold_paths: dict[str, int] = {}

        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue

            rel = Path(name)
            key = str(rel).replace("\\", "/").casefold()
            if key in used_casefold_paths:
                collision_count += 1
                idx = used_casefold_paths[key]
                used_casefold_paths[key] += 1
                rel = rel.with_name(f"{rel.stem}__dup{idx}{rel.suffix}")
            else:
                used_casefold_paths[key] = 1

            dest = extract_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    return txt_count, collision_count


def iter_txt_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".txt":
            yield p


def split_article_segments(text: str) -> list[str]:
    folded = fold_text(text)
    parts = [p.strip() for p in ARTICLE_SPLIT_RE.split(folded) if p.strip()]
    if len(parts) <= 1:
        return []
    return parts


def profile_docs(data_root: Path, max_preview: int = 10) -> tuple[dict, list[DocStats]]:
    txt_files = sorted(iter_txt_files(data_root))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in: {data_root}")

    doc_stats: list[DocStats] = []
    total_decode_fallback = 0

    for path in txt_files:
        raw = path.read_bytes()
        text = decode_text(raw)
        if "\ufffd" in text:
            total_decode_fallback += 1

        words = len(re.findall(r"\S+", text))
        lines = text.splitlines()
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        non_empty_len = [len(ln) for ln in non_empty]
        avg_line = statistics.mean(non_empty_len) if non_empty_len else 0.0

        article_segments = split_article_segments(text)
        article_lengths = [len(seg) for seg in article_segments]

        doc_stats.append(
            DocStats(
                path=str(path),
                chars=len(text),
                words=words,
                non_empty_lines=len(non_empty),
                avg_non_empty_line_len=avg_line,
                article_segments=len(article_segments),
                article_seg_mean_chars=statistics.mean(article_lengths) if article_lengths else 0.0,
            )
        )

    chars = [d.chars for d in doc_stats]
    words = [d.words for d in doc_stats]
    non_empty_lines = [d.non_empty_lines for d in doc_stats]
    avg_line_len = [d.avg_non_empty_line_len for d in doc_stats]
    article_seg_mean_chars = [d.article_seg_mean_chars for d in doc_stats if d.article_seg_mean_chars > 0]

    est_tokens_by_char = [c / 4.0 for c in chars]
    est_tokens_by_word = [w * 1.2 for w in words]

    summary = {
        "files": {
            "txt_count": len(doc_stats),
            "decode_with_replacement_count": total_decode_fallback,
            "sample_paths": [d.path for d in doc_stats[:max_preview]],
        },
        "length_distribution": {
            "chars": {
                "min": min(chars),
                "p25": q(chars, 0.25),
                "p50": q(chars, 0.50),
                "p75": q(chars, 0.75),
                "p90": q(chars, 0.90),
                "max": max(chars),
                "mean": statistics.mean(chars),
            },
            "words": {
                "min": min(words),
                "p25": q(words, 0.25),
                "p50": q(words, 0.50),
                "p75": q(words, 0.75),
                "p90": q(words, 0.90),
                "max": max(words),
                "mean": statistics.mean(words),
            },
            "non_empty_lines": {
                "min": min(non_empty_lines),
                "p25": q(non_empty_lines, 0.25),
                "p50": q(non_empty_lines, 0.50),
                "p75": q(non_empty_lines, 0.75),
                "p90": q(non_empty_lines, 0.90),
                "max": max(non_empty_lines),
                "mean": statistics.mean(non_empty_lines),
            },
            "avg_non_empty_line_len": {
                "min": min(avg_line_len),
                "p25": q(avg_line_len, 0.25),
                "p50": q(avg_line_len, 0.50),
                "p75": q(avg_line_len, 0.75),
                "p90": q(avg_line_len, 0.90),
                "max": max(avg_line_len),
                "mean": statistics.mean(avg_line_len),
            },
            "estimated_tokens_per_doc": {
                "char_based_mean": statistics.mean(est_tokens_by_char),
                "char_based_p50": q(est_tokens_by_char, 0.50),
                "char_based_p90": q(est_tokens_by_char, 0.90),
                "word_based_mean": statistics.mean(est_tokens_by_word),
            },
        },
        "article_structure": {
            "docs_with_detected_articles": sum(1 for d in doc_stats if d.article_segments > 0),
            "article_segment_mean_chars": statistics.mean(article_seg_mean_chars) if article_seg_mean_chars else 0.0,
            "article_segment_p50_chars": q(article_seg_mean_chars, 0.50),
            "article_segment_p75_chars": q(article_seg_mean_chars, 0.75),
        },
    }
    return summary, doc_stats


def build_chunk_recommendation(summary: dict) -> dict:
    p50_chars = summary["length_distribution"]["chars"]["p50"]
    p75_chars = summary["length_distribution"]["chars"]["p75"]
    seg_p50 = summary["article_structure"]["article_segment_p50_chars"]
    seg_p75 = summary["article_structure"]["article_segment_p75_chars"]

    # Heuristic for legal texts: one meaningful legal section per chunk when possible.
    if seg_p50 > 0:
        base = int(min(1800, max(900, seg_p50 + 250)))
    else:
        base = int(min(1800, max(900, p50_chars * 0.18)))

    if p75_chars > 30000:
        base = min(1700, base)

    overlap = int(max(120, min(360, base * 0.18)))
    min_chunk = int(max(280, base * 0.35))

    return {
        "strategy": "structure_aware_recursive_chunking",
        "why": [
            "Legal text has clear hierarchy (phan/chuong/muc/dieu); split by structure first.",
            "Fallback recursive split controls very long sections and keeps chunks retrieval-friendly.",
            "Overlap preserves references across adjacent legal sections.",
        ],
        "recommended_config": {
            "chunk_size_chars": base,
            "chunk_overlap_chars": overlap,
            "min_chunk_chars": min_chunk,
            "length_function": "len",
            "keep_separator": True,
            "pre_split_regex": LEGAL_SPLIT_REGEX,
            "recursive_separators": ["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        },
        "retrieval_hint": {
            "top_k_start": 8,
            "rerank_top_n": 20,
            "notes": "Start with top_k=8 for recall, then rerank to reduce noisy contexts.",
        },
        "observed_data_points": {
            "doc_chars_p50": p50_chars,
            "doc_chars_p75": p75_chars,
            "article_seg_chars_p50": seg_p50,
            "article_seg_chars_p75": seg_p75,
        },
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Extract Dataset.zip and profile legal corpus for chunking decisions."
    )
    parser.add_argument("--zip-path", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis/dataset_profile.json"))
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args()

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {args.zip_path}")

    print(f"[1/3] Extracting: {args.zip_path} -> {args.extract_dir}")
    txt_count, collision_count = extract_zip(args.zip_path, args.extract_dir, force=args.force_reextract)
    print(f"      TXT files in archive: {txt_count}")
    print(f"      Case-insensitive filename collisions handled: {collision_count}")

    print("[2/3] Profiling extracted corpus...")
    summary, _ = profile_docs(args.extract_dir)

    recommendation = build_chunk_recommendation(summary)
    payload = {
        "summary": summary,
        "chunking_recommendation": recommendation,
    }

    print(f"[3/3] Writing profile report: {args.output_json}")
    save_json(args.output_json, payload)

    cfg = recommendation["recommended_config"]
    print("\nSuggested chunk config:")
    print(
        f"- chunk_size_chars={cfg['chunk_size_chars']}, "
        f"chunk_overlap_chars={cfg['chunk_overlap_chars']}, min_chunk_chars={cfg['min_chunk_chars']}"
    )
    print("- strategy=structure_aware_recursive_chunking")
    print(f"- pre_split_regex={cfg['pre_split_regex']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
