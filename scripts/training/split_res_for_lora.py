#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý Việt Nam. "
    "Chỉ trả lời dựa trên thông tin được cung cấp trong bộ dữ liệu huấn luyện."
)


def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    if any(x in text for x in ("Ã", "Ä", "á»", "áº", "Ä‘", "Å")):
        try:
            return text.encode("latin1").decode("utf-8")
        except Exception:
            return text
    return text


def fold_text(text: str) -> str:
    text = fix_mojibake(str(text))
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def canonical_doc_id(text: str) -> str:
    t = fold_text(str(text)).upper()
    return re.sub(r"[^A-Z0-9]", "", t)


def resolve_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = list(df.columns)
    norm = {c: fold_text(str(c)) for c in cols}

    question_col = next((c for c, n in norm.items() if "cau hoi" in n), None)
    if question_col is None:
        raise ValueError(f"Cannot find question column. Columns={cols}")

    answer_col = None
    for c, n in norm.items():
        if "tra loi" not in n:
            continue
        if any(bad in n for bad in ("urax", "dung khong", "reference")):
            continue
        answer_col = c
        break
    if answer_col is None:
        raise ValueError(f"Cannot find ground-truth answer column. Columns={cols}")

    vbpl_col = None
    for c, n in norm.items():
        if ("so hieu" in n and ("vbpl" in n or "van ban" in n)) or ("trich xuat" in n):
            vbpl_col = c
            break
    if vbpl_col is None:
        raise ValueError(f"Cannot find VBPL column. Columns={cols}")

    return question_col, answer_col, vbpl_col


def parse_vbpl_tokens(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = fix_mojibake(str(value)).strip()
    if not s:
        return []
    parts = re.split(r"[\n,;|]+", s)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if fold_text(p) == "khong trich xuat duoc":
            continue
        out.append(p)
    return out


def load_missing_doc_set(path: Path | None) -> set[str]:
    out: set[str] = set()
    if path is None or not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = fix_mojibake(raw).strip()
        if not line or line.startswith("==="):
            continue
        if "canonical=" in line:
            can = line.split("canonical=", 1)[1].strip()
            if can:
                out.add(canonical_doc_id(can))
            continue
        out.add(canonical_doc_id(line))
    out.discard("")
    return out


def sample_by_group(df: pd.DataFrame, n: int, seed: int, group_col: str) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_rand"] = rng.random(len(work))
    work = work.sort_values("_rand").drop(columns=["_rand"])

    group_sizes = work[group_col].value_counts().to_dict()
    raw = {g: (n * size / len(work)) for g, size in group_sizes.items()}
    take = {g: int(np.floor(v)) for g, v in raw.items()}
    assigned = sum(take.values())

    # Allocate leftover slots to largest fractional remainders.
    leftovers = sorted(((raw[g] - take[g], g) for g in group_sizes), reverse=True)
    i = 0
    while assigned < n and i < len(leftovers):
        g = leftovers[i][1]
        if take[g] < group_sizes[g]:
            take[g] += 1
            assigned += 1
        i += 1

    parts: list[pd.DataFrame] = []
    for g, k in take.items():
        if k <= 0:
            continue
        gdf = work[work[group_col] == g]
        parts.append(gdf.head(k))

    out = pd.concat(parts, axis=0).sort_values("row_index")
    if len(out) > n:
        out = out.head(n)
    elif len(out) < n:
        need = n - len(out)
        rest = work[~work["row_index"].isin(out["row_index"])]
        out = pd.concat([out, rest.head(need)], axis=0).sort_values("row_index")
    return out


def build_strat_key(df: pd.DataFrame) -> pd.Series:
    vbpl_bin = np.where(df["vbpl_count"] <= 1, "vbpl1", "vbpl2p")
    qlen = df["question"].fillna("").astype(str).str.len()
    qlen_bin = pd.cut(qlen, bins=[-1, 80, 160, 10_000], labels=["q_short", "q_mid", "q_long"])
    qlen_bin = qlen_bin.astype(str).replace("nan", "q_mid")
    return pd.Series(vbpl_bin, index=df.index) + "|" + qlen_bin


def build_hf_records(df: pd.DataFrame, system_prompt: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload: dict[str, Any] = {
            "id": f"res_{int(row['row_index'])}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Cau hoi:\n{row['question']}"},
                {"role": "assistant", "content": str(row["ground_truth"])},
            ],
            "meta": {
                "row_index": int(row["row_index"]),
                "vbpl_extracted": str(row["vbpl_extracted"]),
                "bucket": str(row["bucket"]),
                "split": str(row["split"]),
            },
        }
        records.append(payload)
    return records


def export_hf_jsonl(df: pd.DataFrame, out_jsonl: Path, system_prompt: str) -> None:
    records = build_hf_records(df, system_prompt)
    out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


def export_evaluate_csv(df: pd.DataFrame, out_csv: Path) -> None:
    cols = [
        "row_index",
        "question",
        "ground_truth",
        "vbpl_extracted",
        "vbpl_count",
        "bucket",
        "split",
    ]
    df[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split RES.xlsx into training_lora + evaluate datasets.")
    p.add_argument("--res-xlsx", type=Path, default=Path("RES.xlsx"))
    p.add_argument("--sheet", type=str, default=None)
    p.add_argument("--missing-docs", type=Path, default=Path("data/datahuggingface/important_docs_missing.txt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate-n", type=int, default=150, help="Answerable rows for final metrics evaluation.")
    p.add_argument("--ce-val-n", type=int, default=150, help="Validation rows for CE/SFT training.")
    p.add_argument("--drop-duplicate-question", action="store_true")
    p.add_argument("--training-dir", type=Path, default=Path("data/training_lora"))
    p.add_argument("--evaluate-dir", type=Path, default=Path("data/evaluate"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.res_xlsx.exists():
        raise FileNotFoundError(f"RES file not found: {args.res_xlsx}")

    xls = pd.ExcelFile(args.res_xlsx)
    sheet = args.sheet if args.sheet else xls.sheet_names[0]
    df = xls.parse(sheet)
    q_col, a_col, vbpl_col = resolve_columns(df)

    raw = pd.DataFrame(
        {
            "row_index": df.index.astype(int),
            "question": df[q_col].map(lambda x: fix_mojibake(str(x)).strip()),
            "ground_truth": df[a_col].map(lambda x: fix_mojibake(str(x)).strip()),
            "vbpl_extracted": df[vbpl_col].map(lambda x: fix_mojibake(str(x)).strip()),
        }
    )
    raw["vbpl_tokens"] = raw["vbpl_extracted"].map(parse_vbpl_tokens)
    raw["vbpl_count"] = raw["vbpl_tokens"].map(len)

    missing_set = load_missing_doc_set(args.missing_docs)

    def has_missing(tokens: list[str]) -> bool:
        ids = {canonical_doc_id(t) for t in tokens if canonical_doc_id(t)}
        return bool(ids.intersection(missing_set))

    raw["has_missing_doc"] = raw["vbpl_tokens"].map(has_missing)
    raw["bucket"] = np.select(
        condlist=[
            raw["vbpl_count"] <= 0,
            raw["has_missing_doc"],
        ],
        choicelist=[
            "no_vbpl",
            "vbpl_missing_in_corpus",
        ],
        default="answerable",
    )

    if args.drop_duplicate_question:
        before = len(raw)
        raw = raw.drop_duplicates(subset=["question"], keep="first").copy()
        print(f"Deduplicated questions: {before} -> {len(raw)}")

    answerable = raw[raw["bucket"] == "answerable"].copy()
    if len(answerable) < args.evaluate_n + args.ce_val_n + 1:
        raise ValueError(
            f"Not enough answerable rows ({len(answerable)}) for evaluate_n={args.evaluate_n} and ce_val_n={args.ce_val_n}"
        )

    answerable["strat_key"] = build_strat_key(answerable)
    eval_df = sample_by_group(answerable, n=args.evaluate_n, seed=args.seed, group_col="strat_key").copy()
    eval_df["split"] = "evaluate"

    remain = raw[~raw["row_index"].isin(eval_df["row_index"])].copy()
    # CE validation is sampled from remaining rows, keeping bucket balance.
    remain["strat_key"] = remain["bucket"].astype(str)
    ce_val_df = sample_by_group(remain, n=args.ce_val_n, seed=args.seed + 1, group_col="strat_key").copy()
    ce_val_df["split"] = "val"

    train_df = remain[~remain["row_index"].isin(ce_val_df["row_index"])].copy()
    train_df["split"] = "train"

    # Keep only rows with non-empty question/answer for training.
    train_df = train_df[(train_df["question"] != "") & (train_df["ground_truth"] != "")]
    ce_val_df = ce_val_df[(ce_val_df["question"] != "") & (ce_val_df["ground_truth"] != "")]
    eval_df = eval_df[(eval_df["question"] != "") & (eval_df["ground_truth"] != "")]

    args.training_dir.mkdir(parents=True, exist_ok=True)
    args.evaluate_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = args.training_dir / "train_messages.jsonl"
    val_jsonl = args.training_dir / "val_messages.jsonl"
    eval_csv = args.evaluate_dir / "evaluate.csv"

    export_hf_jsonl(train_df, train_jsonl, DEFAULT_SYSTEM_PROMPT)
    export_hf_jsonl(ce_val_df, val_jsonl, DEFAULT_SYSTEM_PROMPT)
    export_evaluate_csv(eval_df, eval_csv)

    summary = {
        "sheet": sheet,
        "columns": {"question": q_col, "answer": a_col, "vbpl": vbpl_col},
        "counts": {
            "total_rows": int(len(raw)),
            "answerable_rows": int((raw["bucket"] == "answerable").sum()),
            "no_vbpl_rows": int((raw["bucket"] == "no_vbpl").sum()),
            "vbpl_missing_in_corpus_rows": int((raw["bucket"] == "vbpl_missing_in_corpus").sum()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(ce_val_df)),
            "evaluate_rows": int(len(eval_df)),
        },
        "paths": {
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "evaluate_csv": str(eval_csv),
        },
        "seed": args.seed,
        "evaluate_n": args.evaluate_n,
        "ce_val_n": args.ce_val_n,
        "drop_duplicate_question": args.drop_duplicate_question,
    }

    summary_path = args.training_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Split completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
