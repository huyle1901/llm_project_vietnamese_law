from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalize_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def is_doc_no_col(col: str) -> bool:
    c = normalize_col(col)
    return (
        ("số hiệu" in c and ("vbpl" in c or "trích xuất" in c or "van ban" in c))
        or ("so hieu" in c and ("vbpl" in c or "trich" in c or "van ban" in c))
        or c in {"document_number", "doc_no", "so_ky_hieu"}
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract document numbers from RES.xlsx")
    p.add_argument("--excel", type=Path, default=Path("RES.xlsx"))
    p.add_argument("--out", type=Path, default=Path("data/datahuggingface/important_docs_from_excel.txt"))
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    sheets = pd.read_excel(args.excel, sheet_name=None)

    collected: list[str] = []
    used_cols: list[str] = []

    for sheet_name, df in sheets.items():
        for col in df.columns:
            if not is_doc_no_col(col):
                continue
            used_cols.append(f"{sheet_name}:{col}")
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            vals = vals[~vals.str.lower().isin({"nan", "none", "null"})]
            collected.extend(vals.tolist())

    unique = []
    seen = set()
    for v in collected:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")

    print("excel:", args.excel)
    print("used_columns:", len(used_cols))
    for c in used_cols:
        print(" -", c)
    print("doc_numbers:", len(unique))
    print("out:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
