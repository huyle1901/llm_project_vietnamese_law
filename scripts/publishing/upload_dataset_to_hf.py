#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception as e:
        raise RuntimeError(
            f"Found {dotenv_path} but python-dotenv is not installed. "
            "Install it with: uv pip install python-dotenv"
        ) from e
    load_dotenv(dotenv_path=dotenv_path, override=False)


def resolve_token(cli_token: str | None) -> str:
    token = (cli_token or "").strip()
    if token:
        return token
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()

    raise RuntimeError(
        "Missing Hugging Face token. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) "
        "or pass --token."
    )


def write_readme(
    readme_path: Path,
    repo_id: str,
    train_count: int,
    val_count: int,
) -> None:
    total = train_count + val_count
    if total < 1000:
        size_category = "n<1K"
    elif total < 10000:
        size_category = "1K<n<10K"
    elif total < 100000:
        size_category = "10K<n<100K"
    else:
        size_category = "100K<n<1M"

    content = f"""---
language:
- vi
pretty_name: legal-rag-splits
license: other
task_categories:
- question-answering
- text-generation
size_categories:
- {size_category}
configs:
- config_name: sft_jsonl
  data_files:
  - split: train
    path: training_lora/train_messages.jsonl
  - split: validation
    path: training_lora/val_messages.jsonl
---

# {repo_id}

This dataset was prepared for legal RAG fine-tuning/evaluation.

## Files
- `training_lora/train_messages.jsonl`: SFT/LoRA train split
- `training_lora/val_messages.jsonl`: SFT/LoRA validation split
- `training_lora/split_summary.json`: split metadata

## Split sizes
- train: {train_count}
- val: {val_count}
"""
    readme_path.write_text(content, encoding="utf-8")


def count_jsonl_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        return sum(1 for _ in reader)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload local dataset splits to Hugging Face Hub (dataset repo).")
    p.add_argument("--repo-id", required=True, help="Example: your_username/legal-rag-splits")
    p.add_argument("--token", default="", help="HF write token. Prefer env var HF_TOKEN.")
    p.add_argument("--private", action="store_true", help="Create private dataset repo.")
    p.add_argument("--branch", default="main")
    p.add_argument("--commit-message", default="Upload dataset splits for LoRA/SFT and evaluation")
    p.add_argument("--training-dir", type=Path, default=Path("data/training_lora"))
    p.add_argument("--evaluate-dir", type=Path, default=Path("data/evaluate"))
    p.add_argument("--dotenv-path", type=Path, default=Path(".env"), help="Path to .env file for HF_TOKEN.")
    p.add_argument("--skip-readme", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate and print plan only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_file(args.dotenv_path)

    train_jsonl = args.training_dir / "train_messages.jsonl"
    val_jsonl = args.training_dir / "val_messages.jsonl"
    split_summary = args.training_dir / "split_summary.json"

    required = [train_jsonl, val_jsonl]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    train_count = count_jsonl_lines(train_jsonl)
    val_count = count_jsonl_lines(val_jsonl)

    print("Upload plan:")
    print(f"- training_dir: {args.training_dir.resolve()}")
    print(f"- train_jsonl: {train_jsonl.resolve()}")
    print(f"- val_jsonl: {val_jsonl.resolve()}")
    print(f"- repo_id: {args.repo_id}")
    print(f"- branch: {args.branch}")
    print(f"- training_lora/train_messages.jsonl ({train_count} rows)")
    print(f"- training_lora/val_messages.jsonl ({val_count} rows)")
    if split_summary.exists():
        print("- training_lora/split_summary.json")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    token = resolve_token(args.token)

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise RuntimeError("Please install huggingface_hub first: uv pip install huggingface_hub") from e

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "training_lora").mkdir(parents=True, exist_ok=True)
        (root / "evaluate").mkdir(parents=True, exist_ok=True)

        shutil.copy2(train_jsonl, root / "training_lora" / train_jsonl.name)
        shutil.copy2(val_jsonl, root / "training_lora" / val_jsonl.name)
        if split_summary.exists():
            shutil.copy2(split_summary, root / "training_lora" / split_summary.name)

        if not args.skip_readme:
            write_readme(
                readme_path=root / "README.md",
                repo_id=args.repo_id,
                train_count=train_count,
                val_count=val_count,
            )

        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo="",
            revision=args.branch,
            commit_message=args.commit_message,
            token=token,
        )

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
