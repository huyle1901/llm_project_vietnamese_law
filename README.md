# Vietnamese Legal RAG Assignment

This repository contains a retrieval-augmented generation pipeline for Vietnamese legal question answering.

The project improves a simple assignment baseline with:

- structure-aware legal chunking
- metadata-rich retrieval units
- hybrid sparse + dense retrieval
- reciprocal rank fusion
- reranking
- document focus and context expansion
- grounded prompting
- an optional QLoRA adapter on top of `Qwen/Qwen2.5-3B-Instruct`

## Main Components

### Runtime scripts

- `scripts/rag_hybrid_opensearch_faiss_qwen.py`
  - main retrieval + generation pipeline
  - supports FAISS dense retrieval and sparse retrieval via local BM25 or OpenSearch
- `scripts/evaluate_rag_random_subset.py`
  - evaluation utilities and metric computation

### Data preparation scripts

- `scripts/data/prepare_and_profile_dataset.py`
- `scripts/data/chunk_legal_corpus.py`
- `scripts/data/extract_doc_numbers_from_excel.py`
- `scripts/data/join_content_metadata.py`
- `scripts/data/download_hf_legal_dataset.py`
- `scripts/data/filter_important_docs_from_hf.py`

### Indexing scripts

- `scripts/indexing/kaggle_embed_jsonl_full.py`
- `scripts/indexing/faiss_ingest_langchain.py`
- `scripts/indexing/faiss_query_langchain.py`
- `scripts/indexing/opensearch_ingest_chunks.py`
- `scripts/indexing/test_opensearch_query.py`

### Training and publishing

- `scripts/training/split_res_for_lora.py`
- `scripts/publishing/upload_dataset_to_hf.py`

## Main Notebooks

Only the notebooks needed for final reproduction are included in the GitHub release:

- `notebooks/rag_evaluate_base.ipynb`
  - evaluate the improved base RAG system on the fixed held-out split
- `notebooks/rag_evaluate_qlora.ipynb`
  - evaluate the improved RAG system with the QLoRA adapter on the same split
- `notebooks/qlora-law.ipynb`
  - train the QLoRA adapter
- `notebooks/kaggle_embed_jsonl_full.ipynb`
  - full-corpus embedding notebook for Kaggle

## Dataset Packaging

The full raw corpus is too large for a lightweight GitHub release, so this repository keeps the processed artifacts needed to reproduce retrieval and evaluation.

### Included in the repo

- `RES.xlsx`
- `data/datahuggingface/corpus_important_docs.jsonl`
- `data/datahuggingface/content_important_docs.jsonl`
- `data/datahuggingface/metadata_important_docs.jsonl`
- `data/datahuggingface/corpus_important_docs_chunks.jsonl`
- `data/datahuggingface/important_docs_from_excel.txt`
- `data/datahuggingface/important_docs_missing.txt`
- `data/datahuggingface/important_docs_summary.json`
- `data/faiss/law_chunks_e5_base/index.faiss`
- `data/faiss/law_chunks_e5_base/index.pkl`
- `data/evaluate/evaluate.csv`
- `data/evaluate/evaluate_messages.jsonl`
- `data/training_lora/train_messages.jsonl`
- `data/training_lora/val_messages.jsonl`
- `data/training_lora/split_summary.json`
- `data/external/hf_vietnamese_legal_documents_metadata.parquet`

### Excluded from the repo

- `data/raw/export_1/`
- `data/datahuggingface/content_full.jsonl`
- `data/datahuggingface/metadata_full.jsonl`
- `data/datahuggingface/hf_vohuutridung_raw/`
- temporary embedding shards and local cache files

## Evaluation Split

The final comparison uses the same fixed held-out file:

- `data/evaluate/evaluate.csv`

This split is derived from `RES.xlsx` with:

```bash
uv run python scripts/training/split_res_for_lora.py --evaluate-n 150 --ce-val-n 150
```

Meaning:

- `train_messages.jsonl`: QLoRA training data
- `val_messages.jsonl`: 150-row validation split
- `evaluate.csv`: fixed 150-row held-out RAG evaluation split

The current reported runs use the first 100 rows of that fixed split.

## Quick Start

### 1. Install dependencies

This project uses `uv`.

```bash
uv sync --group embed --group llm --group dev
```

If you only need the base project dependencies:

```bash
uv sync
```

### 2. Run the base pipeline notebook

Open:

- `notebooks/rag_evaluate_base.ipynb`

Required assets:

- `RES.xlsx`
- `data/datahuggingface/corpus_important_docs_chunks.jsonl`
- `data/faiss/law_chunks_e5_base/`
- `data/evaluate/evaluate.csv`

### 3. Run the QLoRA evaluation notebook

Open:

- `notebooks/rag_evaluate_qlora.ipynb`

Required assets:

- the same retrieval assets as the base notebook
- the published or local QLoRA adapter

### 4. Retrain the QLoRA adapter

Open:

- `notebooks/qlora-law.ipynb`

Required assets:

- `data/training_lora/train_messages.jsonl`
- `data/training_lora/val_messages.jsonl`

## Colab / Kaggle Notes

- The base and QLoRA evaluation notebooks are designed around a Google Drive project root such as:
  - `/content/drive/MyDrive/LLM_RAG_PROJECT/`
- `notebooks/kaggle_embed_jsonl_full.ipynb` is intended for Kaggle embedding runs
- local experiments can use OpenSearch BM25, but the reported Colab evaluation runs use local BM25 + FAISS

## Report

The submission report is stored in:

- `reports/RAG_Assignment_Report.tex`
- `reports/RAG_Assignment_Report.md`

Architecture figures used by the report are stored in:

- `reports/assets/rag_system_architecture.png`
- `reports/assets/rag_system_architecture_simple.png`

## Notes

- The original baseline result in the report is not a perfect same-split comparison because it was obtained on a different and smaller sample.
- The improved base system and the QLoRA system are compared on the same fixed held-out split.
- If you use recent `peft` versions in Colab and encounter adapter-loading issues, remove incompatible `torchao` builds from the runtime before loading the adapter.
