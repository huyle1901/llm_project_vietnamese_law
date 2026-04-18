# Vietnamese Legal RAG Project

This repository contains a Vietnamese legal question-answering project built around a retrieval-augmented generation (RAG) pipeline. The project covers:

- dataset preparation and filtering
- legal document chunking and metadata construction
- dense and sparse retrieval indexing
- RAG evaluation
- QLoRA training data preparation and notebooks

## Repository Structure

```text
.
|-- README.md                                  # project overview and repository guide
|-- .gitignore                                 # Git ignore rules for local-only and large files
|-- pyproject.toml                             # main project configuration and uv dependencies
|-- uv.lock                                    # locked package versions for reproducible installs
|-- uv.toml                                    # additional uv configuration
|-- requirements_hf.txt                        # dependency list for Hugging Face related workflows
|-- docker-compose.opensearch.yml              # local OpenSearch setup for sparse retrieval
|-- RES.xlsx                                   # source spreadsheet used to build train/val/eval splits
|-- important_docs.txt                         # list of selected important legal documents
|
|-- analysis/
|   |-- dataset_profile.json                   # dataset profiling summary
|   `-- res_xlsx_summary.json                  # summary statistics extracted from RES.xlsx
|
|-- data/
|   |-- datahuggingface/
|   |   |-- .gitkeep                           # keeps the directory in Git
|   |   |-- corpus_important_docs.jsonl        # filtered corpus of important legal documents
|   |   |-- content_important_docs.jsonl       # full content for the selected important documents
|   |   |-- metadata_important_docs.jsonl      # metadata for the selected important documents
|   |   |-- important_docs_from_excel.txt      # document identifiers extracted from the Excel file
|   |   |-- important_docs_missing.txt         # documents that were expected but not matched
|   |   `-- important_docs_summary.json        # summary of the filtering and matching process
|   |
|   |-- evaluate/
|   |   |-- evaluate.csv                       # fixed evaluation split used for RAG benchmarking
|   |   `-- evaluate_messages.jsonl            # evaluation data in chat/message format
|   |
|   |-- external/
|   |   `-- hf_vietnamese_legal_documents_metadata.parquet
|   |                                          # source metadata downloaded from Hugging Face
|   |
|   |-- faiss/
|   |   `-- law_chunks_e5_base/
|   |       `-- index.pkl                      # metadata mapping for the FAISS index used in the repo
|   |
|   `-- training_lora/
|       |-- train_messages.jsonl               # QLoRA training set
|       |-- val_messages.jsonl                 # QLoRA validation set
|       `-- split_summary.json                 # summary of train/val/eval split generation
|
|-- notebooks/
|   |-- rag_evaluate_base.ipynb                # notebook for evaluating the base RAG system
|   |-- rag_evaluate_qlora.ipynb               # notebook for evaluating RAG with a QLoRA adapter
|   |-- qlora-law.ipynb                        # notebook for QLoRA training
|   `-- kaggle_embed_jsonl_full.ipynb          # notebook for full-corpus embedding on Kaggle
|
|-- scripts/
|   |-- README.md                              # short notes for the scripts directory
|   |-- rag_hybrid_opensearch_faiss_qwen.py    # main RAG pipeline: retrieve, rerank, and generate
|   |-- evaluate_rag_random_subset.py          # evaluation and metric computation script
|   |
|   |-- data/
|   |   |-- prepare_and_profile_dataset.py     # dataset preparation and profiling
|   |   |-- download_hf_legal_dataset.py       # download legal data from Hugging Face
|   |   |-- extract_doc_numbers_from_excel.py  # extract document identifiers from RES.xlsx
|   |   |-- filter_important_docs_from_hf.py   # filter important documents from the larger corpus
|   |   |-- join_content_metadata.py           # merge document content with metadata
|   |   `-- chunk_legal_corpus.py              # split legal documents into retrieval chunks
|   |
|   |-- indexing/
|   |   |-- kaggle_embed_jsonl_full.py         # generate embeddings for the full corpus
|   |   |-- faiss_ingest_langchain.py          # ingest chunk data into FAISS
|   |   |-- faiss_query_langchain.py           # test/query the FAISS index
|   |   |-- opensearch_ingest_chunks.py        # ingest chunk data into OpenSearch
|   |   `-- test_opensearch_query.py           # test sparse retrieval queries
|   |
|   |-- training/
|   |   `-- split_res_for_lora.py              # build train/val/eval splits from RES.xlsx
|   |
|   `-- publishing/
|       `-- upload_dataset_to_hf.py            # upload prepared artifacts to Hugging Face
```

## Main Components

### 1. Data

- `RES.xlsx` is the original source file used to create training, validation, and evaluation splits.
- `data/datahuggingface/` stores the filtered legal corpus and its associated metadata.
- `data/evaluate/evaluate.csv` is the fixed benchmark split used to compare RAG runs.
- `data/training_lora/` contains chat-formatted data for QLoRA training and validation.

### 2. RAG Pipeline

- `scripts/rag_hybrid_opensearch_faiss_qwen.py` is the main runtime pipeline.
- The system combines dense retrieval, sparse retrieval, ranking fusion, and answer generation.
- `scripts/evaluate_rag_random_subset.py` is used to run evaluation and compute metrics.

### 3. Indexing

- `scripts/indexing/faiss_ingest_langchain.py` and `scripts/indexing/faiss_query_langchain.py` are used for FAISS-based retrieval.
- `scripts/indexing/opensearch_ingest_chunks.py` and `scripts/indexing/test_opensearch_query.py` are used for OpenSearch-based retrieval.
- `data/faiss/law_chunks_e5_base/index.pkl` stores the metadata mapping required by the FAISS workflow currently tracked in the repo.

### 4. QLoRA

- `notebooks/qlora-law.ipynb` is used to train the adapter.
- `data/training_lora/train_messages.jsonl` and `val_messages.jsonl` are the fine-tuning datasets.
- `notebooks/rag_evaluate_qlora.ipynb` is used to evaluate the retrieval pipeline with the trained adapter.

## Quick Start

### Install the environment

```bash
uv sync
```

If you also need the optional dependency groups for embedding, LLM workflows, or development:

```bash
uv sync --group embed --group llm --group dev
```

### Run the main notebooks

- `notebooks/rag_evaluate_base.ipynb`: evaluate the base RAG system
- `notebooks/rag_evaluate_qlora.ipynb`: evaluate RAG with QLoRA
- `notebooks/qlora-law.ipynb`: train the adapter

### Run the evaluation script

```bash
uv run python scripts/evaluate_rag_random_subset.py
```

## Notes

- This README documents only the files and folders currently included in the GitHub version of the repository.
- Large local artifacts and temporary files are intentionally excluded if they are not part of the tracked release.
