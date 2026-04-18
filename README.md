# Vietnamese Legal RAG Project

This repository contains a Vietnamese legal question-answering project built around a retrieval-augmented generation (RAG) pipeline. It includes:

- dataset preparation and filtering
- legal document chunking
- dense retrieval with FAISS
- sparse retrieval with OpenSearch
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

- `scripts/indexing/kaggle_embed_jsonl_full.py` and `notebooks/kaggle_embed_jsonl_full.ipynb` are used to generate embeddings on Kaggle.
- `scripts/indexing/faiss_ingest_langchain.py` and `scripts/indexing/faiss_query_langchain.py` are used for FAISS-based retrieval.
- `scripts/indexing/opensearch_ingest_chunks.py` and `scripts/indexing/test_opensearch_query.py` are used for OpenSearch-based retrieval.
- `data/faiss/law_chunks_e5_base/index.pkl` is tracked, but the full local FAISS index also requires `index.faiss`, which must be rebuilt locally.

### 4. QLoRA

- `notebooks/qlora-law.ipynb` is used to train the adapter.
- `data/training_lora/train_messages.jsonl` and `val_messages.jsonl` are the fine-tuning datasets.
- `notebooks/rag_evaluate_qlora.ipynb` is used to evaluate the retrieval pipeline with the trained adapter.

## Environment Setup

Install the base environment:

```bash
uv sync
```

Install optional groups if you need embedding, LLM workflows, or development tools:

```bash
uv sync --group embed --group llm --group dev
```

## End-to-End Setup for `evaluate_rag_random_subset.py`

The evaluation script does not run from the tracked files alone. You need to build the chunk data, create dense embeddings on Kaggle, rebuild the FAISS index locally, ingest the chunks into OpenSearch, and run an LLM endpoint.

### 1. Required inputs

The default evaluation flow expects these files or services:

- `RES.xlsx`
- `data/datahuggingface/important_docs_missing.txt`
- `data/datahuggingface/corpus_important_docs_chunks.jsonl`
- `data/faiss/law_chunks_e5_base/`
- OpenSearch at `http://localhost:9200`
- an LLM endpoint at `http://localhost:11434`

The repo already includes the source files needed to rebuild the missing local artifacts:

- `data/datahuggingface/content_important_docs.jsonl`
- `data/datahuggingface/metadata_important_docs.jsonl`
- `data/datahuggingface/corpus_important_docs.jsonl`

### Optional: rebuild the filtered corpus from upstream sources

If you want to regenerate the filtered corpus instead of using the tracked `*_important_docs*.jsonl` files, run the full preprocessing chain below.

Extract document numbers from `RES.xlsx`:

```bash
uv run python scripts/data/extract_doc_numbers_from_excel.py \
  --excel RES.xlsx \
  --out data/datahuggingface/important_docs_from_excel.txt
```

Download the full Hugging Face dataset export:

```bash
uv run python scripts/data/download_hf_legal_dataset.py \
  --dataset vohuutridung/vietnamese-legal-documents \
  --raw-dir data/datahuggingface/hf_legal_dataset_raw \
  --content-out data/datahuggingface/content_full.jsonl \
  --metadata-out data/datahuggingface/metadata_full.jsonl
```

Filter the full dataset down to the important documents:

```bash
uv run python scripts/data/filter_important_docs_from_hf.py \
  --important-docs data/datahuggingface/important_docs_from_excel.txt \
  --metadata data/datahuggingface/metadata_full.jsonl \
  --content data/datahuggingface/content_full.jsonl \
  --out-metadata data/datahuggingface/metadata_important_docs.jsonl \
  --out-content data/datahuggingface/content_important_docs.jsonl \
  --out-missing data/datahuggingface/important_docs_missing.txt \
  --out-summary data/datahuggingface/important_docs_summary.json
```

Join the filtered content and metadata into one corpus file:

```bash
uv run python scripts/data/join_content_metadata.py \
  --content data/datahuggingface/content_important_docs.jsonl \
  --metadata data/datahuggingface/metadata_important_docs.jsonl \
  --out data/datahuggingface/corpus_important_docs.jsonl
```

After that, continue with the chunking and indexing steps below.

### 2. Build the chunk JSONL used by retrieval

Create the chunk file expected by the indexing and evaluation scripts:

```bash
uv run python scripts/data/chunk_legal_corpus.py \
  --input data/datahuggingface/content_important_docs.jsonl \
  --output data/datahuggingface/corpus_important_docs_chunks.jsonl
```

This file is required by:

- `scripts/indexing/kaggle_embed_jsonl_full.py`
- `scripts/indexing/opensearch_ingest_chunks.py`
- `scripts/indexing/faiss_ingest_langchain.py`
- `scripts/evaluate_rag_random_subset.py`

### 3. Generate dense embeddings on Kaggle

Dense embeddings are expected to be generated on Kaggle with `multilingual-e5-base`.

You can use either:

- `notebooks/kaggle_embed_jsonl_full.ipynb`
- `scripts/indexing/kaggle_embed_jsonl_full.py`

Example Kaggle command:

```bash
python scripts/indexing/kaggle_embed_jsonl_full.py \
  --input-jsonl /kaggle/input/<your-dataset>/corpus_important_docs_chunks.jsonl \
  --output-dir /kaggle/working/emb_out \
  --model-name intfloat/multilingual-e5-base \
  --device cuda
```

The Kaggle output folder will contain files such as:

- `manifest.json`
- `embeddings_00000.npy`
- `metadata_00000.parquet`

Download the full output folder from Kaggle after embedding finishes.

### 4. Put the downloaded Kaggle export into a local folder

After downloading from Kaggle, place the exported embedding folder somewhere local in this repo, for example:

```text
data/faiss_exports/emb_out/
```

That folder should contain at least:

- `manifest.json`
- all `embeddings_*.npy` shards
- all matching `metadata_*` shard files

### 5. Build the local FAISS index from the downloaded embeddings

Use the downloaded Kaggle export to rebuild the FAISS index used by dense retrieval:

```bash
uv run python scripts/indexing/faiss_ingest_langchain.py \
  --manifest data/faiss_exports/emb_out/manifest.json \
  --emb-root data/faiss_exports/emb_out \
  --text-jsonl data/datahuggingface/corpus_important_docs_chunks.jsonl \
  --index-dir data/faiss/law_chunks_e5_base \
  --overwrite
```

After this step, `data/faiss/law_chunks_e5_base/` should contain the local FAISS artifacts, including:

- `index.faiss`
- `index.pkl`
- `index_info.json`

`index.faiss` is not stored in GitHub because it is too large, so rebuilding it locally is required.

### 6. Start OpenSearch

Start the local OpenSearch service:

```bash
docker compose -f docker-compose.opensearch.yml up -d
```

### 7. Ingest the chunks into OpenSearch

Ingest the chunk JSONL into OpenSearch for sparse retrieval:

```bash
uv run python scripts/indexing/opensearch_ingest_chunks.py \
  --input-jsonl data/datahuggingface/corpus_important_docs_chunks.jsonl \
  --opensearch-url http://localhost:9200 \
  --index-name law_chunks_bm25 \
  --recreate-index
```

This is the database ingestion step used by the BM25/OpenSearch side of the hybrid retriever.

Important distinction:

- Kaggle embeddings are used to rebuild the local FAISS index.
- OpenSearch ingestion uses the chunk JSONL, not the `.npy` embedding shards.

### 8. Run an LLM endpoint

By default, the evaluation script expects:

- base URL: `http://localhost:11434`
- model: `qwen2.5:3b-instruct`

If you use Ollama, a typical setup is:

```bash
ollama pull qwen2.5:3b-instruct
ollama serve
```

If your LLM endpoint is different, pass `--llm-base-url` and `--llm-model` explicitly when running evaluation.

### 9. Run the evaluation script

Once the chunk file, FAISS index, OpenSearch index, and LLM endpoint are ready, run:

```bash
uv run python scripts/evaluate_rag_random_subset.py \
  --res-xlsx RES.xlsx \
  --missing-docs data/datahuggingface/important_docs_missing.txt \
  --chunks-jsonl data/datahuggingface/corpus_important_docs_chunks.jsonl \
  --faiss-index-dir data/faiss/law_chunks_e5_base \
  --opensearch-url http://localhost:9200 \
  --opensearch-index law_chunks_bm25 \
  --llm-base-url http://localhost:11434 \
  --llm-model qwen2.5:3b-instruct
```

Outputs are written by default under:

```text
analysis/eval_runs/
```

Each run saves:

- `rag_eval_results.csv`
- `sampled_questions.csv`
- `summary.json`

## Optional Data Preparation Scripts

These scripts are useful if you want to rebuild earlier project artifacts, but they are not required for the minimal evaluation path above:

- `scripts/data/prepare_and_profile_dataset.py`
- `scripts/data/download_hf_legal_dataset.py`
- `scripts/data/extract_doc_numbers_from_excel.py`
- `scripts/data/filter_important_docs_from_hf.py`
- `scripts/data/join_content_metadata.py`
- `scripts/training/split_res_for_lora.py`

## Notebooks

- `notebooks/rag_evaluate_base.ipynb`: evaluate the base RAG system
- `notebooks/rag_evaluate_qlora.ipynb`: evaluate RAG with QLoRA
- `notebooks/qlora-law.ipynb`: train the adapter
- `notebooks/kaggle_embed_jsonl_full.ipynb`: run full-corpus embedding on Kaggle

## Notes

- This README documents both the tracked files and the local artifacts that must be rebuilt to run retrieval and evaluation end to end.
- Large artifacts such as `index.faiss` and chunk/embedding outputs are intentionally kept out of Git when they are too large or machine-specific.
