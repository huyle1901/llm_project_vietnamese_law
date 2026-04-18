# Scripts Layout

This directory is organized by function.

## `scripts/data`

Data preparation and corpus construction:

- `prepare_and_profile_dataset.py`
- `chunk_legal_corpus.py`
- `extract_doc_numbers_from_excel.py`
- `join_content_metadata.py`
- `filter_important_docs_from_hf.py`
- `download_hf_legal_dataset.py`

## `scripts/indexing`

Indexing and retrieval utilities:

- `faiss_ingest_langchain.py`
- `faiss_query_langchain.py`
- `kaggle_embed_jsonl_full.py`
- `opensearch_ingest_chunks.py`
- `test_opensearch_query.py`

## `scripts/training`

Training-data preparation:

- `split_res_for_lora.py`

## `scripts/publishing`

Dataset publication utilities:

- `upload_dataset_to_hf.py`

## Root-Level Runtime Scripts

Two core runtime scripts remain at the root because the notebooks import them directly:

- `rag_hybrid_opensearch_faiss_qwen.py`
- `evaluate_rag_random_subset.py`
