# Vietnamese Legal RAG Project

Repository nay chua pipeline RAG tieng Viet cho bai toan hoi dap van ban phap luat, gom cac buoc:

- chuan bi va loc du lieu van ban
- chunking va tao metadata
- indexing cho dense retrieval va sparse retrieval
- danh gia he thong RAG
- tao du lieu huan luyen va notebook QLoRA

## Cau truc thu muc

```text
.
|-- README.md                                  # mo ta tong quan du an va cau truc repo
|-- .gitignore                                 # quy tac bo qua file/thu muc khong dua len Git
|-- pyproject.toml                             # cau hinh project va dependency chinh cho uv
|-- uv.lock                                    # lock file de cai dung dung version package
|-- uv.toml                                    # cau hinh uv bo sung cho moi truong local
|-- requirements_hf.txt                        # danh sach package phuc vu cac buoc Hugging Face
|-- docker-compose.opensearch.yml              # dung OpenSearch local cho sparse retrieval
|-- RES.xlsx                                   # file du lieu goc dung de tao tap train/val/evaluate
|-- important_docs.txt                         # danh sach van ban quan trong duoc trich rieng
|
|-- analysis/
|   |-- dataset_profile.json                   # thong ke va profiling cua bo du lieu da xu ly
|   `-- res_xlsx_summary.json                  # tom tat noi dung va phan bo cua RES.xlsx
|
|-- data/
|   |-- datahuggingface/
|   |   |-- .gitkeep                           # giu thu muc trong Git
|   |   |-- corpus_important_docs.jsonl        # tap hop van ban quan trong da loc
|   |   |-- content_important_docs.jsonl       # noi dung chi tiet cua van ban quan trong
|   |   |-- metadata_important_docs.jsonl      # metadata di kem cho bo van ban quan trong
|   |   |-- important_docs_from_excel.txt      # danh sach van ban lay tu file Excel
|   |   |-- important_docs_missing.txt         # cac van ban khong tim thay trong qua trinh doi chieu
|   |   `-- important_docs_summary.json        # tong hop ket qua loc va doi chieu van ban
|   |
|   |-- evaluate/
|   |   |-- evaluate.csv                       # tap mau co dinh dung de benchmark RAG
|   |   `-- evaluate_messages.jsonl            # du lieu evaluate o dang message/chat
|   |
|   |-- external/
|   |   `-- hf_vietnamese_legal_documents_metadata.parquet
|   |                                          # metadata goc tai ve tu nguon Hugging Face
|   |
|   |-- faiss/
|   |   `-- law_chunks_e5_base/
|   |       `-- index.pkl                      # metadata/map cua chi muc FAISS dang dung
|   |
|   `-- training_lora/
|       |-- train_messages.jsonl               # du lieu train cho QLoRA
|       |-- val_messages.jsonl                 # du lieu validation cho QLoRA
|       `-- split_summary.json                 # thong tin cach chia train/val/evaluate
|
|-- notebooks/
|   |-- rag_evaluate_base.ipynb                # notebook danh gia base RAG
|   |-- rag_evaluate_qlora.ipynb               # notebook danh gia RAG co adapter QLoRA
|   |-- qlora-law.ipynb                        # notebook huan luyen adapter QLoRA
|   `-- kaggle_embed_jsonl_full.ipynb          # notebook tao embedding toan bo tren Kaggle
|
|-- scripts/
|   |-- README.md                              # ghi chu ngan cho nhom script
|   |-- rag_hybrid_opensearch_faiss_qwen.py    # pipeline RAG chinh: retrieve, rerank, generate
|   |-- evaluate_rag_random_subset.py          # script tinh metric va danh gia he thong
|   |
|   |-- data/
|   |   |-- prepare_and_profile_dataset.py     # chuan bi du lieu va sinh thong ke tong quan
|   |   |-- download_hf_legal_dataset.py       # tai bo du lieu phap luat tu Hugging Face
|   |   |-- extract_doc_numbers_from_excel.py  # trich so ky hieu van ban tu RES.xlsx
|   |   |-- filter_important_docs_from_hf.py   # loc nhom van ban quan trong tu nguon lon
|   |   |-- join_content_metadata.py           # ghep noi dung van ban voi metadata
|   |   `-- chunk_legal_corpus.py              # chia van ban thanh cac chunk phuc vu retrieval
|   |
|   |-- indexing/
|   |   |-- kaggle_embed_jsonl_full.py         # tao embedding cho corpus lon
|   |   |-- faiss_ingest_langchain.py          # nap du lieu vao chi muc FAISS
|   |   |-- faiss_query_langchain.py           # thu truy van FAISS
|   |   |-- opensearch_ingest_chunks.py        # nap chunk vao OpenSearch
|   |   `-- test_opensearch_query.py           # test truy van sparse retrieval
|   |
|   |-- training/
|   |   `-- split_res_for_lora.py              # chia du lieu RES.xlsx cho train/val/evaluate
|   |
|   `-- publishing/
|       `-- upload_dataset_to_hf.py            # day artifact/du lieu len Hugging Face
```

## Thanh phan chinh

### 1. Du lieu

- `RES.xlsx` la nguon dau vao goc de tao bo train/validation/evaluation.
- `data/datahuggingface/` chua bo van ban phap luat da loc va metadata di kem.
- `data/evaluate/evaluate.csv` la tap danh gia co dinh de so sanh cac phien ban RAG.
- `data/training_lora/` chua du lieu huan luyen theo dinh dang chat cho adapter.

### 2. Pipeline RAG

- `scripts/rag_hybrid_opensearch_faiss_qwen.py` la entry point chinh cho he thong.
- He thong ket hop dense retrieval, sparse retrieval, hop nhat ket qua va tao cau tra loi.
- `scripts/evaluate_rag_random_subset.py` dung de chay benchmark va tinh metric.

### 3. Indexing

- `scripts/indexing/faiss_ingest_langchain.py` va `scripts/indexing/faiss_query_langchain.py` phuc vu FAISS.
- `scripts/indexing/opensearch_ingest_chunks.py` va `scripts/indexing/test_opensearch_query.py` phuc vu OpenSearch.
- `data/faiss/law_chunks_e5_base/index.pkl` giu metadata can thiet cua chi muc dang su dung trong repo.

### 4. QLoRA

- `notebooks/qlora-law.ipynb` dung de huan luyen adapter.
- `data/training_lora/train_messages.jsonl` va `val_messages.jsonl` la du lieu cho qua trinh fine-tuning.
- `notebooks/rag_evaluate_qlora.ipynb` dung de danh gia he thong sau khi nap adapter.

## Cach dung nhanh

### Cai moi truong

```bash
uv sync
```

Neu can them nhom dependency phuc vu embed, LLM hoac phat trien:

```bash
uv sync --group embed --group llm --group dev
```

### Chay cac notebook chinh

- `notebooks/rag_evaluate_base.ipynb`: danh gia base RAG
- `notebooks/rag_evaluate_qlora.ipynb`: danh gia RAG + QLoRA
- `notebooks/qlora-law.ipynb`: huan luyen adapter

### Chay script danh gia

```bash
uv run python scripts/evaluate_rag_random_subset.py
```

## Ghi chu

- README nay chi mo ta nhung file va thu muc dang co mat trong ban cap nhat tren GitHub.
- Cac artifact lon hoac file tam phuc vu local run khong duoc liet ke o day neu khong nam trong repo hien tai.
