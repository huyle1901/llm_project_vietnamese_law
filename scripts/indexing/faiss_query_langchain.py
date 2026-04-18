#!/usr/bin/env python
from __future__ import annotations

import argparse

try:
    from langchain_core.embeddings import Embeddings
except Exception:
    class Embeddings:  # type: ignore[no-redef]
        pass

q = "Kể từ 1/11/2025, có đúng là những giao dịch nộp/rút/chuyển tiền trong một ngày đạt từ 400 triệu đồng trở lên—kể cả khi người thực hiện không có tài khoản—vẫn phải được tổ chức thực hiện thủ tục nhận biết khách hàng không?"



class E5Embeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        device: str,
        normalize: bool = True,
        prefix_mode: str = "query_passage",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.prefix_mode = prefix_mode
        self.model = SentenceTransformer(model_name, device=device, model_kwargs={"use_safetensors": True})

    def _to_passage(self, text: str) -> str:
        if self.prefix_mode == "query_passage" and not text.startswith("passage:"):
            return f"passage: {text}"
        return text

    def _to_query(self, text: str) -> str:
        if self.prefix_mode == "query_passage" and not text.startswith("query:"):
            return f"query: {text}"
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [self._to_passage(t) for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        q = self._to_query(text)
        vec = self.model.encode([q], normalize_embeddings=self.normalize, show_progress_bar=False)
        return vec[0].tolist()


def run_query(args: argparse.Namespace) -> int:
    try:
        import torch
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        raise RuntimeError("Query mode requires torch, sentence-transformers, and langchain-community.") from e

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    embeddings = E5Embeddings(
        model_name=args.model_name,
        device=device,
        normalize=args.normalize,
        prefix_mode=args.prefix_mode,
    )

    vs = FAISS.load_local(
        str(args.index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    hits = vs.similarity_search_with_score(args.query, k=args.top_k)

    print(f"Index dir: {args.index_dir}")
    print(f"Model: {args.model_name} | device={device}")
    print(f"Query: {args.query}")
    print(f"Returned: {len(hits)}")

    for i, (doc, score) in enumerate(hits, 1):
        meta = doc.metadata or {}
        text = " ".join((doc.page_content or "").split())
        if len(text) > args.snippet_chars:
            text = text[: args.snippet_chars] + "..."

        print("\n" + "=" * 90)
        print(f"#{i} | score={score}")
        print(
            f"chunk_id={meta.get('chunk_id')} | doc_no={meta.get('document_number')} "
            f"| section={meta.get('section_type')} | article={meta.get('article_no')} | clause={meta.get('clause_no')}"
        )
        print(f"title={meta.get('title')}")
        print(f"snippet={text}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query LangChain FAISS index with E5 embeddings")
    parser.add_argument("--index-dir", type=str, default="data/faiss/law_chunks_e5_base")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prefix-mode", choices=["query_passage", "none"], default="query_passage")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--query", type=str, default=q)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_query(args)


if __name__ == "__main__":
    raise SystemExit(main())
