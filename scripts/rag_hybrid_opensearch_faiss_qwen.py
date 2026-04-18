#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from langchain_core.embeddings import Embeddings
except Exception:
    class Embeddings:  # type: ignore[no-redef]
        pass


q = "Kể từ 1/11/2025, có đúng là những giao dịch nộp/rút/chuyển tiền trong một ngày đạt từ 400 triệu đồng trở lên—kể cả khi người thực hiện không có tài khoản—vẫn phải được tổ chức thực hiện thủ tục nhận biết khách hàng không?"

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý Việt Nam. "
    "Chỉ được sử dụng thông tin trong CONTEXT. Không được dùng kiến thức bên ngoài. "
    "Bỏ qua mọi đoạn không liên quan trực tiếp đến câu hỏi. "
    "Nếu không đủ dữ liệu thì nói rõ: 'Không đủ dữ liệu trong CONTEXT'. "
    "Bắt buộc trả lời bằng TIẾNG VIỆT. Không dùng tiêu đề tiếng Anh. "
    "Bắt buộc dùng 4 mục đúng tên như sau: "
    "(1) Căn cứ pháp lý:, (2) Nội dung áp dụng:, (3) Ngoại lệ/không áp dụng:, (4) Tóm lại:. "
    "Trích dẫn theo định dạng [1], [2], ..."
)

ANSWER_REQUIREMENTS = (
    "Yêu cầu trả lời:\n"
    "- Trả lời bằng tiếng Việt có dấu, rõ ràng, đầy đủ thông tin, đúng trọng tâm câu hỏi.\n"
    "- Ưu tiên căn cứ pháp lý liên quan trực tiếp trong CONTEXT; không sử dụng kiến thức bên ngoài.\n"
    "- Nếu CONTEXT có đủ căn cứ, nên mở đầu bằng 'Căn cứ ...'.\n"
    "- Trình bày theo luồng: căn cứ chính -> nội dung áp dụng -> ngoại lệ (nếu có) -> kết luận.\n"
    "- Mỗi ý quan trọng cần có trích dẫn [1], [2], ...\n"
    "- Nếu CONTEXT không đủ để kết luận chắc chắn, nói rõ 'Không đủ dữ liệu trong CONTEXT'.\n"
)


STOPWORDS = {
    "la", "va", "voi", "cho", "cua", "theo", "khi", "neu", "thi", "duoc", "khong", "co", "cac",
    "mot", "nhung", "trong", "tu", "den", "tai", "nay", "do", "bao", "gom", "phan", "dieu",
    "khoan", "diem", "nguoi", "to", "chuc", "khach", "hang", "giao", "dich", "nhan", "biet",
    "quy", "dinh", "luat", "thong", "tu", "nghi", "dinh", "quyet", "dinh", "hoac", "which",
    "what", "how", "where", "when", "does", "must", "from", "with", "that", "this",
}


_LOCAL_TEXT_GENERATOR_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Any]] = {}


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


@dataclass
class RetrievedItem:
    chunk_id: str
    payload: dict[str, Any]
    source: str
    rank: int
    raw_score: float | None = None


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", fold_text(text))


def normalize_doc_number(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("_", "-")
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s).upper()
    s = re.sub(r"\s+", "", s)
    return s


def extract_target_doc_numbers(query: str) -> set[str]:
    # Accept common legal doc ids like:
    # 46/2021/NĐ-CP, 50/2024/TT-NHNN, 07/VBHN-VPQH, 01/NHNN-CK
    token_re = re.compile(r"[0-9A-Za-zÀ-ỹĐđ][0-9A-Za-zÀ-ỹĐđ/\-]*")
    out: set[str] = set()
    for m in token_re.finditer(query):
        tok = m.group(0).strip().strip(".,;:()[]{}\"'")
        if "/" not in tok:
            continue
        parts = tok.split("/")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            if re.search(r"[A-Za-zÀ-ỹĐđ]", parts[2]):
                out.add(normalize_doc_number(tok))
        elif len(parts) == 2 and parts[0].isdigit():
            if re.search(r"[A-Za-zÀ-ỹĐđ]", parts[1]):
                out.add(normalize_doc_number(tok))
    return {x for x in out if x}


def extract_target_article_numbers(query: str) -> set[int]:
    folded = fold_text(query)
    out: set[int] = set()
    for m in re.finditer(r"\b(dieu)\s+(\d+)\b", folded):
        try:
            out.add(int(m.group(2)))
        except ValueError:
            pass
    return out


def extract_query_terms(query: str, min_len: int = 3, max_terms: int = 40) -> set[str]:
    terms = [t for t in tokenize(query) if len(t) >= min_len and t not in STOPWORDS]
    return set(terms[:max_terms])


def text_overlap_score(query_terms: set[str], text: str) -> tuple[int, float]:
    if not query_terms:
        return 0, 0.0
    tokens = set(tokenize(text))
    overlap = len(query_terms.intersection(tokens))
    ratio = overlap / max(1, len(query_terms))
    return overlap, ratio


def get_opensearch_client(opensearch_url: str):
    try:
        from opensearchpy import OpenSearch
    except Exception as e:
        raise RuntimeError("Missing dependency: opensearchpy") from e

    return OpenSearch(
        hosts=[opensearch_url],
        use_ssl=opensearch_url.startswith("https"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def search_opensearch(
    client: Any,
    index_name: str,
    query: str,
    top_k: int,
    opensearch_operator: str,
    opensearch_minimum_should_match: str,
) -> list[RetrievedItem]:
    multi_match: dict[str, Any] = {
        "query": query,
        "fields": ["chunk_text^3", "title^2", "document_number^6", "legal_type", "legal_sectors"],
        "type": "best_fields",
        "operator": opensearch_operator,
    }
    if opensearch_minimum_should_match:
        multi_match["minimum_should_match"] = opensearch_minimum_should_match

    body = {
        "size": top_k,
        "query": {"multi_match": multi_match},
        "_source": [
            "id",
            "chunk_id",
            "chunk_text",
            "section_type",
            "article_no",
            "clause_no",
            "document_number",
            "title",
            "url",
            "legal_type",
            "legal_sectors",
            "issuing_authority",
            "issuance_date",
            "signers",
        ],
    }
    res = client.search(index=index_name, body=body)

    out: list[RetrievedItem] = []
    for rank, h in enumerate(res.get("hits", {}).get("hits", []), start=1):
        src = h.get("_source", {}) or {}
        cid = str(src.get("chunk_id") or "")
        if not cid:
            continue
        out.append(
            RetrievedItem(
                chunk_id=cid,
                payload=src,
                source="opensearch",
                rank=rank,
                raw_score=h.get("_score"),
            )
        )
    return out


def search_faiss(
    index_dir: Path,
    query: str,
    top_k: int,
    model_name: str,
    device: str,
    normalize: bool,
    prefix_mode: str,
) -> list[RetrievedItem]:
    try:
        import torch
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        raise RuntimeError("FAISS query requires torch, sentence-transformers, and langchain-community") from e

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    embeddings = E5Embeddings(
        model_name=model_name,
        device=device,
        normalize=normalize,
        prefix_mode=prefix_mode,
    )

    vs = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    hits = vs.similarity_search_with_score(query, k=top_k)

    out: list[RetrievedItem] = []
    for rank, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata or {}
        payload = dict(meta)
        payload.setdefault("chunk_text", doc.page_content)
        cid = str(payload.get("chunk_id") or "")
        if not cid:
            continue
        out.append(
            RetrievedItem(
                chunk_id=cid,
                payload=payload,
                source="faiss",
                rank=rank,
                raw_score=float(score),
            )
        )
    return out


def rrf_fuse(lists: list[list[RetrievedItem]], k: int = 60) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for lst in lists:
        for item in lst:
            rrf = 1.0 / (k + item.rank)
            cid = item.chunk_id
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "payload": item.payload,
                    "rrf_score": 0.0,
                    "sources": set(),
                }
            merged[cid]["rrf_score"] += rrf
            merged[cid]["sources"].add(item.source)

    fused = list(merged.values())
    for x in fused:
        x["sources"] = sorted(x["sources"])
    return fused


def enrich_and_filter_by_relevance(
    items: list[dict[str, Any]],
    query_terms: set[str],
    min_overlap: int,
    keep_at_least: int,
    required_doc_numbers: set[str] | None = None,
    required_article_numbers: set[int] | None = None,
) -> list[dict[str, Any]]:
    required_doc_numbers = {normalize_doc_number(x) for x in (required_doc_numbers or set()) if str(x).strip()}
    required_article_numbers = set(required_article_numbers or set())

    enriched: list[dict[str, Any]] = []

    for it in items:
        p = it.get("payload", {}) or {}
        text_for_match = " ".join(
            [
                str(p.get("document_number") or ""),
                str(p.get("title") or ""),
                str(p.get("legal_type") or ""),
                str(p.get("legal_sectors") or ""),
                str(p.get("chunk_text") or ""),
            ]
        )
        overlap_count, overlap_ratio = text_overlap_score(query_terms, text_for_match)

        doc_no = normalize_doc_number(str(p.get("document_number") or ""))
        doc_match = bool(required_doc_numbers) and doc_no in required_doc_numbers

        article_match = False
        if required_article_numbers:
            raw_article = p.get("article_no")
            try:
                if raw_article is not None and str(raw_article).strip() != "":
                    article_no = int(float(raw_article))
                    article_match = article_no in required_article_numbers
            except Exception:
                article_match = False

        constraint_bonus = 0.0
        if doc_match:
            constraint_bonus += 5.0
        if article_match:
            constraint_bonus += 2.0
        if doc_match and article_match:
            constraint_bonus += 2.0

        it = dict(it)
        it["doc_match"] = doc_match
        it["article_match"] = article_match
        it["overlap_count"] = overlap_count
        it["overlap_ratio"] = overlap_ratio
        it["hybrid_score"] = it.get("rrf_score", 0.0) + (0.01 * overlap_count) + overlap_ratio + constraint_bonus
        enriched.append(it)

    enriched.sort(
        key=lambda x: (
            1 if x.get("doc_match") else 0,
            1 if x.get("article_match") else 0,
            x.get("hybrid_score", 0.0),
            x.get("overlap_count", 0),
            x.get("rrf_score", 0.0),
        ),
        reverse=True,
    )

    # If query explicitly names a legal document, force priority to that document.
    candidate = enriched
    if required_doc_numbers:
        matched_docs = [x for x in enriched if x.get("doc_match")]
        if matched_docs:
            # If enough chunks from requested legal doc(s), keep focus on them.
            if len(matched_docs) >= keep_at_least:
                candidate = matched_docs
            else:
                candidate = matched_docs + [x for x in enriched if not x.get("doc_match")]

    filtered = [x for x in candidate if x.get("overlap_count", 0) >= min_overlap]
    if len(filtered) < keep_at_least:
        filtered = candidate[:keep_at_least]
    return filtered



def limit_chunks_per_doc(items: list[dict[str, Any]], max_chunks_per_doc: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    counter: dict[str, int] = {}
    for it in items:
        p = it.get("payload", {}) or {}
        doc_key = str(p.get("document_number") or p.get("id") or "")
        if not doc_key:
            out.append(it)
            continue
        seen = counter.get(doc_key, 0)
        if seen >= max_chunks_per_doc:
            continue
        counter[doc_key] = seen + 1
        out.append(it)
    return out


def _doc_norm_from_item(item: dict[str, Any]) -> str:
    p = item.get("payload", {}) or {}
    return normalize_doc_number(str(p.get("document_number") or ""))


def rank_documents_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_map: dict[str, dict[str, Any]] = {}
    for ch in chunks:
        p = ch.get("payload", {}) or {}
        doc_norm = normalize_doc_number(str(p.get("document_number") or ""))
        if not doc_norm:
            continue

        score = float(
            ch.get("rerank_score")
            if ch.get("rerank_score") is not None
            else ch.get("hybrid_score")
            if ch.get("hybrid_score") is not None
            else ch.get("rrf_score", 0.0)
        )
        row = doc_map.setdefault(
            doc_norm,
            {
                "doc_norm": doc_norm,
                "doc_display": str(p.get("document_number") or ""),
                "scores": [],
                "doc_match_count": 0,
                "article_match_count": 0,
            },
        )
        row["scores"].append(score)
        if ch.get("doc_match"):
            row["doc_match_count"] += 1
        if ch.get("article_match"):
            row["article_match_count"] += 1

    ranked: list[dict[str, Any]] = []
    for _, row in doc_map.items():
        scores = sorted(row["scores"], reverse=True)
        max_s = scores[0]
        mean_top2 = sum(scores[:2]) / min(2, len(scores))
        doc_score = (
            max_s
            + (0.35 * mean_top2)
            + (0.5 * row["doc_match_count"])
            + (0.2 * row["article_match_count"])
            + (0.01 * len(scores))
        )
        ranked.append(
            {
                "doc_norm": row["doc_norm"],
                "doc_display": row["doc_display"],
                "doc_score": float(doc_score),
                "chunk_count": len(scores),
                "doc_match_count": row["doc_match_count"],
                "article_match_count": row["article_match_count"],
            }
        )

    ranked.sort(
        key=lambda x: (
            x.get("doc_score", 0.0),
            x.get("doc_match_count", 0),
            x.get("article_match_count", 0),
            x.get("chunk_count", 0),
        ),
        reverse=True,
    )
    return ranked


def select_primary_documents(
    ranked_docs: list[dict[str, Any]],
    required_doc_numbers: set[str] | None,
    max_docs: int,
    score_margin: float,
) -> list[str]:
    max_docs = max(1, int(max_docs))
    required = {normalize_doc_number(x) for x in (required_doc_numbers or set()) if str(x).strip()}

    if required:
        req_ranked = [x["doc_norm"] for x in ranked_docs if x.get("doc_norm") in required]
        if req_ranked:
            return req_ranked[:max_docs]

    if not ranked_docs:
        return []

    top_score = float(ranked_docs[0].get("doc_score", 0.0))
    selected = [ranked_docs[0]["doc_norm"]]
    for row in ranked_docs[1:]:
        if len(selected) >= max_docs:
            break
        s = float(row.get("doc_score", 0.0))
        if top_score <= 0:
            break
        rel_gap = (top_score - s) / max(1e-9, abs(top_score))
        if rel_gap <= score_margin:
            selected.append(row["doc_norm"])
    return selected


def apply_document_focus(
    chunks: list[dict[str, Any]],
    mode: str,
    required_doc_numbers: set[str] | None,
    max_primary_docs: int,
    primary_score_margin: float,
    primary_doc_quota: int,
    other_doc_quota: int,
) -> list[dict[str, Any]]:
    if not chunks or mode == "off":
        return chunks

    required = {normalize_doc_number(x) for x in (required_doc_numbers or set()) if str(x).strip()}
    if mode == "strict" and required:
        strict_only = [ch for ch in chunks if _doc_norm_from_item(ch) in required]
        if strict_only:
            return strict_only

    ranked_docs = rank_documents_from_chunks(chunks)
    primary_docs = select_primary_documents(
        ranked_docs=ranked_docs,
        required_doc_numbers=required if mode in ("auto", "strict") else set(),
        max_docs=max_primary_docs,
        score_margin=primary_score_margin,
    )
    if not primary_docs:
        return chunks

    primary_set = set(primary_docs)
    out: list[dict[str, Any]] = []
    primary_count: dict[str, int] = {}
    other_count: dict[str, int] = {}

    for ch in chunks:
        doc_norm = _doc_norm_from_item(ch)
        if not doc_norm:
            continue
        if doc_norm in primary_set:
            seen = primary_count.get(doc_norm, 0)
            if primary_doc_quota > 0 and seen >= primary_doc_quota:
                continue
            primary_count[doc_norm] = seen + 1
            out.append(ch)
        else:
            if other_doc_quota <= 0:
                continue
            seen = other_count.get(doc_norm, 0)
            if seen >= other_doc_quota:
                continue
            other_count[doc_norm] = seen + 1
            out.append(ch)

    return out if out else chunks


def approx_tokens(text: str) -> int:
    words = text.split()
    return int(len(words) * 1.33) if words else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    words = text.split()
    if not words:
        return text
    max_words = max(1, int(max_tokens / 1.33))
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


@dataclass
class ChunkStore:
    by_chunk_id: dict[str, dict[str, Any]]
    by_doc: dict[str, list[dict[str, Any]]]


def _chunk_seq(chunk_id: str) -> int | None:
    m = re.search(r"_(\d+)$", str(chunk_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


@lru_cache(maxsize=2)
def load_chunk_store(path_str: str) -> ChunkStore:
    path = Path(path_str)
    by_chunk_id: dict[str, dict[str, Any]] = {}
    by_doc: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return ChunkStore(by_chunk_id=by_chunk_id, by_doc=by_doc)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("chunk_id") or "")
            if not cid:
                continue
            by_chunk_id[cid] = row
            doc_key = normalize_doc_number(str(row.get("document_number") or ""))
            if doc_key:
                by_doc.setdefault(doc_key, []).append(row)

    for doc_key, rows in by_doc.items():
        rows.sort(key=lambda r: (_chunk_seq(str(r.get("chunk_id") or "")) or -1))
        by_doc[doc_key] = rows

    return ChunkStore(by_chunk_id=by_chunk_id, by_doc=by_doc)


def expand_context_chunks(
    base_chunks: list[dict[str, Any]],
    store: ChunkStore | None,
    neighbor_window: int = 1,
    same_article_limit: int = 2,
    max_additional: int = 4,
) -> list[dict[str, Any]]:
    if not store or not store.by_chunk_id or max_additional <= 0:
        return base_chunks

    out = list(base_chunks)
    seen = {str(x.get("chunk_id") or "") for x in out}
    added = 0

    for base in base_chunks:
        if added >= max_additional:
            break
        p = base.get("payload", {}) or {}
        doc_key = normalize_doc_number(str(p.get("document_number") or ""))
        if not doc_key:
            continue
        rows = store.by_doc.get(doc_key) or []
        if not rows:
            continue

        current_cid = str(base.get("chunk_id") or "")
        current_seq = _chunk_seq(current_cid)
        raw_article = p.get("article_no")
        article_no: int | None = None
        try:
            if raw_article is not None and str(raw_article).strip() != "":
                article_no = int(float(raw_article))
        except Exception:
            article_no = None

        if article_no is not None and same_article_limit > 0:
            same_article_rows = []
            for r in rows:
                ra = r.get("article_no")
                if ra is None or str(ra).strip() == "":
                    continue
                try:
                    if int(float(ra)) == article_no:
                        same_article_rows.append(r)
                except Exception:
                    continue
            kept = 0
            for r in same_article_rows:
                cid = str(r.get("chunk_id") or "")
                if not cid or cid in seen:
                    continue
                out.append(
                    {
                        "chunk_id": cid,
                        "payload": r,
                        "rrf_score": 0.0,
                        "hybrid_score": 0.0,
                        "overlap_count": 0,
                        "sources": ["expand"],
                    }
                )
                seen.add(cid)
                added += 1
                kept += 1
                if kept >= same_article_limit or added >= max_additional:
                    break

        if current_seq is None or added >= max_additional:
            continue

        for r in rows:
            cid = str(r.get("chunk_id") or "")
            if not cid or cid in seen:
                continue
            seq = _chunk_seq(cid)
            if seq is None or abs(seq - current_seq) > neighbor_window:
                continue
            out.append(
                {
                    "chunk_id": cid,
                    "payload": r,
                    "rrf_score": 0.0,
                    "hybrid_score": 0.0,
                    "overlap_count": 0,
                    "sources": ["expand"],
                }
            )
            seen.add(cid)
            added += 1
            if added >= max_additional:
                break

    return out


def build_context(chunks: list[dict[str, Any]], max_context_tokens: int = 2200) -> str:
    lines: list[str] = []
    used_tokens = 0
    idx = 1
    for ch in chunks:
        p = ch.get("payload", {}) or {}
        chunk_text = str(p.get("chunk_text") or "")
        block = (
            f"[{idx}] title: {p.get('title')}\n"
            f"document_number: {p.get('document_number')} | article: {p.get('article_no')} | clause: {p.get('clause_no')}\n"
            f"url: {p.get('url')}\n"
            f"content: {chunk_text}\n"
        )
        block_tokens = approx_tokens(block)

        if max_context_tokens > 0 and used_tokens + block_tokens > max_context_tokens:
            remaining = max_context_tokens - used_tokens
            if remaining <= 120:
                break
            trimmed_text = truncate_to_tokens(chunk_text, max_tokens=max(80, remaining - 80))
            block = (
                f"[{idx}] title: {p.get('title')}\n"
                f"document_number: {p.get('document_number')} | article: {p.get('article_no')} | clause: {p.get('clause_no')}\n"
                f"url: {p.get('url')}\n"
                f"content: {trimmed_text}\n"
            )
            block_tokens = approx_tokens(block)
            if max_context_tokens > 0 and used_tokens + block_tokens > max_context_tokens:
                break

        lines.append(block)
        used_tokens += block_tokens
        idx += 1
    return "\n".join(lines)


def build_answer_prompt(question: str, context: str) -> str:
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n{ANSWER_REQUIREMENTS}"


_RERANKER_CACHE: dict[tuple[str, str], Any] = {}


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    use_reranker: bool,
    reranker_model: str,
    reranker_device: str,
    reranker_top_n: int,
) -> list[dict[str, Any]]:
    if not use_reranker or len(chunks) <= 1:
        return chunks

    cache_key = (reranker_model, reranker_device)
    reranker = _RERANKER_CACHE.get(cache_key)
    if reranker is None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except Exception as e:
            print(f"Warning: reranker unavailable, fallback to no-rerank. detail={e}")
            return chunks

        device = reranker_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        try:
            reranker = CrossEncoder(reranker_model, device=device)
            _RERANKER_CACHE[cache_key] = reranker
        except Exception as e:
            print(f"Warning: cannot load reranker model, fallback to no-rerank. detail={e}")
            return chunks

    pairs = []
    for ch in chunks:
        p = ch.get("payload", {}) or {}
        passage = f"{p.get('title')}\n{p.get('chunk_text')}"
        pairs.append((query, passage))

    try:
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception as e:
        print(f"Warning: reranker inference failed, fallback to no-rerank. detail={e}")
        return chunks

    reranked = []
    for ch, s in zip(chunks, scores):
        x = dict(ch)
        x["rerank_score"] = float(s)
        reranked.append(x)

    reranked.sort(
        key=lambda x: (
            1 if x.get("doc_match") else 0,
            1 if x.get("article_match") else 0,
            x.get("rerank_score", 0.0),
            x.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )

    if reranker_top_n > 0:
        return reranked[:reranker_top_n]
    return reranked


def _get_local_text_generator(
    model_name: str,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    key = (model_name, device, dtype, trust_remote_code)
    cached = _LOCAL_TEXT_GENERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as e:
        raise RuntimeError("Local LLM backend requires transformers + torch installed") from e

    run_device = device
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available for local LLM. Falling back to CPU.")
        run_device = "cpu"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype in dtype_map:
        model_kwargs["torch_dtype"] = dtype_map[dtype]
    elif dtype == "auto" and run_device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_index = 0 if run_device.startswith("cuda") else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
    )

    _LOCAL_TEXT_GENERATOR_CACHE[key] = (generator, tokenizer)
    return generator, tokenizer


def call_local_chat_completion(
    model_name: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> str:
    generator, tokenizer = _get_local_text_generator(
        model_name=model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{user_prompt}\n\n"
            "Tra loi:"
        )

    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-5)

    out = generator(prompt_text, **gen_kwargs)
    text = out[0]["generated_text"]
    if text.startswith(prompt_text):
        text = text[len(prompt_text) :]
    return text.strip()


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_qwen_chat_completion(
    base_url: str,
    model_name: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    base = base_url.rstrip("/")
    openai_url = f"{base}/v1/chat/completions"
    openai_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        data = _post_json(openai_url, openai_payload, timeout=timeout)
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # Ollama may not expose OpenAI-compatible endpoint depending on version/config.
        if e.code != 404:
            raise RuntimeError(f"LLM HTTPError {e.code}: {body}") from e
    except TimeoutError as e:
        # Keep fallback path for Ollama native endpoint.
        if "11434" not in base:
            raise RuntimeError(
                f"LLM request timed out after {timeout}s at {openai_url}. "
                "Increase --llm-timeout or reduce --max-tokens."
            ) from e
    except urllib.error.URLError as e:
        if "11434" not in base:
            raise RuntimeError(
                f"Cannot connect to LLM endpoint: {openai_url}. "
                "Please start your Qwen/OpenAI-compatible server or pass --llm-base-url."
            ) from e

    # Fallback for Ollama native API.
    ollama_url = f"{base}/api/chat"
    ollama_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        data = _post_json(ollama_url, ollama_payload, timeout=timeout)
    except TimeoutError as e:
        raise RuntimeError(
            f"LLM request timed out after {timeout}s at {ollama_url}. "
            "Increase --llm-timeout or reduce --max-tokens / --final-k."
        ) from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTPError {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to LLM endpoint: {ollama_url}. "
            "Please start your LLM server or pass --llm-base-url."
        ) from e

    if isinstance(data.get("message"), dict):
        return str(data["message"].get("content", "")).strip()
    if "response" in data:
        return str(data["response"]).strip()
    raise RuntimeError(f"Unexpected LLM response format: {data}")


def generate_llm_answer(
    llm_backend: str,
    llm_model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    llm_timeout: int,
    llm_base_url: str,
    llm_local_device: str,
    llm_local_dtype: str,
    llm_trust_remote_code: bool,
) -> str:
    if llm_backend == "local":
        return call_local_chat_completion(
            model_name=llm_model,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            device=llm_local_device,
            dtype=llm_local_dtype,
            trust_remote_code=llm_trust_remote_code,
        )

    return call_qwen_chat_completion(
        base_url=llm_base_url,
        model_name=llm_model,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid RAG: OpenSearch + FAISS -> Qwen answer")
    parser.add_argument("--query", type=str, default=q)

    parser.add_argument("--opensearch-url", type=str, default="http://localhost:9200")
    parser.add_argument("--opensearch-index", type=str, default="law_chunks_bm25")
    parser.add_argument("--opensearch-operator", choices=["and", "or"], default="and")
    parser.add_argument("--opensearch-minimum-should-match", type=str, default="")

    parser.add_argument("--faiss-index-dir", type=Path, default=Path("data/faiss/law_chunks_e5_base"))
    parser.add_argument("--e5-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prefix-mode", choices=["query_passage", "none"], default="query_passage")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument("--bm25-k", type=int, default=30)
    parser.add_argument("--faiss-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2)
    parser.add_argument("--min-query-term-overlap", type=int, default=2)
    parser.add_argument("--keep-at-least", type=int, default=5)
    parser.add_argument("--chunks-jsonl", type=Path, default=Path("data/datahuggingface/corpus_important_docs_chunks.jsonl"))
    parser.add_argument("--context-neighbor-window", type=int, default=1)
    parser.add_argument("--context-same-article-limit", type=int, default=2)
    parser.add_argument("--context-max-additional", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=2200)
    parser.add_argument("--use-reranker", action="store_true", default=True)
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-device", type=str, default="cuda")
    parser.add_argument("--reranker-top-n", type=int, default=20)
    parser.add_argument("--doc-focus-mode", choices=["off", "auto", "strict"], default="auto")
    parser.add_argument("--doc-focus-max-primary-docs", type=int, default=1)
    parser.add_argument("--doc-focus-primary-margin", type=float, default=0.12)
    parser.add_argument("--doc-focus-primary-doc-quota", type=int, default=4)
    parser.add_argument("--doc-focus-other-doc-quota", type=int, default=1)

    parser.add_argument("--llm-backend", choices=["api", "local"], default="api")
    parser.add_argument("--llm-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--llm-local-device", type=str, default="cuda")
    parser.add_argument("--llm-local-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--llm-trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--llm-timeout", type=int, default=600)

    parser.add_argument("--show-context", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = parse_args()
    try:
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        raise RuntimeError("Missing dependency: langchain-community") from e

    client = get_opensearch_client(args.opensearch_url)
    bm25_hits = search_opensearch(
        client=client,
        index_name=args.opensearch_index,
        query=args.query,
        top_k=args.bm25_k,
        opensearch_operator=args.opensearch_operator,
        opensearch_minimum_should_match=args.opensearch_minimum_should_match,
    )

    faiss_hits = search_faiss(
        index_dir=args.faiss_index_dir,
        query=args.query,
        top_k=args.faiss_k,
        model_name=args.e5_model,
        device=args.device,
        normalize=args.normalize,
        prefix_mode=args.prefix_mode,
    )

    target_docs = extract_target_doc_numbers(args.query)
    target_articles = extract_target_article_numbers(args.query)

    fused = rrf_fuse([bm25_hits, faiss_hits], k=args.rrf_k)
    query_terms = extract_query_terms(args.query)
    fused = enrich_and_filter_by_relevance(
        fused,
        query_terms=query_terms,
        min_overlap=args.min_query_term_overlap,
        keep_at_least=args.keep_at_least,
        required_doc_numbers=target_docs,
        required_article_numbers=target_articles,
    )
    fused = rerank_chunks(
        query=args.query,
        chunks=fused,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_top_n=args.reranker_top_n,
    )
    fused = apply_document_focus(
        chunks=fused,
        mode=args.doc_focus_mode,
        required_doc_numbers=target_docs,
        max_primary_docs=args.doc_focus_max_primary_docs,
        primary_score_margin=args.doc_focus_primary_margin,
        primary_doc_quota=args.doc_focus_primary_doc_quota,
        other_doc_quota=args.doc_focus_other_doc_quota,
    )
    fused = limit_chunks_per_doc(fused, max_chunks_per_doc=args.max_chunks_per_doc)
    base_chunks = fused[: args.final_k]

    chunk_store: ChunkStore | None = None
    if args.chunks_jsonl.exists():
        chunk_store = load_chunk_store(str(args.chunks_jsonl.resolve()))
    final_chunks = expand_context_chunks(
        base_chunks=base_chunks,
        store=chunk_store,
        neighbor_window=args.context_neighbor_window,
        same_article_limit=args.context_same_article_limit,
        max_additional=args.context_max_additional,
    )

    if not final_chunks:
        raise RuntimeError("No relevant chunks retrieved after filtering. Try lowering --min-query-term-overlap.")

    context = build_context(final_chunks, max_context_tokens=args.max_context_tokens)
    prompt = build_answer_prompt(args.query, context)

    answer = generate_llm_answer(
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        user_prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        llm_timeout=args.llm_timeout,
        llm_base_url=args.llm_base_url,
        llm_local_device=args.llm_local_device,
        llm_local_dtype=args.llm_local_dtype,
        llm_trust_remote_code=args.llm_trust_remote_code,
    )

    print(f"Query: {args.query}")
    print(
        "Retrieved: "
        f"bm25={len(bm25_hits)} | faiss={len(faiss_hits)} | base={len(base_chunks)} | final={len(final_chunks)}"
    )

    if args.show_context:
        print("\n" + "-" * 100)
        print("CONTEXT")
        print("-" * 100)
        print(context)

    print("\n" + "=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(answer)

    print("\n" + "=" * 100)
    print("SOURCES")
    print("=" * 100)
    for i, ch in enumerate(final_chunks, 1):
        p = ch.get("payload", {}) or {}
        print(
            f"[{i}] chunk_id={ch.get('chunk_id')} rrf={ch.get('rrf_score'):.6f} hybrid={ch.get('hybrid_score'):.6f} "
            f"overlap={ch.get('overlap_count')} src={ch.get('sources')} | "
            f"doc={p.get('document_number')} | article={p.get('article_no')} | clause={p.get('clause_no')}"
        )
        print(f"    title={p.get('title')}")
        print(f"    url={p.get('url')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
