#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make sibling script imports work in both direct-run and module-run modes.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from rag_hybrid_opensearch_faiss_qwen import (
    apply_document_focus,
    E5Embeddings,
    RetrievedItem,
    build_answer_prompt,
    build_context,
    call_qwen_chat_completion,
    expand_context_chunks,
    enrich_and_filter_by_relevance,
    extract_target_article_numbers,
    extract_target_doc_numbers,
    extract_query_terms,
    get_opensearch_client,
    load_chunk_store,
    limit_chunks_per_doc,
    rerank_chunks,
    rrf_fuse,
    search_opensearch,
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


def normalize_text(text: str) -> str:
    text = fold_text(str(text)).replace("\n", " ").strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


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


def load_res_sheet(path: Path, sheet_name: str | None) -> tuple[pd.DataFrame, tuple[str, str, str], str]:
    xls = pd.ExcelFile(path)
    chosen = sheet_name if sheet_name else xls.sheet_names[0]
    df = xls.parse(chosen)
    cols = resolve_columns(df)
    return df, cols, chosen


def parse_vbpl_tokens(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = fix_mojibake(str(value)).strip()
    if not s:
        return []
    parts = re.split(r"[\n,;|]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            if fold_text(p) == "khong trich xuat duoc":
                continue
            out.append(p)
    return out


def load_missing_doc_set(path: Path) -> set[str]:
    out: set[str] = set()
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


def select_random_subset(
    df: pd.DataFrame,
    vbpl_col: str,
    missing_doc_ids: set[str],
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    keep_mask = []
    stats = {"total": len(df), "excluded_missing": 0, "excluded_empty_vbpl": 0}

    for _, row in df.iterrows():
        tokens = parse_vbpl_tokens(row.get(vbpl_col))
        if not tokens:
            keep_mask.append(False)
            stats["excluded_empty_vbpl"] += 1
            continue
        cands = {canonical_doc_id(t) for t in tokens if canonical_doc_id(t)}
        if cands.intersection(missing_doc_ids):
            keep_mask.append(False)
            stats["excluded_missing"] += 1
            continue
        keep_mask.append(True)

    eligible = df.loc[keep_mask].copy()
    stats["eligible"] = len(eligible)

    if len(eligible) < n:
        raise ValueError(f"Eligible rows ({len(eligible)}) < n ({n}).")

    sampled = eligible.sample(n=n, random_state=seed).copy()
    sampled = sampled.sort_index().reset_index().rename(columns={"index": "row_index"})
    return sampled, stats


def jaccard_similarity(pred: str, true: str) -> float:
    pred_words = set(normalize_text(pred).split())
    true_words = set(normalize_text(true).split())
    if not pred_words or not true_words:
        return 0.0
    return len(pred_words & true_words) / len(pred_words | true_words)


def token_overlap_score(pred: str, true: str) -> float:
    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not true_words:
        return 0.0
    from collections import Counter

    return sum((Counter(pred_words) & Counter(true_words)).values()) / len(true_words)


def bleu_score_simple(pred: str, true: str) -> float:
    from collections import Counter

    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not pred_words or not true_words:
        return 0.0
    overlap = sum((Counter(pred_words) & Counter(true_words)).values())
    precision = overlap / len(pred_words)
    bp = 1.0 if len(pred_words) >= len(true_words) else np.exp(1 - len(true_words) / len(pred_words))
    return bp * precision


def rouge_l_score(pred: str, true: str) -> float:
    pred_words = normalize_text(pred).split()
    true_words = normalize_text(true).split()
    if not pred_words or not true_words:
        return 0.0
    m, n = len(pred_words), len(true_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == true_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def cosine_similarity_score(pred: str, true: str, emb_model: E5Embeddings) -> float:
    if not pred or not true:
        return 0.0
    pv = emb_model.embed_query(pred)
    tv = emb_model.embed_query(true)
    return float(cosine_similarity([pv], [tv])[0][0])


@dataclass
class RagConfig:
    opensearch_url: str
    opensearch_index: str
    opensearch_operator: str
    opensearch_minimum_should_match: str
    bm25_k: int
    faiss_k: int
    rrf_k: int
    final_k: int
    max_chunks_per_doc: int
    min_query_term_overlap: int
    keep_at_least: int
    llm_base_url: str
    llm_model: str
    temperature: float
    max_tokens: int
    llm_timeout: int
    use_reranker: bool
    reranker_model: str
    reranker_device: str
    reranker_top_n: int
    doc_focus_mode: str
    doc_focus_max_primary_docs: int
    doc_focus_primary_margin: float
    doc_focus_primary_doc_quota: int
    doc_focus_other_doc_quota: int
    context_neighbor_window: int
    context_same_article_limit: int
    context_max_additional: int
    max_context_tokens: int


def search_faiss_loaded(vs: FAISS, query: str, top_k: int) -> list[RetrievedItem]:
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


def rag_answer_for_question(
    question: str,
    cfg: RagConfig,
    client: Any,
    faiss_vs: FAISS,
    chunk_store: Any,
) -> tuple[str, list[dict[str, Any]]]:
    bm25_hits = search_opensearch(
        client=client,
        index_name=cfg.opensearch_index,
        query=question,
        top_k=cfg.bm25_k,
        opensearch_operator=cfg.opensearch_operator,
        opensearch_minimum_should_match=cfg.opensearch_minimum_should_match,
    )
    faiss_hits = search_faiss_loaded(faiss_vs, question, cfg.faiss_k)

    target_docs = extract_target_doc_numbers(question)
    target_articles = extract_target_article_numbers(question)

    fused = rrf_fuse([bm25_hits, faiss_hits], k=cfg.rrf_k)
    query_terms = extract_query_terms(question)
    fused = enrich_and_filter_by_relevance(
        fused,
        query_terms=query_terms,
        min_overlap=cfg.min_query_term_overlap,
        keep_at_least=cfg.keep_at_least,
        required_doc_numbers=target_docs,
        required_article_numbers=target_articles,
    )
    fused = rerank_chunks(
        query=question,
        chunks=fused,
        use_reranker=cfg.use_reranker,
        reranker_model=cfg.reranker_model,
        reranker_device=cfg.reranker_device,
        reranker_top_n=cfg.reranker_top_n,
    )
    fused = apply_document_focus(
        chunks=fused,
        mode=cfg.doc_focus_mode,
        required_doc_numbers=target_docs,
        max_primary_docs=cfg.doc_focus_max_primary_docs,
        primary_score_margin=cfg.doc_focus_primary_margin,
        primary_doc_quota=cfg.doc_focus_primary_doc_quota,
        other_doc_quota=cfg.doc_focus_other_doc_quota,
    )
    fused = limit_chunks_per_doc(fused, max_chunks_per_doc=cfg.max_chunks_per_doc)
    base_chunks = fused[: cfg.final_k]
    if not base_chunks:
        return "", []

    final_chunks = expand_context_chunks(
        base_chunks=base_chunks,
        store=chunk_store,
        neighbor_window=cfg.context_neighbor_window,
        same_article_limit=cfg.context_same_article_limit,
        max_additional=cfg.context_max_additional,
    )
    context = build_context(final_chunks, max_context_tokens=cfg.max_context_tokens)
    prompt = build_answer_prompt(question, context)
    answer = call_qwen_chat_completion(
        base_url=cfg.llm_base_url,
        model_name=cfg.llm_model,
        user_prompt=prompt,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.llm_timeout,
    )
    return answer, final_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG on random subset from RES.xlsx, excluding docs in important_docs_missing.txt"
    )
    parser.add_argument("--res-xlsx", type=Path, default=Path("RES.xlsx"))
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--missing-docs", type=Path, default=Path("data/datahuggingface/important_docs_missing.txt"))
    parser.add_argument("--n", type=int, choices=[1,3, 5, 10, 20, 30], default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--faiss-index-dir", type=Path, default=Path("data/faiss/law_chunks_e5_base"))
    parser.add_argument("--e5-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--metric-embedding-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prefix-mode", choices=["query_passage", "none"], default="query_passage")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")

    parser.add_argument("--opensearch-url", type=str, default="http://localhost:9200")
    parser.add_argument("--opensearch-index", type=str, default="law_chunks_bm25")
    parser.add_argument("--opensearch-operator", choices=["and", "or"], default="and")
    parser.add_argument("--opensearch-minimum-should-match", type=str, default="")
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

    parser.add_argument("--llm-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:3b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--llm-timeout", type=int, default=900)

    parser.add_argument("--output-dir", type=Path, default=Path("analysis/eval_runs"))
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = parse_args()

    try:
        from langchain_community.vectorstores import FAISS
    except Exception as e:
        raise RuntimeError("Missing dependency: langchain-community") from e
    if not args.res_xlsx.exists():
        raise FileNotFoundError(f"RES.xlsx not found: {args.res_xlsx}")
    if not args.missing_docs.exists():
        raise FileNotFoundError(f"Missing-doc list not found: {args.missing_docs}")

    missing_set = load_missing_doc_set(args.missing_docs)
    df, (q_col, a_col, vbpl_col), sheet = load_res_sheet(args.res_xlsx, args.sheet)
    sampled_df, sample_stats = select_random_subset(df, vbpl_col, missing_set, args.n, args.seed)

    emb = E5Embeddings(
        model_name=args.e5_model,
        device=args.device,
        normalize=args.normalize,
        prefix_mode=args.prefix_mode,
    )
    metric_emb = E5Embeddings(
        model_name=args.metric_embedding_model,
        device=args.device,
        normalize=True,
        prefix_mode=args.prefix_mode,
    )
    faiss_vs = FAISS.load_local(str(args.faiss_index_dir), emb, allow_dangerous_deserialization=True)
    os_client = get_opensearch_client(args.opensearch_url)
    chunk_store = load_chunk_store(str(args.chunks_jsonl.resolve())) if args.chunks_jsonl.exists() else None

    cfg = RagConfig(
        opensearch_url=args.opensearch_url,
        opensearch_index=args.opensearch_index,
        opensearch_operator=args.opensearch_operator,
        opensearch_minimum_should_match=args.opensearch_minimum_should_match,
        bm25_k=args.bm25_k,
        faiss_k=args.faiss_k,
        rrf_k=args.rrf_k,
        final_k=args.final_k,
        max_chunks_per_doc=args.max_chunks_per_doc,
        min_query_term_overlap=args.min_query_term_overlap,
        keep_at_least=args.keep_at_least,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        llm_timeout=args.llm_timeout,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_top_n=args.reranker_top_n,
        doc_focus_mode=args.doc_focus_mode,
        doc_focus_max_primary_docs=args.doc_focus_max_primary_docs,
        doc_focus_primary_margin=args.doc_focus_primary_margin,
        doc_focus_primary_doc_quota=args.doc_focus_primary_doc_quota,
        doc_focus_other_doc_quota=args.doc_focus_other_doc_quota,
        context_neighbor_window=args.context_neighbor_window,
        context_same_article_limit=args.context_same_article_limit,
        context_max_additional=args.context_max_additional,
        max_context_tokens=args.max_context_tokens,
    )

    results: list[dict[str, Any]] = []
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Evaluating"):
        question = fix_mojibake(str(row[q_col])).strip()
        ground_truth = fix_mojibake(str(row[a_col])).strip()
        vbpl = fix_mojibake(str(row[vbpl_col])).strip()

        try:
            pred, final_chunks = rag_answer_for_question(question, cfg, os_client, faiss_vs, chunk_store)
        except Exception as e:
            pred = ""
            final_chunks = []
            err = str(e)
        else:
            err = ""

        metrics = {
            "Cosine_Similarity": cosine_similarity_score(pred, ground_truth, metric_emb),
            "Jaccard_Similarity": jaccard_similarity(pred, ground_truth),
            "Token_Overlap": token_overlap_score(pred, ground_truth),
            "BLEU_Score": bleu_score_simple(pred, ground_truth),
            "ROUGE_L": rouge_l_score(pred, ground_truth),
        }

        results.append(
            {
                "row_index": int(row["row_index"]),
                "question": question,
                "ground_truth": ground_truth,
                "vbpl_extracted": vbpl,
                "prediction": pred,
                "retrieved_docs": "|".join(
                    str((ch.get("payload", {}) or {}).get("document_number") or "") for ch in final_chunks
                ),
                "error": err,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    mean_metrics = {
        k: float(results_df[k].mean())
        for k in ["Cosine_Similarity", "Jaccard_Similarity", "Token_Overlap", "BLEU_Score", "ROUGE_L"]
    }

    run_name = args.run_name.strip() or f"n{args.n}_seed{args.seed}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "rag_eval_results.csv"
    sampled_csv = out_dir / "sampled_questions.csv"
    summary_json = out_dir / "summary.json"

    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    sampled_df.to_csv(sampled_csv, index=False, encoding="utf-8-sig")

    summary = {
        "sheet": sheet,
        "columns": {"question": q_col, "answer": a_col, "vbpl": vbpl_col},
        "sample_stats": sample_stats,
        "n": args.n,
        "seed": args.seed,
        "mean_metrics": mean_metrics,
        "config": {
            "opensearch_url": args.opensearch_url,
            "opensearch_index": args.opensearch_index,
            "faiss_index_dir": str(args.faiss_index_dir),
            "e5_model": args.e5_model,
            "metric_embedding_model": args.metric_embedding_model,
            "llm_base_url": args.llm_base_url,
            "llm_model": args.llm_model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "llm_timeout": args.llm_timeout,
            "chunks_jsonl": str(args.chunks_jsonl),
            "max_chunks_per_doc": args.max_chunks_per_doc,
            "max_context_tokens": args.max_context_tokens,
            "use_reranker": args.use_reranker,
            "reranker_model": args.reranker_model,
            "reranker_device": args.reranker_device,
            "reranker_top_n": args.reranker_top_n,
            "doc_focus_mode": args.doc_focus_mode,
            "doc_focus_max_primary_docs": args.doc_focus_max_primary_docs,
            "doc_focus_primary_margin": args.doc_focus_primary_margin,
            "doc_focus_primary_doc_quota": args.doc_focus_primary_doc_quota,
            "doc_focus_other_doc_quota": args.doc_focus_other_doc_quota,
        },
        "outputs": {
            "results_csv": str(results_csv),
            "sampled_csv": str(sampled_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== RANDOM EVAL SUMMARY ===")
    print(f"sheet={sheet} | n={args.n} | seed={args.seed}")
    print(
        "eligible="
        f"{sample_stats.get('eligible', 0)} / total={sample_stats.get('total', 0)} "
        f"(excluded_missing={sample_stats.get('excluded_missing', 0)}, "
        f"excluded_empty_vbpl={sample_stats.get('excluded_empty_vbpl', 0)})"
    )
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"saved: {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
