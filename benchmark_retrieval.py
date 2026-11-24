import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from src.config import BASE_DIR, TOP_K_RESULTS
from src.rag_chain import (
    expand_to_full_pages,
    get_bm25_index,
    get_reranker,
    hybrid_retrieve,
    rerank_documents,
    tokenize,
)
from src.vector_store import get_vector_store

DEFAULT_DATASET_PATH = BASE_DIR / "eval_dataset.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "retrieval_benchmark_results.csv"


def load_eval_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load ground-truth benchmark queries from JSON."""
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_retrieval(docs: List[Any], expected_file: str, required_facts: List[str]) -> Dict[str, Any]:
    """Compute Hit@K, Reciprocal Rank (MRR), Target-Fact Recall, and context size."""
    files = [d.metadata.get("file_name", "") for d in docs]
    hit = 1.0 if expected_file in files else 0.0

    rr = 0.0
    for idx, fname in enumerate(files, start=1):
        if fname == expected_file:
            rr = 1.0 / idx
            break

    combined_text = "\n".join(d.page_content for d in docs).lower()
    matched = [fact for fact in required_facts if fact.lower() in combined_text]
    fact_recall = len(matched) / len(required_facts) if required_facts else 1.0

    return {
        "hit": hit,
        "rr": rr,
        "fact_recall": fact_recall,
        "chars": len(combined_text),
        "top1_file": files[0] if files else "None",
        "top1_page": docs[0].metadata.get("page_number", "?") if docs else "?",
    }


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    runs: int = 3,
) -> None:
    """Run an apples-to-apples retrieval ablation across dense baselines and the v2 pipeline."""
    queries = load_eval_dataset(dataset_path)
    vs = get_vector_store()

    # Warm up cached BM25 index & Cross-Encoder and time uncached rebuild vs warm lookup
    t_cold = time.perf_counter()
    _, corpus = get_bm25_index("All")
    bm25_cold_s = time.perf_counter() - t_cold
    _ = get_reranker()

    t_rebuild = time.perf_counter()
    _ = BM25Okapi([tokenize(doc.page_content) for doc in corpus])
    bm25_rebuild_ms = (time.perf_counter() - t_rebuild) * 1000.0

    t_warm = time.perf_counter()
    for _ in range(1000):
        get_bm25_index("All")
    bm25_warm_us = ((time.perf_counter() - t_warm) / 1000.0) * 1e6

    print(f"Loaded {len(queries)} ground-truth queries across {len(corpus)} chunks.")
    print(f"BM25 Uncached Rebuild: {bm25_rebuild_ms:.1f} ms | Cached Warm Lookup: {bm25_warm_us:.2f} us (cold load: {bm25_cold_s:.2f}s)")

    configs = [
        ("Dense Baseline (k=5, raw chunks)", "dense", 5),
        ("Hybrid RRF + Reranker + Full-Page (k=5)", "hybrid_v2", 5),
        ("Dense Baseline (k=10, raw chunks)", "dense", TOP_K_RESULTS),
        ("Hybrid RRF + Reranker (k=10, chunks only)", "hybrid_chunks", TOP_K_RESULTS),
        ("Hybrid RRF + Reranker + Full-Page (k=10)", "hybrid_v2", TOP_K_RESULTS),
    ]

    csv_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for label, mode, k in configs:
        run_latencies: List[float] = []
        per_query_metrics: Dict[str, Dict[str, Any]] = {}

        for run_idx in range(1, runs + 1):
            for item in queries:
                q_start = time.perf_counter()
                if mode == "dense":
                    docs = vs.similarity_search(item["user_input"], k=k)
                elif mode == "hybrid_chunks":
                    cand = hybrid_retrieve(vs, item["user_input"], k=k, category="All")
                    docs = rerank_documents(item["user_input"], cand, top_k=k)
                else:
                    cand = hybrid_retrieve(vs, item["user_input"], k=k, category="All")
                    reranked = rerank_documents(item["user_input"], cand, top_k=k)
                    docs = expand_to_full_pages(vs, reranked)
                elapsed = time.perf_counter() - q_start
                run_latencies.append(elapsed)

                res = score_retrieval(docs, item["expected_file"], item["required_facts"])
                if run_idx == 1:
                    per_query_metrics[item["id"]] = res

                csv_rows.append({
                    "config": label,
                    "mode": mode,
                    "k": k,
                    "run": run_idx,
                    "query_id": item["id"],
                    "category": item["category"],
                    "query": item["user_input"],
                    "expected_file": item["expected_file"],
                    "hit_at_k": res["hit"],
                    "reciprocal_rank": round(res["rr"], 4),
                    "fact_recall": round(res["fact_recall"], 4),
                    "context_chars": res["chars"],
                    "latency_s": round(elapsed, 4),
                    "top1_file": res["top1_file"],
                    "top1_page": res["top1_page"],
                })

        n_q = len(per_query_metrics)
        hit_rate = sum(m["hit"] for m in per_query_metrics.values()) / n_q * 100.0
        mrr = sum(m["rr"] for m in per_query_metrics.values()) / n_q
        fact_recall = sum(m["fact_recall"] for m in per_query_metrics.values()) / n_q * 100.0
        avg_chars = sum(m["chars"] for m in per_query_metrics.values()) / n_q
        mean_lat = statistics.mean(run_latencies)
        std_lat = statistics.stdev(run_latencies) if len(run_latencies) > 1 else 0.0

        summaries.append({
            "label": label,
            "k": k,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "fact_recall": fact_recall,
            "avg_chars": avg_chars,
            "mean_lat": mean_lat,
            "std_lat": std_lat,
        })

    # Save reproducible per-run CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n================ APPLES-TO-APPLES RETRIEVAL ABLATION ({len(queries)} Queries x {runs} Runs) ================")
    for s in summaries:
        print(
            f"{s['label']:46s} | Hit@{s['k']:<2d}: {s['hit_rate']:5.1f}% | "
            f"MRR: {s['mrr']:.3f} | FactRecall: {s['fact_recall']:5.1f}% | "
            f"AvgChars: {s['avg_chars']:7.0f} | Latency: {s['mean_lat']:.3f}s +/- {s['std_lat']:.3f}s"
        )
    print(f"\nSaved {len(csv_rows)} benchmark rows to '{output_csv.name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reproducible apples-to-apples retrieval ablation benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to ground-truth JSON dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to save CSV results.")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs per configuration.")
    args = parser.parse_args()
    run_benchmark(dataset_path=args.dataset, output_csv=args.output, runs=args.runs)
