import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.config import BASE_DIR, GEMINI_MODEL_NAME, JUDGE_OPENAI_MODEL_NAME, OPENAI_MODEL_NAME, TOP_K_RESULTS
from src.rag_chain import ask_question, get_judge_llm
from src.vector_store import get_embedding_model

DEFAULT_DATASET_PATH = BASE_DIR / "eval_dataset.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "ragas_evaluation_results.csv"


def load_eval_dataset(dataset_path: Path = DEFAULT_DATASET_PATH, limit: int = 0) -> List[Dict[str, str]]:
    """Load ground-truth questions and reference answers from eval_dataset.json."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit > 0 else data


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    top_k: int = TOP_K_RESULTS,
    limit: int = 0,
    runs: int = 1,
) -> None:
    """Run the RAG pipeline on the evaluation dataset and grade with an independent judge LLM."""
    test_set = load_eval_dataset(dataset_path=dataset_path, limit=limit)
    judge_lc_llm = get_judge_llm()
    judge_llm = LangchainLLMWrapper(judge_lc_llm)
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    gen_name = GEMINI_MODEL_NAME if os.getenv("GOOGLE_API_KEY") else OPENAI_MODEL_NAME
    judge_name = getattr(judge_lc_llm, "model", getattr(judge_lc_llm, "model_name", JUDGE_OPENAI_MODEL_NAME))
    print(f"Generator LLM: {gen_name} | Independent Judge LLM: {judge_name}")

    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(strictness=1),
    ]

    all_run_dfs = []
    for run_idx in range(1, runs + 1):
        samples = []
        print(f"\n=== RAGAS Evaluation Run {run_idx}/{runs} ({len(test_set)} questions, top_k={top_k}) ===")
        for i, item in enumerate(test_set, start=1):
            question = item["user_input"]
            reference = item["reference"]
            print(f"[{i}/{len(test_set)}] {question}")

            result = ask_question(question, k=top_k)
            answer = "".join(result["stream"])
            contexts = [doc.page_content for doc in result["sources"]]

            samples.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            })

        eval_dataset = EvaluationDataset.from_list(samples)
        scores = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(max_workers=4, max_retries=6, max_wait=20, timeout=300),
        )
        print(f"\nRun {run_idx} Results: {scores}")
        df_run = scores.to_pandas()
        df_run.insert(0, "run", run_idx)
        all_run_dfs.append(df_run)

    combined_df = pd.concat(all_run_dfs, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)

    metric_cols = [c for c in ["llm_context_precision_with_reference", "context_recall", "faithfulness", "answer_relevancy"] if c in combined_df.columns]
    print(f"\n================ AGGREGATE RAGAS RESULTS ({runs} Run(s) x {len(test_set)} Questions) ================")
    for col in metric_cols:
        valid = combined_df[col].dropna()
        print(f"  {col}: mean={valid.mean():.4f} (n={len(valid)}/{len(combined_df)})")
    print(f"Saved detailed per-question scores to '{output_csv.name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with an independent judge LLM.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to eval_dataset.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to output CSV.")
    parser.add_argument("--top-k", type=int, default=TOP_K_RESULTS, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of queries (0 = all).")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated evaluation runs.")
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_csv=args.output,
        top_k=args.top_k,
        limit=args.limit,
        runs=args.runs,
    )
