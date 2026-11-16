from typing import List, Dict
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy
)

from src.rag_chain import ask_question, get_llm
from src.vector_store import get_embedding_model
from src.config import TOP_K_RESULTS

# Ground-truth evaluation benchmark (Question + Expected Reference Answer)
EVAL_TEST_SET: List[Dict[str, str]] = [
    {
        "user_input": "What are the requirements for a therapeutic radiation machine treatment room operating above 500 kV?",
        "reference": (
            "All protective barriers must be fixed except for access doors to the treatment room "
            "or movable beam interceptors. The control panel must be located outside the treatment room, "
            "and the room must have adequate shielding."
        ),
    },
    {
        "user_input": "What facility and equipment diagrams must applicants submit for rooms where radioactive materials are used?",
        "reference": (
            "Applicants must describe assigned areas for receipt, storage, preparation, and measurement "
            "of radioactive materials, and submit a scale diagram showing locations of shielding, proximity "
            "of radiation sources to unrestricted areas, and ventilation system schematics if materials may become airborne."
        ),
    },
    {
        "user_input": "What are the four main types of ionizing radiation discussed in basic health physics?",
        "reference": (
            "The four main types of radiation are alpha particles, beta particles, gamma rays (and X-rays), and neutrons."
        ),
    },
]

def run_evaluation(top_k: int=TOP_K_RESULTS) -> None:
    """Run the RAG pipeline on each test question and score with RAGAS metrics."""
    samples = []

    print(f"Running RAG pipeline on {len(EVAL_TEST_SET)} evaluation questions (top_k={top_k})...")
    for i, item in enumerate(EVAL_TEST_SET, start=1):
        question = item["user_input"]
        reference = item["reference"]
        print(f"\n[{i}/{len(EVAL_TEST_SET)}] Question: {question}")

        # 1. Run our Hybrid RRF + Cross-Encoder + Full-Page RAG pipeline
        result = ask_question(question, k=top_k)
        # Collect the full streamed response into a single string
        answer = "".join(result["stream"])
        contexts = [doc.page_content for doc in result["sources"]]

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

    # 2. Convert collected responses into a RAGAS EvaluationDataset
    eval_dataset = EvaluationDataset.from_list(samples)

    # 3. Wrap our existing LLM and local BGE embeddings for RAGAS Judge
    judge_llm = LangchainLLMWrapper(get_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
    ]

    print("\nRunning RAGAS evaluation metrics...")
    scores = evaluate(dataset=eval_dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=RunConfig(max_workers=2, max_retries=6))

    print("\n================ RAGAS EVALUATION RESULTS ================")
    print(scores)
    df = scores.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("Saved detailed per-question scores to 'ragas_evaluation_results.csv'!")


if __name__ == "__main__":
    run_evaluation()