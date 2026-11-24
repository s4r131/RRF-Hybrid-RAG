import os
import re
import time
from functools import lru_cache
from typing import Dict, Any, List, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config import (
    TOP_K_RESULTS,
    RERANKER_MODEL_NAME,
    GEMINI_MODEL_NAME,
    OPENAI_MODEL_NAME,
    JUDGE_GEMINI_MODEL_NAME,
    JUDGE_OPENAI_MODEL_NAME,
)
from src.vector_store import get_vector_store

SYSTEM_PROMPT = """You are an expert technical assistant answering questions strictly based on the provided reference documents.

Rules:
1. Base your answer ONLY on the provided context below.
2. If the answer is not contained in the context, state clearly that you do not have enough information in the provided documents. Do not guess or make up information.
3. ALWAYS cite the specific source document ('file_name') and 'page_number' when stating facts or regulations.
4. Be thorough and comprehensive: include ALL relevant requirements, bullet points, conditions, and details found in the context rather than summarizing briefly.

Recent Conversation History:
{chat_history}

Context:
{context}
"""

def get_llm():
    """Initialize the configured generator LLM (Google Gemini default or OpenAI)."""
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            max_retries=4,
            timeout=45,
        )
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            temperature=0.0,
            max_retries=4,
        )
    else:
        raise ValueError("No API key found! Please set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file.")


def get_judge_llm():
    """Initialize an independent judge LLM distinct from the RAG generator (no self-grading)."""
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    # Prefer cross-provider judging if both keys are configured
    if has_google and has_openai:
        return ChatOpenAI(model=JUDGE_OPENAI_MODEL_NAME, temperature=0.0, max_retries=6)

    if has_google:
        if JUDGE_GEMINI_MODEL_NAME == GEMINI_MODEL_NAME:
            raise ValueError(
                f"Judge model ({JUDGE_GEMINI_MODEL_NAME}) must differ from generator model ({GEMINI_MODEL_NAME})."
            )
        return ChatGoogleGenerativeAI(
            model=JUDGE_GEMINI_MODEL_NAME,
            thinking_level="minimal",
            max_retries=6,
            timeout=60,
        )

    if has_openai:
        if JUDGE_OPENAI_MODEL_NAME == OPENAI_MODEL_NAME:
            raise ValueError(
                f"Judge model ({JUDGE_OPENAI_MODEL_NAME}) must differ from generator model ({OPENAI_MODEL_NAME})."
            )
        return ChatOpenAI(model=JUDGE_OPENAI_MODEL_NAME, temperature=0.0, max_retries=6)

    raise ValueError("No API key found for judge LLM! Please set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file.")

def tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())

@lru_cache(maxsize=32)
def get_bm25_index(category: str = "All"):
    """Build and cache (BM25Okapi index, corpus) per category in RAM."""
    vector_store = get_vector_store()
    where_filter = {"category": category} if category and category != "All" else None
    data = vector_store.get(where=where_filter)
    corpus = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    if not corpus:
        return None, []
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in corpus])
    return bm25, corpus


def hybrid_retrieve(vector_store, query: str, k: int, category: str = "All") -> List[Document]:
    """Combine MMR Vector Search and BM25 Keyword Search using Reciprocal Rank Fusion (RRF)."""
    # 1. Dense Vector Search (MMR)
    search_kwargs: Dict[str, Any] = {"k": k, "fetch_k": k * 3}
    if category and category != "All":
        search_kwargs["filter"] = {"category": category}
    vector_docs = vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs).invoke(query)

    # 2. Sparse Keyword Search (BM25) - uses cached index for this category
    bm25, corpus =  get_bm25_index(category)
    bm25_docs: List[Document] = []
    if bm25 is not None and corpus:
        scores = bm25.get_scores(tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [corpus[i] for i in top_indices if scores[i] > 0]

    # 3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
    rrf_scores: Dict[tuple, float] = {}
    doc_map: Dict[tuple, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = (doc.metadata.get("file_name"), doc.metadata.get("page_number"), doc.metadata.get("start_index"))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = (doc.metadata.get("file_name"), doc.metadata.get("page_number"), doc.metadata.get("start_index"))
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)[: k * 2]
    return [doc_map[key] for key in sorted_keys]

@lru_cache(maxsize=1)
def get_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Load and cache the Cross-Encoder neural reranker model (local cache first)."""
    try:
        return CrossEncoder(model_name, local_files_only=True)
    except Exception:
        return CrossEncoder(model_name)

def rerank_documents(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """Score (query, chunk) pairs with a Cross-Encoder and return the top_k highest-scoring chunks."""
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]

def merge_overlapping_chunks(pairs: List[tuple]) -> str:
    """Stitch sorted (text, metadata) chunks into one page without duplicating boundary text."""
    full_page_text = ""
    for text, _ in pairs:
        if not full_page_text:
            full_page_text = text
            continue

        max_overlap = min(len(full_page_text), len(text))
        overlap_len = 0
        for size in range(max_overlap, 0, -1):
            if full_page_text.endswith(text[:size]):
                overlap_len = size
                break

        if overlap_len > 0:
            full_page_text += text[overlap_len:]
        else:
            full_page_text += "\n" + text

    return full_page_text

def expand_to_full_pages(vector_store, retrieved_docs: List[Document]) -> List[Document]:
    """For every retrieved chunk, fetch all chunks from that same PDF page in reading order."""
    seen_pages = set()
    expanded_docs: List[Document] = []

    for doc in retrieved_docs:
        file_name = doc.metadata.get("file_name")
        page_num = doc.metadata.get("page_number")
        page_key = (file_name, page_num)

        # Skip if we already reconstructed this full page
        if page_key in seen_pages or not file_name or page_num is None:
            continue
        seen_pages.add(page_key)

        # Query ChromaDB for all chunks belonging to this exact file + page_number
        results = vector_store.get(where={"$and": [{"file_name": file_name}, {"page_number": page_num}]})

        # Sort the page's chunks by start_index so the page reads top-to-bottom
        pairs = list(zip(results["documents"], results["metadatas"]))
        pairs.sort(key=lambda item: item[1].get("start_index", 0))

        # Stitch chunks together cleanly using exact suffix-prefix overlap detection
        full_page_text = merge_overlapping_chunks(pairs)

        expanded_docs.append(Document(page_content=full_page_text, metadata=doc.metadata))
        

    return expanded_docs

def format_docs_with_citations(docs: List[Document]) -> str:
    """Format retrieved Document chunks into a single string with source metadata headers."""
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "Unknown File")
        category = doc.metadata.get("category", "General")
        page_num = doc.metadata.get("page_number", "?")

        header = f"[Source {i} | File: {file_name} | Category: {category} | Page: {page_num}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)

def build_search_query(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Only merge prior user context when the new query looks like a follow-up."""
    if not chat_history:
        return query

    follow_up_markers = {"it", "its", "they", "them", "their", "that", "those", "this", "these"}
    tokens = set(tokenize(query))
    if len(query.split()) <= 5 or bool(tokens & follow_up_markers):
        user_msgs = [m["content"] for m in chat_history if m["role"] == "user"]
        if user_msgs:
            return f"{user_msgs[-1]} {query}"
    return query

def ask_question(query: str, k: int = TOP_K_RESULTS, category: str = "All", chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Retrieve top-k diverse chunks via MMR, expand to full pages, and generate a cited answer"""
    start_time = time.time()

    # Format recent chat history (last 4 messages) for follow-up content
    history_text = "None"

    if chat_history:
        recent = chat_history[-4:]
        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content'][:300]}" for m in recent)
    
    search_query = build_search_query(query, chat_history)
    print(f"\n[RAG] 1/3 Running Hybrid Search (MMR + BM25) for: '{search_query}' (top_k={k}, category='{category}')...")

    # 1. Run Hybrid Search (MMR Vector + BM25 Keyword with Reciprocal Rank Fusion)
    vector_store = get_vector_store()
    candidate_docs = hybrid_retrieve(vector_store, search_query, k=k, category=category)

    # 1b. Stage 2: Neural Cross-Encoder Reranking ( narrows 2*k candidates -> top k 
    retrieved_docs = rerank_documents(search_query, candidate_docs, top_k=k)

    # 1c. Stage 3: Expand matched chunks into their complete PDF pages so no context is cut off
    full_page_docs = expand_to_full_pages(vector_store, retrieved_docs)
    print(f"[RAG] 2/3 Hybrid RRF ({len(candidate_docs)} candidates) -> Reranked to {len(retrieved_docs)} chunks -> {len(full_page_docs)} full pages in {time.time() - start_time:.2f}s.")

    # 2. Format the retrieved chunks with their citation headers
    context_text = format_docs_with_citations(full_page_docs)

    # 3. Build the prompt and stream tokens from the LLM in real time
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    chain = prompt | get_llm()

    def stream_generator():
        llm_start = time.time()
        print("[RAG] 3/3 Streaming response from LLM...")
        for chunk in chain.stream({
            "chat_history": history_text,
            "context": context_text,
            "question": query
        }):
            if chunk.text:
                yield chunk.text
        print(f"[RAG] Done! Stream finished in {time.time() - llm_start:.2f}s (Total: {time.time() - start_time:.2f}s).")

    return {"stream": stream_generator(), "sources": full_page_docs}