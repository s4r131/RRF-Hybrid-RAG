# RRF-Hybrid-RAG: Multi-Stage Retrieval & Evaluation Engine

A domain-agnostic, production-engineered **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **ChromaDB**, **HuggingFace (`BAAI/bge-small-en-v1.5`)**, **BM25 (`rank-bm25`)**, **Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**, **RAGAS**, and **Google Gemini (`gemini-3.5-flash-lite` / `gemini-3.5-flash`) / OpenAI (`gpt-4o-mini` / `gpt-4o`)**.

Designed to scale across hundreds of PDFs and thousands of pages of technical manuals, regulatory specifications, or research documents, this pipeline was evaluated across **36 ground-truth queries spanning 36 unique PDFs and 14 categories repeated across 3 runs (540 evaluations)** over a **17,444-chunk (6,579-page, 158-file)** corpus. At the same `k=10` cutoff, Hybrid RRF + Cross-Encoder Reranking raised **Source Hit@10 from `91.7%` to `100.0%`**, **MRR from `0.772` to `0.905`**, and **Target-Fact Recall from `88.7%` to `95.6%`** (with Full-Page Expansion) at **`0.224s` average local retrieval latency**, while automatically preserving folder hierarchies as filterable category metadata.

---

## Benchmark & Ablation Results

Reproduced locally via `python benchmark_retrieval.py --runs 3` on **`eval_dataset.json`** (**36 queries × 3 runs = 540 total runs**, exported to [`retrieval_benchmark_results.csv`](retrieval_benchmark_results.csv)) over the indexed **17,444-chunk, 6,579-page, 158-file** corpus:

### 1. Apples-to-Apples Retrieval Pipeline Ablation (`k=5` and `k=10`)

| Pipeline Configuration | `K` | Source Hit@K | MRR | Target-Fact Recall | Avg Context Returned | Avg Local Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector Baseline (`raw chunks`)** | `5` | `88.9%` | `0.767` | `84.5%` | `4,382` chars | `0.009s ± 0.001s` |
| **Hybrid RRF + Cross-Encoder + Full-Page (`v2`)** | `5` | **`100.0%`** | **`0.913`** | `90.7%` | `10,881` chars (**2.5x**) | `0.163s ± 0.027s` |
| **Dense Vector Baseline (`raw chunks`)** | `10` | `91.7%` | `0.772` | `88.7%` | `8,708` chars | `0.011s ± 0.002s` |
| **Hybrid RRF + Cross-Encoder (`chunks only`)** | `10` | **`100.0%`** | **`0.905`** | `93.5%` | `9,102` chars (**1.05x**) | `0.190s ± 0.053s` |
| **Production (`Hybrid RRF + Cross-Encoder + Full-Page`)** | `10` | **`100.0%`** | **`0.905`** | **`95.6%`** | **`21,253` chars (2.4x)** | **`0.224s ± 0.048s`** |

### 2. Component Isolation & Engineering Impact

| Optimization | Before | After | Measured Gain |
| :--- | :---: | :---: | :---: |
| **Hybrid RRF + Cross-Encoder (`k=10`, chunks only vs. Dense@10)** | `91.7%` Hit@10 / `0.772` MRR / `88.7%` Recall | **`100.0%` Hit@10 / `0.905` MRR / `93.5%` Recall** | **+8.3 pts** Hit@10, **+0.133** MRR at ~equal char count |
| **Full-Page Expansion (`Hybrid@10 chunks` $\rightarrow$ `+ Full-Page`)** | `93.5%` Fact Recall (`9,102` chars) | **`95.6%` Fact Recall (`21,253` chars)** | Completes cut-off tables & lists cleanly |
| **BM25 Index Caching (`@lru_cache(maxsize=32)`)** | `556.1 ms` / query (17,444-chunk rebuild) | **`37.25 µs`** warm lookup (`1.80s` cold load) | **~14,900x faster** warm index lookup |

### Key Takeaways & Evaluation Design
* **Consistent Gains at Identical `K`**: Holding `k` constant, Hybrid RRF + Cross-Encoder beats Dense Vector search at both cutoffs—raising Hit@5 (`88.9%` $\rightarrow$ **`100.0%`**) and MRR@5 (`0.767` $\rightarrow$ **`0.913`**), as well as Hit@10 (`91.7%` $\rightarrow$ **`100.0%`**) and MRR@10 (`0.772` $\rightarrow$ **`0.905`**).
* **Hybrid RRF Rescues Sparse Identifier Misses**: At `k=10`, Dense Vector similarity alone missed all chunks from **3 target files** containing specific document/form identifiers (`NRC Regulatory Guide 8-40.pdf`, `NRC_313a(amp).pdf`, and `moe section 1.pdf`). BM25 + RRF + Cross-Encoder recovered all three (**`100.0%` Hit@10**) before any full-page expansion was applied.
* **Isolating Reranking from Context Expansion**: Comparing `Hybrid@10 (chunks only)` (`9,102` chars) against `Dense@10` (`8,708` chars) proves that nearly all ranking precision (**`0.905` vs `0.772` MRR**) and +4.8 percentage points of Fact Recall (`88.7%` $\rightarrow$ `93.5%`) come directly from RRF + Cross-Encoder reranking rather than larger context windows; Stage 3 Full-Page Expansion then lifts Fact Recall further to **`95.6%`**.
* **Independent-Judge RAGAS Suite (`evaluate.py`)**: Answer generation (`get_llm()`, defaulting to `gemini-3.5-flash-lite` or `gpt-4o-mini`) and RAGAS evaluation (`get_judge_llm()`, defaulting to `gemini-3.5-flash` or cross-provider `gpt-4o` when both keys are set) strictly enforce **distinct models** (raising a `ValueError` if equal) so the generator never grades its own outputs. The suite loads all 36 queries from `eval_dataset.json`, supports `--runs N` repeated averages with `ResponseRelevancy(strictness=1)` and `RunConfig(max_workers=4, timeout=300)`, and exports per-question scores to `ragas_evaluation_results.csv`. End-to-end 36-query RAGAS execution requires an API billing tier beyond Google's 20-request/day Free Tier Flash quota (approx. 11+ judge LLM calls per full-page query).

---

## System Architecture & Pipeline Workflow

The system is decoupled into two independent pipelines:
1. **Offline Ingestion Pipeline (`ingest.py`)**: Scans the target document directory, filters out excluded/duplicate folders in $O(1)$ time, extracts page text and category metadata via `PyMuPDF`, splits pages into overlapping chunks with exact character start indices, embeds chunks locally using `BAAI/bge-small-en-v1.5`, and persists them to ChromaDB in SQLite-safe batches.
2. **Online Multi-Stage Retrieval & Generation Pipeline (`app.py` $\rightarrow$ `src/rag_chain.py`)**:
   - **Stage 1 (Candidate Retrieval)**: Combines **Dense Vector Search (MMR)** and **Sparse Keyword Search (BM25)** via **Reciprocal Rank Fusion (RRF)** to select the top $2k$ candidate chunks.
   - **Stage 2 (Neural Reranking)**: Scores all $2k$ `(query, chunk)` pairs jointly through a **Cross-Encoder Transformer** (`ms-marco-MiniLM-L-6-v2`) to select the top $k$ highest-relevance chunks.
   - **Stage 3 (Full-Page Context Reconstruction & Streaming)**: Reconstructs full PDF pages seamlessly without overlap duplication, and streams grounded, page-cited answers in real time.

```mermaid
flowchart TD
    subgraph Offline["1. Offline Ingestion Pipeline (ingest.py)"]
        A["Raw PDF Directory<br/>(Structured Category Subfolders)"] -->|"O(1) Set Filter<br/>PyMuPDFLoader"| B["Enriched Document Pages<br/>(file_name, category, page_number)"]
        B -->|"RecursiveCharacterTextSplitter<br/>(size=1000, overlap=200, start_index)"| C["Overlapping Text Chunks"]
        C -->|"Local HuggingFaceEmbeddings<br/>(BAAI/bge-small-en-v1.5, L2 Normalized)"| D[("ChromaDB Vector Store<br/>(Batched 500/step)")]
    end

    subgraph Online["2. Online Hybrid Retrieval, Reranking & Generation Pipeline (app.py & src/rag_chain.py)"]
        Q["User Question +<br/>Recent Chat History"] -->|"Query Contextualization"| SQ["Contextualized Search Query"]
        SQ -->|"Dense Semantic Search"| V["MMR Vector Search<br/>(Top-K Diverse Chunks)"]
        SQ -->|"Sparse Keyword Search"| K["BM25Okapi Keyword Search<br/>(Exact Codes & Identifiers)"]
        D -.-> V
        D -.-> K
        V & K -->|"Reciprocal Rank Fusion (RRF)"| RRF["Top-2K Hybrid Candidates"]
        RRF -->|"Cross-Encoder Transformer<br/>(ms-marco-MiniLM-L-6-v2)"| CE["Top-K Reranked Chunks"]
        CE -->|"Metadata Filter (file_name + page_number)<br/>+ Overlap Slicing"| FP["Reconstructed Full PDF Pages"]
        FP -->|"LCEL Prompt to LLM<br/>(chain.stream)"| ST["Live Token Stream +<br/>Expandable Source Citations"]
    end
```

---

## Mathematical Foundations of the Retrieval Pipeline

### 1. L2 Vector Normalization & Cosine Similarity (`src/vector_store.py`)
Each text chunk is mapped to a $d = 384$-dimensional dense vector $\mathbf{v} \in \mathbb{R}^{384}$ using `BAAI/bge-small-en-v1.5`. By setting `encode_kwargs={"normalize_embeddings": True}`, every vector is L2-normalized to unit length $\lVert\mathbf{v}\rVert_2 = 1$:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{d} v_i^2}}
$$

Because both the query vector $\hat{\mathbf{q}}$ and document vectors $\hat{\mathbf{d}}$ have unit norm, **Cosine Similarity** reduces to a fast hardware-accelerated dot product:

$$
\text{sim}(\hat{\mathbf{q}}, \hat{\mathbf{d}}) = \cos(\theta) = \frac{\hat{\mathbf{q}} \cdot \hat{\mathbf{d}}}{\lVert\hat{\mathbf{q}}\rVert_2 \, \lVert\hat{\mathbf{d}}\rVert_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \sum_{i=1}^{384} \hat{q}_i \hat{d}_i
$$

---

### 2. Maximal Marginal Relevance — MMR (`src/rag_chain.py`)
Standard top-k similarity search often retrieves near-identical chunks from the same repetitive document. To maximize both **relevance** to the query $Q$ and **diversity** across documents, our vector retriever uses **Maximal Marginal Relevance (MMR)**.

First, it fetches a candidate pool $C$ of size `fetch_k` $= 3k$ chunks. Then it iteratively selects $k$ chunks into the selected set $S$ according to:

$$
\text{MMR} = \arg\max_{d_i \in C \setminus S} \left[ \lambda \, \text{sim}(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \text{sim}(d_i, d_j) \right]
$$

* $\text{sim}(d_i, Q)$ rewards high semantic similarity to the user's question.
* $\max_{d_j \in S} \text{sim}(d_i, d_j)$ penalizes candidates that are redundant with chunks already selected in $S$.

---

### 3. Sparse Keyword Scoring via Okapi BM25 (`src/rag_chain.py`)
While dense embeddings capture conceptual meaning, they can wash out exact technical identifiers, part numbers, or section codes (e.g., `"ISO-9001"`, `"Section 4.2.1"`, `"Form-1040"`). We pair dense retrieval with **Okapi BM25** (`BM25Okapi`), which scores a document chunk $D$ for a tokenized query $Q = (q_1, q_2, \dots, q_n)$ as:

$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

where:
* $f(q_i, D)$ is the term frequency of token $q_i$ in chunk $D$.
* $|D|$ is the token length of chunk $D$, and $\text{avgdl}$ is the average chunk length across the corpus.
* $k_1 \approx 1.5$ controls term-frequency saturation, and $b \approx 0.75$ normalizes for document length.
* $\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$ is the **Inverse Document Frequency**, which assigns high weight to rare technical terms and near-zero weight to common stop words.

---

### 4. Hybrid Merging via Reciprocal Rank Fusion — RRF (`src/rag_chain.py`)
Cosine similarity scores in $[-1, 1]$ and unbounded BM25 scores in $[0, \infty)$ live on incompatible numerical scales and cannot be summed directly. We merge the ranked lists from MMR Vector Search ($R_{\text{vec}}$) and BM25 Keyword Search ($R_{\text{bm25}}$) using **Reciprocal Rank Fusion (RRF)**:

$$
\text{RRF}(d) = \sum_{m \in \lbrace\text{vec},\, \text{bm25}\rbrace} \frac{1}{\eta + \text{rank}_m(d)}
$$

where $\eta = 60$ is the standard RRF smoothing constant and $\text{rank}_m(d) \in [0, k-1]$ is the chunk's 0-based rank in retriever $m$.
* Chunks that rank highly in **both** semantic meaning (MMR) and exact keyword matching (BM25) receive contributions from both terms and rise to the top of the candidate pool $C_{\text{RRF}}$ (of size $2k$).

---

### 5. Stage 2 Neural Reranking via Cross-Encoder (`rerank_documents`)
Embedding models (bi-encoders) encode the question and document chunks *separately* into vectors, which makes them fast for searching thousands of chunks but can miss subtle context between the question and the passage.

Once Hybrid RRF narrows the corpus down to the top `2 * k` candidates, we pass those `[query, chunk_text]` pairs together through a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which reads the question and chunk side-by-side to predict a direct relevance score and keeps the top `k` winners:

```python
pairs = [[query, doc.page_content] for doc in docs]
scores = reranker.predict(pairs)
scored_docs = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
return [doc for _, doc in scored_docs[:top_k]]
```

---

### 6. Seamless Full-Page Context Expansion (`expand_to_full_pages` & `merge_overlapping_chunks`)
Technical specifications and tables frequently span an entire page (3,000+ characters). Retrieving isolated 1,000-character chunks can cut off bulleted lists mid-way.

For every top-k reranked chunk, we query ChromaDB's metadata index (`file_name` and `page_number`) to fetch **all chunks belonging to that page**, sort them in top-to-bottom reading order by `start_index`, and run exact suffix-prefix matching to remove the 200-character `CHUNK_OVERLAP` seam without ever slicing mid-word (even when splitters strip boundary whitespace):

```python
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
```
This reconstructs the **exact original PDF page** cleanly before sending the context to the LLM without duplicating suffix-prefix overlap seams across multi-chunk pages.

---

## Key Software Engineering & Python Patterns Used

### 1. `@lru_cache` Singleton & Per-Category Index Caching
In `src/vector_store.py` and `src/rag_chain.py`, we decorate expensive initialization and index-building functions with `@lru_cache` from Python's `functools`:
```python
@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    ...

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    ...

@lru_cache(maxsize=32)
def get_bm25_index(category: str = "All"):
    ...

@lru_cache(maxsize=1)
def get_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    ...
```
* **Why?** Without caching, every user question in Streamlit would reload 199 embedding model weights from disk and re-tokenize all 17,444 chunks to rebuild `BM25Okapi` (`556.1 ms` per turn).
* **How it works:** `@lru_cache` memoizes models in RAM (`maxsize=1`) and caches pre-built `BM25Okapi` indices per category filter (`maxsize=32`), reducing warm index lookup time to **`37.25 µs` (~14,900x faster)**.

### 2. Local-First Embedding Loading with Automatic Cloud Fallback
In `get_embedding_model()`, we attempt `model_kwargs={"local_files_only": True}` inside a `try / except` block:
* If the model is already cached locally, it loads in milliseconds with **zero network requests** to HuggingFace Hub.
* If the repository is cloned onto a fresh machine where the model is not yet cached, the `except` block automatically falls back to downloading and caching `BAAI/bge-small-en-v1.5`.

### 3. $O(1)$ Set Intersection for Duplicate Directory Filtering
In `src/document_loader.py`, we filter out duplicate backup folders (`EXCLUDED_DIRS`) using Python set intersection (`__and__`):
```python
def should_skip_path(file_path: Path) -> bool:
    return bool(set(file_path.parts) & EXCLUDED_DIRS)
```

### 4. Python Generators (`yield`) for Low-TTFT Token Streaming
In `src/rag_chain.py`, instead of blocking for 8+ seconds with `chain.invoke()`, `ask_question()` defines a generator (`stream_generator`) that iterates over `chain.stream(...)` and `yield`s text chunks as they arrive from the LLM. In `app.py`, `st.write_stream(result["stream"])` renders tokens live on screen with $< 1\text{s}$ Time-To-First-Token (TTFT).

### 5. Follow-Up Context Isolation (`build_search_query`)
Rather than blindly concatenating prior chat turns onto every query (which pollutes BM25 and reranking when users switch topics), `build_search_query()` checks query length ($\le 5$ words) and pronoun markers via an $O(1)$ set intersection (`tokens & follow_up_markers`) to merge prior context *only* on true follow-up questions.

---

## Project Structure

```text
RRF-Hybrid-RAG/
├── data/                            # Raw PDF dataset organized by category subfolders (git-ignored)
├── chroma_db/                       # Persisted ChromaDB vector store on disk (git-ignored)
├── src/                             # Core modular RAG library
│   ├── __init__.py                  # Package marker
│   ├── config.py                    # Paths, chunk hyperparameters, generator & judge model configs
│   ├── document_loader.py           # Recursive PDF scanner, deduplicator, and metadata enricher
│   ├── chunker.py                   # RecursiveCharacterTextSplitter with start_index tracking
│   ├── vector_store.py              # Cached local BGE embeddings & batched ChromaDB persistence
│   └── rag_chain.py                 # Hybrid Search (MMR + BM25 RRF), Cross-Encoder, page stitching & streaming
├── ingest.py                        # CLI entry point to build/rebuild the vector store
├── app.py                           # Interactive Streamlit chat UI with startup guards, category filters & citations
├── benchmark_retrieval.py           # Reproducible apples-to-apples retrieval ablation runner (--runs 3)
├── evaluate.py                      # Independent-judge RAGAS evaluation suite (--limit, --runs)
├── eval_dataset.json                # 36 verified ground-truth Q&A queries spanning 36 PDFs & 14 categories
├── retrieval_benchmark_results.csv  # Saved 540-row ablation benchmark dataset (36 queries x 3 runs x 5 configs)
├── .env.example                     # Template for environment variables (API keys, judge models, DATA_DIR)
├── .gitignore                       # Excludes secrets (.env), .venv, chroma_db, and raw PDFs
├── requirements.txt                 # Project dependencies
└── README.md                        # Architecture, benchmark data & mathematical documentation
```

---

## Quickstart Guide

### 1. Clone the Repository & Create a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Your Environment
Copy `.env.example` to `.env` and add your **Google Gemini** (free via [Google AI Studio](https://aistudio.google.com/apikey)) or **OpenAI** API key (and optionally set `DATA_DIR` to point to any custom PDF folder):
```powershell
Copy-Item .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_actual_api_key_here
# DATA_DIR=path/to/your/pdf_folder
```

### 3. Run the Offline Ingestion Pipeline
Place your PDFs inside your configured data directory (`DATA_DIR` in `.env` or `src/config.py`) and run:
```powershell
python ingest.py
```

### 4. Launch the Interactive Web App
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser to ask questions, filter by document category, tune `Top-K` retrieval live, and inspect full-page source citations.

### 5. Reproduce the Retrieval Ablation Benchmark
Run the reproducible apples-to-apples dense vs. hybrid retrieval comparison (`Dense@5`, `Hybrid@5`, `Dense@10`, `Hybrid@10 chunks-only`, and `Hybrid@10 + Full-Page`) across 36 queries and 3 repeated runs:
```powershell
python benchmark_retrieval.py --runs 3
```
This exports all 540 per-run records to `retrieval_benchmark_results.csv` and prints aggregate Hit@K, MRR, Target-Fact Recall, and latency.

### 6. Run the Independent-Judge RAGAS Suite
Evaluate end-to-end Context Precision, Context Recall, Faithfulness, and Answer Relevancy using `eval_dataset.json` and a strictly distinct judge model (`gpt-4o` cross-provider when both keys are set, or `gemini-3.5-flash` judging `gemini-3.5-flash-lite`):
```powershell
python evaluate.py --runs 2
```
Use `--limit N` (for a subset) or `--runs R` (to report multi-run averages) and inspect detailed per-question metrics in `ragas_evaluation_results.csv`. Note: running all 36 queries requires a paid/Tier-1 API quota due to multi-call per-page RAGAS grading (~11+ judge calls per query).
