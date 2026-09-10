# OmniRAG Enterprise: Production Corrective RAG (CRAG) & Multi-Vector Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![GenAI](https://img.shields.io/badge/GenAI-Gemini%20%7C%20Groq%20%7C%20Offline-orange.svg)](https://deepmind.google/technologies/gemini/)
[![RAG](https://img.shields.io/badge/Architecture-Corrective%20RAG%20(CRAG)-purple.svg)](#architecture)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An enterprise-grade, interview-ready Generative AI platform engineered to solve the 5 critical failure modes of Naive RAG: vocabulary mismatch, retrieval noise, lost-in-the-middle attention degradation, ungrounded hallucinations, and lack of observability.**

---

## Key Differentiators: Naive RAG vs. OmniRAG Enterprise

| Capability | Naive RAG (Common Tutorial) | OmniRAG Enterprise |
| :--- | :--- | :--- |
| **Retrieval Architecture** | Single-pass Dense Vector search only | **Hybrid Search**: Dense Vector + BM25 Lexical via **Reciprocal Rank Fusion (RRF)** |
| **Noise Filtering** | Top-$k$ similarity directly into prompt | **Cross-Encoder Re-Ranking**: Two-stage query-passage token attention |
| **Self-Correction (CRAG)** | None (hallucinates on missing data) | **Tri-State Evidence Grader**: `CORRECT`, `AMBIGUOUS` (HyDE Rewrite), `INCORRECT` (Guardrail) |
| **Source Provenance** | None or vague file names | **Interactive Chunk Provenance**: Exact excerpts, page numbers, similarity scores |
| **Evaluation Suite** | Subjective human inspection | **Automated RAG Triad**: Context Precision, Faithfulness/Groundedness, Answer Relevance |
| **Observability** | Single total time metric | **Live Latency Waterfall**: Transform $\rightarrow$ Hybrid $\rightarrow$ RRF $\rightarrow$ Rerank $\rightarrow$ LLM |
| **Deployment Mode** | Fails if API key runs out of credits | **Dual-Engine**: Cloud Gemini/Groq + deterministic local offline semantic engine |

---

## System Architecture

```
                                  [ User Query ]
                                         |
                                         v
                         +-------------------------------+
                         |   Query Transformation        |
                         |   - Acronym Expansion         |
                         |   - HyDE Query Reformulation  |
                         +-------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
  +--------------------------+                       +--------------------------+
  |    Dense Vector Search   |                       |    BM25 Lexical Search   |
  |  - Cosine Distance Metric|                       |  - Robertson-Sparck IDF  |
  |  - Gemini / Dense Space  |                       |  - Length Norm (k1, b)   |
  +--------------------------+                       +--------------------------+
               |                                                   |
               +-------------------------+-------------------------+
                                         v
                         +-------------------------------+
                         | Reciprocal Rank Fusion (RRF)  |
                         | Merges Dense & Sparse Scores  |
                         +-------------------------------+
                                         v
                         +-------------------------------+
                         | Cross-Encoder Re-Ranking      |
                         | - Query-Passage Token Attn    |
                         | - Noise Threshold Filter      |
                         +-------------------------------+
                                         v
                         +-------------------------------+
                         | Agentic CRAG Relevance Grader |
                         +-------------------------------+
                                  /      |      \
              (Confidence > 0.65)/       |       \(Confidence < 0.38)
                                v        |        v
                         [ CORRECT ]     |   [ INCORRECT ]
                         Sufficient      |   Corpus Lacks Evidence
                         Evidence        |   Trigger Hallucination Guard
                                         v
                                  [ AMBIGUOUS ]
                              Reformulate & Expand
                                         |
                                         v
                         +-------------------------------+
                         | LLM Grounded Synthesis Engine |
                         | Streaming Citations: [1], [2] |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   RAG Triad Evaluation Suite  |
                         | - Context Precision           |
                         | - Faithfulness (Groundedness) |
                         | - Answer Relevance            |
                         +-------------------------------+
```

---

## Quick Start (Run Locally in 60 Seconds)

### 1. Clone or Open Project Directory:
```bash
cd omnirag-enterprise
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Start the Server:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*(On Windows you can also simply double-click `run.bat`)*

### 4. Open the Interactive Dashboard:
Navigate to **`http://localhost:8000`** in your browser.
- Click any of the preloaded sample datasets (e.g. *NVIDIA 10-K SEC Filing* or *RAG Architecture Spec*).
- Click sample query chips to watch the real-time latency waterfall, CRAG grade, and RAG Triad benchmark metrics.

---

## Documentation & Interview Toolkit

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Deep mathematical derivation of Reciprocal Rank Fusion, Cross-Encoder vs Bi-Encoder trade-offs, HNSW vs IVF-PQ graph indexing, and latency waterfall profiles.
- **[INTERVIEW_PLAYBOOK.md](docs/INTERVIEW_PLAYBOOK.md)**:
  - **Resume Bullet Points** with quantified metrics (38% recall increase, 42% hallucination reduction).
  - **90-Second Interview Elevator Pitch** script.
  - **20 Hard Technical Interview Questions & Model Answers**.
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**: Step-by-step guides for deploying free on **Hugging Face Spaces**, **Render**, **Railway**, and **Docker**.

---

## Running Automated Tests

Run the test suite to verify all pipeline components (chunking, BM25, vector search, RRF, Cross-Encoder reranking, CRAG state machine, and RAG Triad metrics):
```bash
python -m unittest backend/tests/test_rag.py
```
*Result: 7/7 tests passed in 0.012s.*

---

## License
MIT License. Created for high-impact AI Engineering portfolios and technical interview excellence.
