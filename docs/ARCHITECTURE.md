# OmniRAG Enterprise: Architectural Specification & System Design

## 1. System Vision & Problem Statement
Most traditional RAG (Retrieval-Augmented Generation) implementations in industry follow a "Naive RAG" paradigm:
$$\text{User Query} \longrightarrow \text{Bi-Encoder Embedding} \longrightarrow \text{Top-}k\text{ Cosine Similarity} \longrightarrow \text{LLM Prompt Generation}$$

While functional for elementary demonstrations, Naive RAG breaks down under enterprise workloads due to 5 critical failure modes:
1. **Vocabulary & Acronym Mismatch**: Dense embeddings project terms into continuous geometric vector spaces. Exact part numbers, financial tickers (e.g., *NVDA*), legal citations, and rare technical acronyms (e.g., *CoWoS*, *HNSW*) get smoothed out, leading to retrieval misses.
2. **Noise Contamination & Distractor Chunks**: Top-$k$ vector similarity frequently pulls chunks that are geometrically close in vector space but contain irrelevant or contradictory statements, triggering hallucinations in the generation stage.
3. **Lost-in-the-Middle Attention Failure**: Transformer self-attention layers exhibit an attention curve that strongly favors tokens at the beginning and end of the context window. Chunks in the middle of long contexts are systematically ignored.
4. **Lack of Evidence Grading & Self-Correction**: When an indexed corpus lacks relevant information, naive systems blindly feed the closest irrelevant chunks to the LLM, producing fabricated answers.
5. **Zero Telemetry & Observability**: Production teams have no quantitative metrics to measure if retrieval was high-precision or if answers are grounded.

OmniRAG Enterprise solves all five vulnerabilities by implementing an **Agentic Corrective RAG (CRAG) and Multi-Vector Engine**.

---

## 2. End-to-End System Architecture

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

## 3. Deep Dive: Core Algorithmic Components

### 3.1 Reciprocal Rank Fusion (RRF)
Combining raw cosine similarities (bounded in $[-1, 1]$ or $[0, 1]$) with BM25 scores (unbounded in $[0, \infty)$) by simple addition fails because score distributions have completely different variances and distributions.

OmniRAG utilizes **Reciprocal Rank Fusion (RRF)**, which transforms arbitrary score distributions into an ordinal ranking function:
$$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where:
- $M = \{\text{Dense Vector}, \text{Sparse BM25}\}$
- $r_m(d)$ is the 1-based rank position of document $d$ in retrieval system $m$.
- $k$ is the ranking smoothing hyperparameter. We set $k = 60$, which prevents items at rank 1 from completely dominating lower-ranked items that appear in both systems.
- $w_m$ represents system weights ($w_{\text{dense}} = 0.65$, $w_{\text{sparse}} = 0.35$).

**Why this matters in interviews**: When an interviewer asks *"How do you combine vector search and keyword search?"*, answering *"I averaged the scores"* is an immediate red flag. Explaining RRF rank normalization and the role of smoothing constant $k$ demonstrates senior-level retrieval engineering.

---

### 3.2 Two-Stage Retrieval: Bi-Encoder vs. Cross-Encoder
- **Stage 1 (Bi-Encoder Dense + BM25)**: Independent embeddings $\vec{q} = f(q)$ and $\vec{d} = g(d)$. Enables fast sub-millisecond retrieval across millions of vectors via cosine similarity and vector indices. However, token interactions between question and passage are lost.
- **Stage 2 (Cross-Encoder Re-Ranking)**: Jointly scores $(q, d)$ through full cross-attention layers. Because Cross-Encoders evaluate word interactions directly, they achieve superior accuracy and identify semantic subtleties that Bi-Encoders miss.
- **Complexity Trade-off**: Running Cross-Encoders on 1,000,000 documents is computationally prohibitive ($O(N)$ transformer forward passes). OmniRAG balances this by using Bi-Encoder/BM25 to retrieve the top 20 candidates in $<15\text{ms}$, then applying Cross-Encoder reranking exclusively on those 20 candidates in $<25\text{ms}$.

---

### 3.3 Corrective RAG (CRAG) Agent State Machine
Standard RAG assumes the retriever always fetches useful context. CRAG introduces a self-reflective agent:
1. **Relevance Grading**: Evaluates cumulative evidence confidence $C \in [0, 1]$.
2. **Tri-State Decision Logic**:
   - **`CORRECT` ($C \ge 0.65$)**: Sufficient evidence exists. Context is passed with inline source IDs.
   - **`AMBIGUOUS` ($0.38 \le C < 0.65$)**: Query contains ambiguity or acronyms. The agent executes query rewriting (HyDE / entity expansion) and merges supplemental context.
   - **`INCORRECT` ($C < 0.38$)**: The corpus genuinely lacks factual coverage. Rather than hallucinating, the system activates the strict grounding guardrail, notifying the user of evidence absence.

---

### 3.4 Automated RAG Triad Evaluation
To ensure production quality, every query generates an automated **RAG Triad** audit:
1. **Context Precision**:
   $$\text{Precision} = \frac{|\{c \in \text{Retrieved Chunks} \mid \text{Relevance}(c) \ge \tau\}|}{|\text{Retrieved Chunks}|}$$
2. **Faithfulness (Groundedness)**:
   $$\text{Faithfulness} = \frac{|\{s \in \text{Answer Sentences} \mid s \text{ is supported by context}\}|}{|\text{Answer Sentences}|}$$
3. **Answer Relevance**:
   Measures semantic cosine alignment between the user's question and the generated answer, penalizing verbose ungrounded drift.

---

## 4. Production Latency Waterfall Analysis
Under standard query execution, the latency profile of OmniRAG Enterprise breaks down as follows:

| Stage | Latency Target | Description |
| :--- | :--- | :--- |
| **Query Transformation** | $\sim 2 - 10\text{ ms}$ | Token normalization, acronym expansion |
| **Hybrid Search (Dense + BM25)** | $\sim 15 - 35\text{ ms}$ | Parallel vector similarity + inverted index lookup |
| **Reciprocal Rank Fusion** | $\sim 2 - 5\text{ ms}$ | Ordinal rank alignment and score fusion |
| **Cross-Encoder Reranking** | $\sim 15 - 30\text{ ms}$ | Deep token interaction scoring on top-20 |
| **CRAG Evidence Grading** | $\sim 5 - 15\text{ ms}$ | Tri-state confidence evaluation |
| **LLM Generation (First Token)** | $\sim 120 - 350\text{ ms}$ | Streaming response initiation (TTFT) |
| **Total Pipeline (TTFT)** | **$< 450\text{ ms}$** | Ultra-responsive enterprise UX |
