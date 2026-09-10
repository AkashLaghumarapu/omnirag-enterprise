"""
OmniRAG Enterprise - Document Loader & Semantic Chunker
Handles parsing of TXT, Markdown, CSV, and PDF documents with sliding window chunking,
metadata tracking, and pre-packaged enterprise sample datasets.
"""

import os
import re
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    metadata: Dict[str, Any] = {}

class DocumentLoader:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str, doc_title: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Splits raw text into semantically aware overlapping chunks,
        detecting section headers to preserve contextual hierarchy.
        """
        if metadata is None:
            metadata = {}
        
        # Clean text and normalize bullet characters
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for b in ["\uf0b7", "\u2022", "\u25cf", "\u2023", "\u25b6"]:
            text = text.replace(b, "\n- ")
        
        # Detect sections (Markdown headers, item numbers, or capitalized sections like SKILLS, PROJECTS, EDUCATION)
        lines = text.split("\n")
        current_section = "General Overview"
        sections: List[Dict[str, Any]] = []
        current_buffer = []

        header_pattern = re.compile(
            r"^(#{1,4}\s+.*|[A-Z0-9\s\-_]{4,}:|ITEM\s+\d+[A-Z]?\..*|\b(CAREER OBJECTIVE|OBJECTIVE|EXPERIENCE|INTERNSHIPS|PROJECTS|EDUCATION|TECHNICAL SKILLS|SKILLS|CERTIFICATIONS|ACHIEVEMENTS|AWARDS)\b.*)",
            re.IGNORECASE
        )

        for line in lines:
            stripped = line.strip()
            if header_pattern.match(stripped):
                if current_buffer:
                    sections.append({
                        "section": current_section,
                        "content": "\n".join(current_buffer)
                    })
                    current_buffer = []
                current_section = stripped.lstrip("#").strip()
            else:
                if stripped:
                    current_buffer.append(stripped)

        if current_buffer:
            sections.append({
                "section": current_section,
                "content": "\n".join(current_buffer)
            })

        # If no explicit sections found, treat whole text as one
        if not sections:
            sections = [{"section": "Document Content", "content": text}]

        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_name = sec["section"]
            sec_text = sec["content"]
            
            # Slide window over section text
            words = sec_text.split()
            if not words:
                continue

            # Convert character window to approximate word counts
            words_per_chunk = max(30, self.chunk_size // 5)
            overlap_words = max(5, self.chunk_overlap // 5)
            step = max(1, words_per_chunk - overlap_words)

            for i in range(0, len(words), step):
                chunk_words = words[i:i + words_per_chunk]
                if not chunk_words:
                    break
                chunk_content = " ".join(chunk_words)
                
                # Deduplicate tiny trailing chunks
                if len(chunk_words) < 15 and chunks:
                    chunks[-1].content += " " + chunk_content
                    break

                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                page_num = metadata.get("page_number", 1 + (chunk_idx // 3))

                chunk = DocumentChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_index=chunk_idx,
                    content=chunk_content,
                    page_number=page_num,
                    section_header=sec_name,
                    metadata={
                        **metadata,
                        "token_count": len(chunk_words),
                        "char_count": len(chunk_content)
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def load_pdf(self, file_path: str, doc_title: Optional[str] = None) -> List[DocumentChunk]:
        """Loads and extracts text from a PDF file with page numbering."""
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            doc_id = str(uuid.uuid4())[:8]
            title = doc_title or os.path.basename(file_path)
            all_chunks = []

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                page_chunks = self.chunk_text(
                    text=page_text,
                    doc_id=doc_id,
                    doc_title=title,
                    metadata={"page_number": page_idx + 1, "source": file_path}
                )
                all_chunks.extend(page_chunks)

            return all_chunks
        except Exception as e:
            # Fallback to plain text read if pypdf fails
            return self.load_text_file(file_path, doc_title)

    def load_text_file(self, file_path: str, doc_title: Optional[str] = None) -> List[DocumentChunk]:
        """Loads a plain text or markdown file."""
        doc_id = str(uuid.uuid4())[:8]
        title = doc_title or os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.chunk_text(content, doc_id=doc_id, doc_title=title, metadata={"source": file_path})


# Pre-packaged enterprise datasets for instant high-impact demos
SAMPLE_ENTERPRISE_DATASETS = {
    "sec_10k_nvidia": {
        "title": "NVIDIA Corporation - Form 10-K Financial Disclosures & AI Infrastructure",
        "description": "Annual report detailing Data Center revenue, Hopper/Blackwell compute architecture, supply chain constraints, and CapEx risks.",
        "content": """ITEM 1. BUSINESS
Overview
NVIDIA pioneered GPU-accelerated computing to help solve the most challenging computational problems. The architecture combines hardware, interconnects, CUDA software, and full-stack acceleration libraries for high-performance computing (HPC) and artificial intelligence (AI).

Data Center Segment Highlights
Our Compute & Networking segment revenue was $47.5 billion, up 217% year-over-year, driven primarily by the hyperscale deployment of the NVIDIA HGX platform based on Hopper architecture (H100 and H200 Tensor Core GPUs). Data center demand accelerated due to widespread enterprise adoption of generative AI foundation models, large language models (LLMs), and recommender systems.
Customer concentration in the Data Center segment remains notable: approximately 19% of total revenue was derived from Cloud Service Provider A, and 14% from Cloud Service Provider B.

Next-Generation Architecture: Blackwell B200
The Blackwell platform introduces second-generation Transformer Engine technology, NVLink 5 interconnect offering 1.8TB/s bidirectional throughput per GPU, and fifth-generation Tensor Cores. It delivers up to 4x faster training and 30x faster inference throughput compared to the Hopper architecture for trillion-parameter scale generative AI models, while reducing energy consumption by up to 25x.

ITEM 1A. RISK FACTORS
Supply Chain and Manufacturing Constraints
We rely on independent third-party foundries, primarily Taiwan Semiconductor Manufacturing Company (TSMC), to manufacture our semiconductor wafers using advanced packaging technologies (CoWoS - Chip-on-Wafer-on-Substrate). Substrate shortages, silicon wafer capacity limitations, and assembly lead times may constrain our ability to satisfy hyper-scale customer orders in timely fashion. Any geopolitical instability or natural catastrophe affecting Taiwan or advanced packaging facilities would severely disrupt production and material revenue recognition.

Export Controls and Geopolitical Trade Compliance
New licensing regulations promulgated by the U.S. Bureau of Industry and Security (BIS) impose worldwide performance-density thresholds on high-performance accelerators, restricting sales of A100, H100, and customized variants (such as A800 and H800) into China and other designated jurisdictions without explicit governmental export licenses. These geopolitical export controls have materially reduced revenue from affected markets, requiring NVIDIA to architect localized compliance-tailored alternatives.

CapEx and Energy Grid Capacity
Customers deploying high-density AI clusters face significant infrastructure challenges, including electrical substation transformer availability, high-voltage power distribution constraints, and liquid cooling deployment timelines. High capital expenditure (CapEx) commitments from major hyperscalers could experience cyclical digestion periods if monetization velocity of generative AI applications lags infrastructure capital deployments.
"""
    },
    "rag_architecture_spec": {
        "title": "Production RAG Systems: Architecture, Indexing, and Hallucination Mitigations",
        "description": "Technical engineering whitepaper analyzing Hybrid Search (BM25 + Vector), Cross-Encoder Reranking, CRAG self-correction, and HNSW graph indexing.",
        "content": """# Architectural Blueprint: Enterprise Corrective RAG (CRAG)

## 1. Limitations of Naive RAG in Production
Standard Naive RAG architectures (Chunk -> Dense Embedding -> Top-k Vector Similarity -> LLM Synthesis) exhibit critical production failure modes:
1. Lexical Mismatch & Acronym Blindness: Dense semantic models map text into continuous geometric representations. However, exact queries involving SKU numbers, financial tickers, specialized medical nomenclatures, or rare code symbols frequently fail due to nearest-neighbor semantic smoothing.
2. Noise Contamination & Distractor Chunks: Retrieving top-k chunks purely by cosine similarity often introduces irrelevant or contradictory passages. Ingesting distractors degrades LLM generation fidelity, precipitating hallucination.
3. Lost-in-the-Middle Phenomenon: Research demonstrates that transformer attention architectures attend strongly to tokens located at the extreme beginning and end of the context window, while information placed in the middle is disproportionately forgotten.
4. Non-Deterministic Confidence: Naive RAG yields no verifiable confidence score or hallucination check.

## 2. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
To achieve state-of-the-art recall, OmniRAG unifies Dense Vector Search with Sparse BM25 lexical token search via Reciprocal Rank Fusion:
RRF Score(d) = sum_{m in M} (w_m / (k + rank_m(d)))
Where:
- M is the set of retrieval systems (Dense Vector, Sparse BM25).
- k is the rank smoothing constant (empirically optimized to k = 60).
- w_m is the system weighting factor (typically 0.65 for dense, 0.35 for sparse).
RRF is mathematically robust because it normalizes diverse score distributions (cosine similarity [-1, 1] and BM25 unbounded scores [0, inf)) into stable ordinal rankings.

## 3. Two-Stage Retrieval & Cross-Encoder Re-Ranking
While Bi-Encoders compute query and document representations independently for fast vector indexing (O(1) search via HNSW), they cannot capture cross-token attention between the question and the candidate text.
OmniRAG introduces a secondary Cross-Encoder reranker:
1. Candidate Generation: Top-20 candidates are pulled using hybrid RRF.
2. Cross-Attention Scoring: Every (query, passage) pair is jointly processed through cross-encoder layers to compute deep interaction relevance scores.
3. Threshold Filtering: Chunks with cross-encoder relevance scores below tau = 0.45 are discarded as noise.

## 4. Corrective RAG (CRAG) Agentic Loop
The CRAG engine introduces self-evaluating agent nodes:
- Document Grader Node: Analyzes filtered passages against query intent. If cumulative evidence confidence is graded LOW, the system triggers Query Transformation (HyDE / Multi-Query Expansion) and initiates a secondary retrieval pass.
- Faithfulness Guardrail: The synthesized response is cross-checked against source passages to ensure every factual claim contains explicit sentence-level provenance.
"""
    }
}
