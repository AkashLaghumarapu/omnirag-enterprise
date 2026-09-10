"""
OmniRAG Enterprise - Main FastAPI Application Server
Orchestrates Document Ingestion, Hybrid Search, Cross-Encoder Reranking,
Agentic Corrective RAG (CRAG), and Streaming SSE Responses with Full Observability.
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any, Optional

# Ensure backend directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from core.document_loader import DocumentLoader, DocumentChunk, SAMPLE_ENTERPRISE_DATASETS
from core.vector_store import VectorStore
from core.hybrid_search import BM25Searcher, HybridRetriever
from core.reranker import CrossEncoderReranker
from core.crag_agent import CRAGAgent, RetrievalConfidence
from core.evaluator import RAGEvaluator
from core.llm_provider import UniversalLLM

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
os.makedirs(DATA_DIR, exist_ok=True)
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.json")

# Initialize Pipeline Singletons
doc_loader = DocumentLoader(chunk_size=600, chunk_overlap=120)
vector_store = VectorStore(persistence_path=VECTOR_STORE_PATH)
bm25_searcher = BM25Searcher()
hybrid_retriever = HybridRetriever(rrf_k=60, dense_weight=0.65, sparse_weight=0.35)
reranker = CrossEncoderReranker(relevance_threshold=0.35)
crag_agent = CRAGAgent(correct_threshold=0.65, ambiguous_threshold=0.38)
evaluator = RAGEvaluator()
llm_provider = UniversalLLM()
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

def load_saved_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                g_key = cfg.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
                gr_key = cfg.get("groq_key") or os.getenv("GROQ_API_KEY", "")
                oa_key = cfg.get("openai_key") or os.getenv("OPENAI_API_KEY", "")
                prov = cfg.get("provider") or "auto"
                llm_provider.set_keys(gemini_key=g_key, groq_key=gr_key, openai_key=oa_key)
                if g_key:
                    vector_store.set_api_key(g_key)
                llm_provider.provider = prov
        except Exception as e:
            logger.error(f"Error loading settings.json: {e}")

load_saved_settings()

# Sync BM25 index on startup if chunks exist
if vector_store.chunks:
    bm25_searcher.index_chunks(vector_store.chunks)
else:
    # Auto-load initial SEC 10-K sample so the user immediately has data
    sample_key = "sec_10k_nvidia"
    sample_info = SAMPLE_ENTERPRISE_DATASETS[sample_key]
    initial_chunks = doc_loader.chunk_text(
        text=sample_info["content"],
        doc_id=sample_key,
        doc_title=sample_info["title"],
        metadata={"category": "SEC Form 10-K", "is_sample": True}
    )
    vector_store.add_chunks(initial_chunks)
    bm25_searcher.index_chunks(vector_store.chunks)

app = FastAPI(
    title="OmniRAG Enterprise",
    description="Production Agentic Corrective RAG (CRAG) & Multi-Vector Intelligence Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Request / Response Models -----------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_crag: bool = True
    enable_reranking: bool = True
    provider: str = "auto"
    api_key: Optional[str] = None
    doc_id: Optional[str] = None

class SettingsUpdate(BaseModel):
    gemini_key: Optional[str] = None
    groq_key: Optional[str] = None
    openai_key: Optional[str] = None
    provider: Optional[str] = None

class SampleLoadRequest(BaseModel):
    sample_key: str

# ----------------- Endpoints -----------------

@app.get("/api/health")
def get_health():
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "vector_store": stats,
        "bm25_indexed_chunks": bm25_searcher.corpus_size,
        "active_models": {
            "gemini_configured": bool(llm_provider.gemini_api_key),
            "groq_configured": bool(llm_provider.groq_api_key),
            "offline_mode": not bool(llm_provider.gemini_api_key or llm_provider.groq_api_key)
        }
    }

@app.get("/api/documents/samples")
def get_samples():
    return {
        "samples": [
            {
                "key": k,
                "title": v["title"],
                "description": v["description"],
                "char_length": len(v["content"])
            }
            for k, v in SAMPLE_ENTERPRISE_DATASETS.items()
        ]
    }

@app.post("/api/documents/load-sample")
def load_sample(req: SampleLoadRequest):
    if req.sample_key not in SAMPLE_ENTERPRISE_DATASETS:
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    
    sample = SAMPLE_ENTERPRISE_DATASETS[req.sample_key]
    chunks = doc_loader.chunk_text(
        text=sample["content"],
        doc_id=req.sample_key,
        doc_title=sample["title"],
        metadata={"is_sample": True, "category": "Enterprise"}
    )
    added = vector_store.add_chunks(chunks)
    bm25_searcher.index_chunks(vector_store.chunks)
    return {
        "success": True,
        "message": f"Successfully loaded and indexed sample: '{sample['title']}'",
        "chunks_indexed": added,
        "total_corpus_chunks": len(vector_store.chunks)
    }

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_title: Optional[str] = Form(None)
):
    title = doc_title or file.filename
    temp_path = os.path.join(DATA_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    if file.filename.lower().endswith(".pdf"):
        chunks = doc_loader.load_pdf(temp_path, doc_title=title)
    else:
        chunks = doc_loader.load_text_file(temp_path, doc_title=title)

    added = vector_store.add_chunks(chunks)
    bm25_searcher.index_chunks(vector_store.chunks)

    # Clean up uploaded temp file
    try:
        os.remove(temp_path)
    except OSError:
        pass

    return {
        "success": True,
        "doc_id": chunks[0].doc_id if chunks else None,
        "filename": file.filename,
        "title": title,
        "chunks_created": len(chunks),
        "new_chunks_added": added,
        "total_corpus_chunks": len(vector_store.chunks)
    }

@app.get("/api/documents/list")
def list_documents():
    docs: Dict[str, Dict[str, Any]] = {}
    for c in vector_store.chunks:
        if c.doc_id not in docs:
            docs[c.doc_id] = {
                "doc_id": c.doc_id,
                "title": c.doc_title,
                "chunk_count": 0,
                "sections": set(),
                "metadata": c.metadata,
                "is_sample": bool(c.metadata.get("is_sample", False))
            }
        docs[c.doc_id]["chunk_count"] += 1
        if c.section_header:
            docs[c.doc_id]["sections"].add(c.section_header)

    result = []
    for d in docs.values():
        d["sections"] = list(d["sections"])[:5]
        result.append(d)

    return {"documents": result, "total_chunks": len(vector_store.chunks)}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = vector_store.remove_document(doc_id)
    bm25_searcher.index_chunks(vector_store.chunks)
    return {
        "success": True,
        "message": f"Document {doc_id} deleted",
        "chunks_removed": removed,
        "total_chunks": len(vector_store.chunks)
    }

@app.post("/api/documents/clear-samples")
def clear_samples():
    removed = vector_store.remove_samples()
    bm25_searcher.index_chunks(vector_store.chunks)
    return {
        "success": True,
        "message": "Sample documents removed",
        "chunks_removed": removed,
        "total_chunks": len(vector_store.chunks)
    }

@app.post("/api/documents/clear")
def clear_documents():
    vector_store.clear()
    bm25_searcher.index_chunks([])
    return {"success": True, "message": "Index cleared"}

@app.post("/api/settings")
def update_settings(settings: SettingsUpdate):
    current = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            current = {}

    if settings.gemini_key is not None:
        vector_store.set_api_key(settings.gemini_key)
        llm_provider.set_keys(gemini_key=settings.gemini_key)
        current["gemini_key"] = settings.gemini_key
    if settings.groq_key is not None:
        llm_provider.set_keys(groq_key=settings.groq_key)
        current["groq_key"] = settings.groq_key
    if settings.openai_key is not None:
        llm_provider.set_keys(openai_key=settings.openai_key)
        current["openai_key"] = settings.openai_key
    if settings.provider is not None:
        llm_provider.provider = settings.provider
        current["provider"] = settings.provider

    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write settings.json: {e}")

    return {"success": True, "message": "Settings updated and persisted successfully"}

# ----------------- Streaming RAG Pipeline Endpoint -----------------

@app.post("/api/query/stream")
async def execute_query_stream(req: QueryRequest):
    """
    Executes full production RAG pipeline with Server-Sent Events (SSE):
    1. Query Transformation & HyDE Expansion
    2. Dense Vector Retrieval (HNSW Cosine Sim) + Sparse BM25 Retrieval
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder Re-Ranking
    5. Corrective RAG (CRAG) Document Relevance Grading
    6. LLM Token Streaming
    7. Automated RAG Triad Evaluation Benchmark Telemetry
    """
    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed in vector store. Please load a sample or upload a document.")

    # Override API key dynamically if passed from UI
    if req.api_key:
        llm_provider.set_keys(gemini_key=req.api_key)
        vector_store.set_api_key(req.api_key)
    if req.provider:
        llm_provider.provider = req.provider

    async def event_generator():
        latency_map = {}
        t_total_start = time.time()

        # Step 1: Query Transformation / HyDE Expansion
        t0 = time.time()
        expanded_queries = crag_agent.rewrite_query(req.query) if req.enable_crag else [req.query]
        latency_map["query_transformation_ms"] = (time.time() - t0) * 1000

        # Step 2: Hybrid Retrieval (Dense Vector + Sparse BM25)
        t0 = time.time()
        # Collect candidates across expanded queries
        all_dense = []
        all_sparse = []
        seen_dense = set()
        seen_sparse = set()

        for q in expanded_queries:
            dense_res = vector_store.search_dense(q, top_k=req.top_k * 3, filter_doc_id=req.doc_id)
            for c, s in dense_res:
                if c.id not in seen_dense:
                    seen_dense.add(c.id)
                    all_dense.append((c, s))

            sparse_res = bm25_searcher.search_sparse(q, top_k=req.top_k * 3, filter_doc_id=req.doc_id)
            for c, s in sparse_res:
                if c.id not in seen_sparse:
                    seen_sparse.add(c.id)
                    all_sparse.append((c, s))

        latency_map["hybrid_retrieval_ms"] = (time.time() - t0) * 1000

        # Step 3: Reciprocal Rank Fusion (RRF)
        t0 = time.time()
        fused_candidates = hybrid_retriever.fuse_rrf(all_dense, all_sparse, top_k=req.top_k * 3)
        latency_map["rrf_fusion_ms"] = (time.time() - t0) * 1000

        # Step 4: Cross-Encoder Re-Ranking
        t0 = time.time()
        if req.enable_reranking:
            reranked_chunks = reranker.rerank(
                req.query,
                fused_candidates,
                top_k=req.top_k,
                apply_threshold=True
            )
        else:
            reranked_chunks = fused_candidates[:req.top_k]
        latency_map["cross_encoder_rerank_ms"] = (time.time() - t0) * 1000

        # Step 5: CRAG Relevance Grading
        t0 = time.time()
        if req.enable_crag:
            confidence_grade, confidence_score, explanation = crag_agent.grade_retrieval(req.query, reranked_chunks)
        else:
            confidence_grade, confidence_score, explanation = (
                RetrievalConfidence.CORRECT,
                0.90,
                "CRAG grading bypassed."
            )
        latency_map["crag_evaluation_ms"] = (time.time() - t0) * 1000

        # Send Initial Metadata Event to Frontend
        meta_payload = {
            "query": req.query,
            "expanded_queries": expanded_queries,
            "crag": {
                "grade": confidence_grade.value,
                "confidence_score": confidence_score,
                "explanation": explanation
            },
            "sources": [
                {
                    "citation_index": i + 1,
                    "chunk_id": item["chunk"].id,
                    "doc_title": item["chunk"].doc_title,
                    "section_header": item["chunk"].section_header,
                    "page_number": item["chunk"].page_number,
                    "retrieval_method": item.get("retrieval_method", "Hybrid"),
                    "rerank_score": item.get("rerank_score", 0.0),
                    "cross_encoder_score": item.get("cross_encoder_score", 0.0),
                    "content_preview": item["chunk"].content[:260] + "...",
                    "full_content": item["chunk"].content
                }
                for i, item in enumerate(reranked_chunks)
            ]
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

        # Step 6: Stream Synthesized LLM Response
        t0 = time.time()
        full_answer = ""
        async for token in llm_provider.stream_response(req.query, reranked_chunks):
            full_answer += token
            token_payload = {"token": token}
            yield f"event: token\ndata: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.005)

        latency_map["llm_generation_ms"] = (time.time() - t0) * 1000

        # Step 7: Compute Automated RAG Triad Evaluation Benchmark
        benchmark = evaluator.compute_overall_benchmark(
            query=req.query,
            answer=full_answer,
            retrieved_chunks=reranked_chunks,
            context_chunks=[item["chunk"] for item in reranked_chunks],
            latency_map=latency_map
        )

        yield f"event: benchmark\ndata: {json.dumps(benchmark)}\n\n"
        yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ----------------- Mount Frontend Static Files -----------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
