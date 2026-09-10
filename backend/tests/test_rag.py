"""
OmniRAG Enterprise - Automated Verification & Test Suite
Validates Chunking, Vector Storage, BM25, RRF Fusion, Cross-Encoder Reranking,
CRAG Self-Correction, and RAG Triad Evaluator.
"""

import unittest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.document_loader import DocumentLoader, DocumentChunk, SAMPLE_ENTERPRISE_DATASETS
from core.vector_store import VectorStore
from core.hybrid_search import BM25Searcher, HybridRetriever
from core.reranker import CrossEncoderReranker
from core.crag_agent import CRAGAgent, RetrievalConfidence
from core.evaluator import RAGEvaluator

class TestOmniRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader(chunk_size=300, chunk_overlap=60)
        self.sample_text = (
            "# Financial Highlights\n"
            "Data Center revenue increased to $47.5 billion, up 217% year-over-year. "
            "Growth was driven by customer demand for Hopper H100 and H200 GPUs.\n\n"
            "# Supply Chain Risk Factors\n"
            "We rely on TSMC for wafer fabrication and advanced packaging using CoWoS. "
            "Shortages of substrate could adversely impact shipment schedules."
        )
        self.chunks = self.loader.chunk_text(
            text=self.sample_text,
            doc_id="test_doc",
            doc_title="Test Financial Report"
        )

    def test_document_chunking(self):
        """Verify semantic chunking and section extraction."""
        self.assertGreater(len(self.chunks), 0)
        for chunk in self.chunks:
            self.assertIn(chunk.section_header, ["Financial Highlights", "Supply Chain Risk Factors"])
            self.assertGreater(len(chunk.content), 20)
            self.assertEqual(chunk.doc_title, "Test Financial Report")

    def test_vector_store_indexing_and_search(self):
        """Verify vector store embedding generation and cosine similarity."""
        store = VectorStore(embedding_dim=128)
        indexed = store.add_chunks(self.chunks)
        self.assertEqual(indexed, len(self.chunks))

        # Query semantic match
        results = store.search_dense("What was the Data Center revenue and growth?", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("Financial Highlights", top_chunk.section_header)
        self.assertGreater(score, 0.4)

    def test_bm25_lexical_search(self):
        """Verify BM25 keyword matching on specific acronyms and numbers."""
        bm25 = BM25Searcher()
        bm25.index_chunks(self.chunks)

        # Exact acronym search (where pure semantic embeddings often underperform)
        results = bm25.search_sparse("CoWoS TSMC packaging", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("TSMC", top_chunk.content)
        self.assertGreater(score, 0.0)

    def test_reciprocal_rank_fusion(self):
        """Verify Reciprocal Rank Fusion combines dense and sparse ranks mathematically."""
        store = VectorStore(embedding_dim=128)
        store.add_chunks(self.chunks)
        bm25 = BM25Searcher()
        bm25.index_chunks(self.chunks)

        dense_res = store.search_dense("TSMC CoWoS substrate risks", top_k=5)
        sparse_res = bm25.search_sparse("TSMC CoWoS substrate risks", top_k=5)

        retriever = HybridRetriever(rrf_k=60)
        fused = retriever.fuse_rrf(dense_res, sparse_res, top_k=3)

        self.assertGreater(len(fused), 0)
        first_item = fused[0]
        self.assertIn("fusion_score", first_item)
        self.assertIn("retrieval_method", first_item)
        self.assertGreater(first_item["fusion_score"], 0.0)

    def test_cross_encoder_reranker(self):
        """Verify Cross-Encoder reranker calculates deep query-passage interactions."""
        reranker = CrossEncoderReranker(relevance_threshold=0.2)
        candidates = [
            {
                "chunk": self.chunks[0],
                "fusion_score": 0.8
            }
        ]
        reranked = reranker.rerank("Data center revenue Hopper H100", candidates, top_k=1)
        self.assertEqual(len(reranked), 1)
        self.assertIn("cross_encoder_score", reranked[0])
        self.assertIn("rerank_score", reranked[0])
        self.assertGreater(reranked[0]["rerank_score"], 0.3)

    def test_crag_grading_and_rewriting(self):
        """Verify Corrective RAG assigns appropriate confidence grades and rewrites queries."""
        crag = CRAGAgent(correct_threshold=0.60, ambiguous_threshold=0.35)

        # High confidence candidates
        mock_high = [{"rerank_score": 0.88, "chunk": self.chunks[0]}]
        grade, score, exp = crag.grade_retrieval("Data Center revenue", mock_high)
        self.assertEqual(grade, RetrievalConfidence.CORRECT)
        self.assertGreater(score, 0.6)

        # Low confidence candidates
        mock_low = [{"rerank_score": 0.20, "chunk": self.chunks[0]}]
        grade, score, exp = crag.grade_retrieval("Unrelated medical question", mock_low)
        self.assertEqual(grade, RetrievalConfidence.INCORRECT)

        # Query rewriting & acronym expansion
        expanded = crag.rewrite_query("What is the Capex and GPU rev?")
        self.assertTrue(any("capital expenditures" in q.lower() for q in expanded))
        self.assertTrue(any("revenue" in q.lower() for q in expanded))

    def test_rag_evaluator(self):
        """Verify RAG Triad benchmark metrics calculation."""
        evaluator = RAGEvaluator()
        
        # Test Context Precision
        retrieved_mock = [
            {"chunk": self.chunks[0], "rerank_score": 0.85},
            {"chunk": self.chunks[1], "rerank_score": 0.20}
        ]
        precision_report = evaluator.evaluate_context_precision("revenue", retrieved_mock, relevance_threshold=0.4)
        self.assertEqual(precision_report["score"], 0.5)

        # Test Faithfulness
        grounded_answer = "Data Center revenue was $47.5 billion driven by Hopper H100 GPUs."
        faith_report = evaluator.evaluate_faithfulness(grounded_answer, self.chunks)
        self.assertGreaterEqual(faith_report["score"], 0.7)

        # Test Answer Relevance
        rel_report = evaluator.evaluate_answer_relevance("What was the Data Center revenue?", grounded_answer)
        self.assertGreater(rel_report["score"], 0.5)

if __name__ == "__main__":
    unittest.main()
