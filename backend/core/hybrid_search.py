"""
OmniRAG Enterprise - Hybrid Search & Reciprocal Rank Fusion (RRF)
Combines Sparse BM25 Lexical Retrieval with Dense Vector Retrieval using
mathematically sound reciprocal rank fusion to maximize recall and precision.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from .document_loader import DocumentChunk

class BM25Searcher:
    """
    Okapi BM25 implementation for lexical keyword retrieval.
    Tuned with k1=1.5 (term frequency saturation) and b=0.75 (document length normalization).
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = {} # term -> [(chunk_id, freq)]
        self.chunks_map: Dict[str, DocumentChunk] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases, strips punctuation, and extracts alphanumeric tokens."""
        return re.findall(r"\b[a-zA-Z0-9_\-\$]{2,}\b", text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Builds inverted index and term statistics from corpus."""
        self.chunks_map = {c.id: c for c in chunks}
        self.corpus_size = len(chunks)
        self.doc_lengths = {}
        self.doc_freqs = {}
        self.inverted_index = {}

        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        total_tokens = 0
        for chunk in chunks:
            full_text = f"{chunk.doc_title} {chunk.section_header} {chunk.content}"
            tokens = self._tokenize(full_text)
            self.doc_lengths[chunk.id] = len(tokens)
            total_tokens += len(tokens)

            tf = Counter(tokens)
            for term, freq in tf.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((chunk.id, freq))

        self.avgdl = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

    def _calc_idf(self, term: str) -> float:
        """Computes Robertson-Sparck Jones IDF with smoothing."""
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search_sparse(
        self,
        query: str,
        top_k: int = 15,
        filter_doc_id: Optional[str] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Scores documents against query using BM25 formula."""
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = self._calc_idf(term)
            for chunk_id, freq in self.inverted_index[term]:
                chunk = self.chunks_map.get(chunk_id)
                if filter_doc_id and chunk and chunk.doc_id != filter_doc_id:
                    continue

                doc_len = self.doc_lengths.get(chunk_id, self.avgdl)
                # BM25 TF component
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                term_score = idf * (num / denom)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        if not scores:
            return []

        # Normalize BM25 scores to [0, 1] relative to max score
        max_score = max(scores.values()) if scores else 1.0
        results = []
        for chunk_id, raw_score in scores.items():
            chunk = self.chunks_map.get(chunk_id)
            if chunk:
                if filter_doc_id and chunk.doc_id != filter_doc_id:
                    continue
                norm_score = float(raw_score / max_score) if max_score > 0 else 0.0
                results.append((chunk, norm_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class HybridRetriever:
    """
    Fuses Dense Vector Search results with Sparse BM25 results using
    Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.65, sparse_weight: float = 0.35):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse_rrf(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion:
        RRF(d) = sum_{system} w_sys / (k + rank_sys(d))
        Returns list of structured records with chunk metadata and provenance attribution.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}
        dense_raw_scores: Dict[str, float] = {}
        sparse_raw_scores: Dict[str, float] = {}

        # Process Dense results
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_map[chunk.id] = chunk
            dense_ranks[chunk.id] = rank
            dense_raw_scores[chunk.id] = score
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_score

        # Process Sparse BM25 results
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.id] = chunk
            sparse_ranks[chunk.id] = rank
            sparse_raw_scores[chunk.id] = score
            rrf_score = self.sparse_weight / (self.rrf_k + rank)
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + rrf_score

        if not rrf_scores:
            return []

        # Max theoretical RRF score
        max_possible_rrf = (self.dense_weight / (self.rrf_k + 1)) + (self.sparse_weight / (self.rrf_k + 1))

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused_output = []
        for chunk_id, fused_score in sorted_chunks:
            chunk = chunk_map[chunk_id]
            is_dense = chunk_id in dense_ranks
            is_sparse = chunk_id in sparse_ranks
            
            origin = "Hybrid (Dense + BM25)" if (is_dense and is_sparse) else ("Dense Semantic" if is_dense else "BM25 Sparse")
            
            # Normalized fusion score
            normalized_score = min(1.0, fused_score / max_possible_rrf)

            fused_output.append({
                "chunk": chunk,
                "fusion_score": round(normalized_score, 4),
                "retrieval_method": origin,
                "dense_rank": dense_ranks.get(chunk_id),
                "dense_score": round(dense_raw_scores.get(chunk_id, 0.0), 4),
                "sparse_rank": sparse_ranks.get(chunk_id),
                "sparse_score": round(sparse_raw_scores.get(chunk_id, 0.0), 4)
            })

        return fused_output
