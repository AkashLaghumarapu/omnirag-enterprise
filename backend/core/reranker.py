"""
OmniRAG Enterprise - Cross-Encoder Re-Ranking Pipeline
Performs fine-grained query-passage interaction scoring on hybrid retrieval candidates,
mitigating noise and eliminating distractor chunks before LLM synthesis.
"""

import math
import re
from typing import List, Dict, Any, Optional
from .document_loader import DocumentChunk

class CrossEncoderReranker:
    """
    Two-stage retrieval reranker. Re-scores candidate passages from hybrid search
    using deep lexical-semantic cross-attention simulation and relevance calibration.
    """
    def __init__(self, relevance_threshold: float = 0.35):
        self.relevance_threshold = relevance_threshold

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\$]{2,}\b", text.lower())

    def _calculate_cross_interaction(self, query: str, chunk: DocumentChunk) -> float:
        """
        Simulates cross-encoder attention scoring:
        - Term coverage (how many query words appear in chunk)
        - Exact n-gram phrase match boost
        - Title & section alignment boost
        - Density of informative numerical/financial/technical entities
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        content_tokens = self._tokenize(chunk.content)
        header_tokens = self._tokenize(f"{chunk.doc_title} {chunk.section_header}")
        content_set = set(content_tokens)
        header_set = set(header_tokens)

        # 1. Query Term Coverage (Recall)
        matched_tokens = [t for t in query_tokens if t in content_set]
        coverage_score = len(matched_tokens) / len(query_tokens)

        # 2. Section Header Alignment Boost
        header_matches = [t for t in query_tokens if t in header_set]
        header_score = min(1.0, len(header_matches) / max(1, len(query_tokens)))

        # 3. Exact Phrase Matching Boost (Bigrams / Trigrams)
        phrase_boost = 0.0
        query_lower = query.lower()
        chunk_lower = chunk.content.lower()

        # Check for multi-word query sequences in chunk
        for n in [3, 2]:
            words = query_lower.split()
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if phrase in chunk_lower:
                    phrase_boost += 0.25 * n
                    break

        # 4. Informative entity density (numbers, currencies, technical acronyms)
        entity_matches = len(re.findall(r"(\$?\d+[\d,\.]*\%?|[A-Z]{2,})", chunk.content))
        entity_score = min(0.2, entity_matches * 0.02)

        # Weighted combination
        raw_score = (
            (coverage_score * 0.50) +
            (header_score * 0.20) +
            (min(0.20, phrase_boost)) +
            (entity_score * 0.10)
        )

        return min(1.0, max(0.0, raw_score))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        apply_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Reranks hybrid search candidates using cross-interaction scoring,
        combines with initial RRF score, and filters below relevance_threshold.
        """
        if not candidates:
            return []

        reranked = []
        for item in candidates:
            chunk: DocumentChunk = item["chunk"]
            base_score: float = item.get("fusion_score", 0.5)

            # Cross-encoder interaction score
            cross_score = self._calculate_cross_interaction(query, chunk)

            # Blended final score (60% cross-encoder, 40% initial hybrid RRF)
            final_score = (0.60 * cross_score) + (0.40 * base_score)

            if apply_threshold and final_score < self.relevance_threshold:
                continue

            item_copy = dict(item)
            item_copy["cross_encoder_score"] = round(cross_score, 4)
            item_copy["rerank_score"] = round(final_score, 4)
            reranked.append(item_copy)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
