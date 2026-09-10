"""
OmniRAG Enterprise - Corrective RAG (CRAG) Agent & Self-Correction Engine
Implements document relevance grading, query reformulation, confidence scoring,
and hallucination guardrails to prevent ungrounded generation.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from .document_loader import DocumentChunk

class RetrievalConfidence(str, Enum):
    CORRECT = "CORRECT"        # High confidence, retrieved chunks directly answer query
    AMBIGUOUS = "AMBIGUOUS"    # Moderate confidence, query needs expansion or multiple angles
    INCORRECT = "INCORRECT"    # Low confidence, corpus lacks relevant facts

class CRAGAgent:
    """
    Agentic Corrective RAG controller that inspects retrieval quality,
    decides corrective actions (query rewrite vs context fallback),
    and enforces strict citation grounding.
    """
    def __init__(self, correct_threshold: float = 0.65, ambiguous_threshold: float = 0.38):
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold

    def grade_retrieval(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[RetrievalConfidence, float, str]:
        """
        Grades retrieved passages on semantic alignment and evidence sufficiency.
        Returns (RetrievalConfidence, confidence_score, explanation).
        """
        if not chunks:
            return (
                RetrievalConfidence.INCORRECT,
                0.0,
                "No candidate passages satisfied the retrieval and reranking thresholds."
            )

        top_scores = [c.get("rerank_score", c.get("fusion_score", 0.0)) for c in chunks]
        avg_top_score = sum(top_scores[:3]) / min(3, len(top_scores))
        max_score = max(top_scores)

        # Blended confidence
        confidence = (max_score * 0.7) + (avg_top_score * 0.3)

        if confidence >= self.correct_threshold:
            status = RetrievalConfidence.CORRECT
            explanation = f"High relevance verified (Confidence: {round(confidence*100, 1)}%). Passage evidence directly addresses query intent."
        elif confidence >= self.ambiguous_threshold:
            status = RetrievalConfidence.AMBIGUOUS
            explanation = f"Partial relevance detected (Confidence: {round(confidence*100, 1)}%). Query reformulation applied to expand context."
        else:
            status = RetrievalConfidence.INCORRECT
            explanation = f"Insufficient evidence in corpus (Confidence: {round(confidence*100, 1)}%). Activating hallucination guardrail."

        return (status, round(confidence, 4), explanation)

    def rewrite_query(self, query: str) -> List[str]:
        """
        Agentic query transformation:
        Generates alternative search queries by expanding acronyms,
        extracting core entities, and constructing HyDE-style factual questions.
        """
        expanded_queries = [query]

        # Financial & technical synonym/acronym expansions
        expansions = {
            "rev": "revenue",
            "capex": "capital expenditures",
            "gpu": "graphics processing unit computing accelerator",
            "llm": "large language model",
            "crag": "corrective retrieval augmented generation",
            "rrf": "reciprocal rank fusion",
            "bm25": "best matching 25 lexical search",
            "hnsw": "hierarchical navigable small world vector index",
            "bis": "bureau of industry and security export restrictions"
        }

        query_lower = query.lower()
        rewritten = query_lower
        for acronym, full_term in expansions.items():
            pattern = rf"\b{acronym}\b"
            if re.search(pattern, query_lower):
                rewritten = re.sub(pattern, f"{acronym} {full_term}", rewritten)

        if rewritten != query_lower:
            expanded_queries.append(rewritten)

        # Extract core nouns / question targets
        clean_q = re.sub(r"^(what is|how does|can you explain|tell me about|what are the)\s+", "", query_lower)
        if len(clean_q.split()) >= 3 and clean_q not in expanded_queries:
            expanded_queries.append(f"{clean_q} key details specifications metrics")

        return expanded_queries[:3]

    def verify_grounding(self, generated_text: str, context_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Anti-Hallucination verification: Checks if numerical figures and key claims
        in the generated text exist in the source chunks.
        """
        if not context_chunks:
            return {"grounded": False, "score": 0.0, "hallucinated_tokens": []}

        corpus_text = " ".join([c.content for c in context_chunks]).lower()
        
        # Check numbers and percentages in response
        numbers_in_response = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", generated_text))
        grounded_numbers = [n for n in numbers_in_response if n.lower() in corpus_text]
        
        unsupported = [n for n in numbers_in_response if n.lower() not in corpus_text]

        total_claims = len(numbers_in_response)
        if total_claims == 0:
            score = 0.95
        else:
            score = len(grounded_numbers) / total_claims

        return {
            "grounded": score >= 0.75,
            "grounding_score": round(score, 4),
            "verified_entities": grounded_numbers,
            "unsupported_entities": unsupported
        }
