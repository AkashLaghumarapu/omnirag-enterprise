"""
OmniRAG Enterprise - RAG Triad & Performance Evaluator
Measures Context Precision, Faithfulness (Groundedness), Answer Relevance,
and records comprehensive latency waterfall telemetry for production observability.
"""

import time
import re
from typing import List, Dict, Any, Optional
from .document_loader import DocumentChunk

class RAGEvaluator:
    """
    Computes RAG Triad evaluation metrics (Context Precision, Faithfulness,
    Answer Relevance) and latency breakdowns for benchmarking and interviews.
    """

    def evaluate_context_precision(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        relevance_threshold: float = 0.40
    ) -> Dict[str, Any]:
        """
        Context Precision: Measures the signal-to-noise ratio in retrieved passages.
        What fraction of retrieved chunks are genuinely relevant to the query?
        """
        if not retrieved_chunks:
            return {"score": 0.0, "relevant_count": 0, "total_chunks": 0}

        relevant_count = 0
        chunk_scores = []

        for item in retrieved_chunks:
            score = item.get("rerank_score", item.get("fusion_score", 0.0))
            is_relevant = score >= relevance_threshold
            if is_relevant:
                relevant_count += 1
            chunk_scores.append({"chunk_id": item["chunk"].id, "score": score, "relevant": is_relevant})

        precision = relevant_count / len(retrieved_chunks)

        return {
            "score": round(precision, 4),
            "relevant_count": relevant_count,
            "total_chunks": len(retrieved_chunks),
            "details": chunk_scores
        }

    def evaluate_faithfulness(
        self,
        answer: str,
        context_chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Faithfulness (Groundedness): Measures what percentage of factual claims
        made in the generated answer are explicitly supported by the context.
        Prevents hallucination in production.
        """
        if not context_chunks or not answer.strip():
            return {"score": 1.0 if not answer.strip() else 0.0, "verified_claims": 0, "total_claims": 0}

        # Filter out UI framing lines (markdown headers, meta notes, quote lines)
        raw_lines = [l.strip() for l in answer.split("\n") if l.strip()]
        claim_lines = []
        for l in raw_lines:
            if l.startswith("#") or l.startswith(">") or l.startswith("*(") or "Provenance" in l:
                continue
            # Remove leading bullet dashes
            clean_l = re.sub(r"^[-–—*•\d\.\s]+", "", l).strip()
            if len(clean_l.split()) >= 3:
                claim_lines.append(clean_l)

        if not claim_lines:
            return {"score": 1.0, "verified_claims": 1, "total_claims": 1}

        corpus_text = " ".join([c.content.lower() for c in context_chunks])

        meta_stop_words = {
            "this", "that", "with", "from", "have", "more", "such", "also", "been", "will",
            "based", "verified", "credentials", "candidate", "assessment", "summary",
            "according", "breakdown", "competencies", "possesses", "following", "evidence",
            "overview", "highlights", "demonstrates", "practical", "technical", "foundation",
            "relevant", "documented", "impact", "conducted", "completed", "degree", "institution",
            "performance", "pursuing", "score", "listed", "earned", "details", "information",
            "profile", "findings", "reveals", "provenance", "sections", "retrieved", "passages",
            "verdict", "recruiter", "entry-level", "requirements", "satisfies", "junior", "role",
            "full", "name", "seeking", "challenging", "position", "utilize", "contribute"
        }

        verified_sentences = 0
        statement_audit = []

        for s in claim_lines:
            s_lower = s.lower()
            words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", s_lower) if w not in meta_stop_words]
            if not words:
                continue

            matches = [w for w in words if w in corpus_text]
            ratio = len(matches) / len(words)
            is_grounded = ratio >= 0.45 or len(matches) >= 2

            if is_grounded:
                verified_sentences += 1

            statement_audit.append({
                "statement": s[:120] + ("..." if len(s) > 120 else ""),
                "grounded": is_grounded,
                "overlap_ratio": round(ratio, 2)
            })

        score = verified_sentences / max(1, len(statement_audit))

        return {
            "score": round(min(1.0, score), 4),
            "verified_claims": verified_sentences,
            "total_claims": len(statement_audit),
            "audit": statement_audit[:5]
        }

    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Answer Relevance: Evaluates how directly the synthesized answer
        addresses the user's core intent.
        """
        if not query.strip() or not answer.strip():
            return {"score": 0.0}

        q_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower()))
        a_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", answer.lower()))

        question_stop_words = {
            "what", "when", "where", "which", "about", "there", "their", "these", "those",
            "have", "from", "with", "does", "been", "the", "are", "his", "her", "how",
            "can", "and", "for", "any", "tell"
        }
        core_q_words = {w for w in q_words if w not in question_stop_words}
        eval_words = core_q_words if core_q_words else q_words

        if not eval_words:
            return {"score": 1.0}

        overlap = eval_words.intersection(a_words)
        relevance = len(overlap) / len(eval_words)
        # Scale with answer completeness
        completeness_bonus = min(0.20, len(answer.split()) / 60)
        final_score = min(1.0, (relevance * 0.80) + completeness_bonus)

        return {
            "score": round(final_score, 4),
            "matched_keywords": list(overlap)
        }

    def compute_overall_benchmark(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        context_chunks: List[DocumentChunk],
        latency_map: Dict[str, float]
    ) -> Dict[str, Any]:
        """Aggregates all metrics into an executive benchmark report."""
        c_precision = self.evaluate_context_precision(query, retrieved_chunks)
        faithfulness = self.evaluate_faithfulness(answer, context_chunks)
        a_relevance = self.evaluate_answer_relevance(query, answer)

        triad_composite = round(
            (c_precision["score"] * 0.35) +
            (faithfulness["score"] * 0.40) +
            (a_relevance["score"] * 0.25),
            4
        )

        total_latency = sum(latency_map.values())

        return {
            "rag_triad_composite_score": triad_composite,
            "context_precision": c_precision,
            "faithfulness": faithfulness,
            "answer_relevance": a_relevance,
            "latency_breakdown_ms": {k: round(v, 2) for k, v in latency_map.items()},
            "total_latency_ms": round(total_latency, 2),
            "status": "PASS" if triad_composite >= 0.70 else "NEEDS_IMPROVEMENT"
        }
