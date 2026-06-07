"""Re-ranking of the fused finalists (the optional-bonus precision stage).

A cross-encoder reads the (query, clause) pair *together*, so it judges relevance far
more accurately than separate bi-encoder embeddings — the difference between "mentions
liability" and "actually answers whether liability is capped for this breach".

Backends behind one interface:
- ``CrossEncoderReranker`` — bge-reranker-v2-m3 via sentence-transformers (runtime).
- ``LexicalReranker``     — deterministic token-overlap scorer for offline tests, so
  the full retrieval pipeline can be unit-tested without downloading a model.

Swapping in an LLM-selection step (Ranking-Free RAG) later is a new class here.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small stopword list so the lexical fallback scores on content words, not glue words.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to",
    "in", "on", "for", "and", "or", "as", "at", "by", "with", "that", "this", "it",
    "its", "what", "which", "who", "whom", "how", "do", "does", "did", "if", "any",
    "from", "into", "than", "then", "there", "their", "i", "you", "we", "they",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[tuple[str, str]],
               top_k: int) -> list[tuple[str, float]]:
        """Score (id, text) candidates against the query; return top_k (id, score)."""
        ...


class LexicalReranker:
    """Deterministic content-word reranker (offline / tests).

    Scores by query-term coverage — fraction of the query's content words present in
    the passage — which is recall-oriented and not penalized by passage length (unlike
    raw Jaccard). A small overlap-coefficient term breaks ties toward focused passages.
    This is a stand-in for the cross-encoder, good enough to make the pipeline testable.
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[tuple[str, str]],
               top_k: int) -> list[tuple[str, float]]:
        q = _content_tokens(query)
        if not q:
            return [(cid, 0.0) for cid, _ in candidates][:top_k]
        scored = []
        for cid, text in candidates:
            t = _content_tokens(text)
            inter = len(q & t)
            coverage = inter / len(q)
            overlap = inter / (min(len(q), len(t)) or 1)
            scored.append((cid, coverage + 0.001 * overlap))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """bge-reranker-v2-m3 cross-encoder via sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder  # lazy import
        self.name = f"cross-encoder:{model}"
        self._model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[tuple[str, str]],
               top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [[query, text] for _cid, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            ((cid, float(s)) for (cid, _t), s in zip(candidates, scores)),
            key=lambda kv: kv[1], reverse=True,
        )
        return ranked[:top_k]


def get_reranker(settings) -> Reranker:
    """Factory: cross-encoder by default; lexical when reranker_backend='lexical'."""
    backend = getattr(settings.retrieval, "reranker_backend", "cross_encoder")
    if backend == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(model=settings.retrieval.reranker_model)
