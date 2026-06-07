"""Sparse BM25 retrieval over child chunks.

BM25 nails the *exact tokens* that matter enormously in contracts and that dense
search under-weights: "72 hours", "99.9%", section numbers, and party names like
"Vendor XYZ". It is the complement to dense retrieval in the hybrid pipeline.
"""

from __future__ import annotations

import re

from legal_rag.retrieval.store import VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def search(store: VectorStore, query: str, top_k: int,
           filters: dict | None = None) -> list[tuple[str, float]]:
    """Return up to ``top_k`` (child_id, bm25_score), post-filtered by metadata."""
    bm25 = store.ensure_bm25()
    if bm25 is None or not store.child_ids:
        return []

    scores = bm25.get_scores(_tokenize(query))  # aligned to child_ids order
    keep = set(store.child_index_filter(filters))
    ranked = [
        (store.child_ids[i], float(scores[i]))
        for i in range(len(store.child_ids))
        if i in keep
    ]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked[:top_k]
