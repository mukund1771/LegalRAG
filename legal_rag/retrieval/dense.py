"""Dense (semantic) retrieval over child chunk vectors.

Vectors are L2-normalized at ingestion, so a dot product is cosine similarity. For a
small corpus an exact brute-force matmul is faster and simpler than an ANN index; the
``VectorStore`` seam lets us swap in HNSW/Qdrant for the 10k+ path without changing
callers.

Dense search handles *paraphrase* queries — e.g. "can they share data with
subcontractors" matching "disclosure to third-party processors" — that exact-term
search misses.
"""

from __future__ import annotations

import numpy as np

from legal_rag.retrieval.store import VectorStore


def search(store: VectorStore, query_vector: np.ndarray, top_k: int,
           filters: dict | None = None) -> list[tuple[str, float]]:
    """Return up to ``top_k`` (child_id, cosine_score) for the query, post-filtered."""
    if store.child_vectors.shape[0] == 0:
        return []
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(q)
    if norm:
        q = q / norm

    scores = store.child_vectors @ q  # (n_children,)
    keep = store.child_index_filter(filters)
    if not keep:
        return []

    keep_scores = [(i, float(scores[i])) for i in keep]
    keep_scores.sort(key=lambda t: t[1], reverse=True)
    return [(store.child_ids[i], s) for i, s in keep_scores[:top_k]]
