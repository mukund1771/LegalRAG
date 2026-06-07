"""Pure-function ranking metrics for retrieval evaluation.

All take ``ranked_rel`` — a list of booleans, one per retrieved item in rank order,
True if that item is a gold-relevant passage — plus ``n_relevant`` (total gold passages
for the query). Keeping these as pure functions makes them trivially unit-testable.
"""

from __future__ import annotations

import math


def hit_at_k(ranked_rel: list[bool], k: int) -> float:
    return 1.0 if any(ranked_rel[:k]) else 0.0


def recall_at_k(ranked_rel: list[bool], k: int, n_relevant: int) -> float:
    if not n_relevant:
        return 0.0
    return sum(1 for r in ranked_rel[:k] if r) / n_relevant


def precision_at_k(ranked_rel: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for r in ranked_rel[:k] if r) / k


def mrr(ranked_rel: list[bool]) -> float:
    for i, r in enumerate(ranked_rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rel: list[bool], k: int, n_relevant: int) -> float:
    gains = [1.0 if r else 0.0 for r in ranked_rel[:k]]
    dcg = _dcg(gains)
    ideal = [1.0] * min(n_relevant, k)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg else 0.0
