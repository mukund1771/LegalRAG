"""Reciprocal Rank Fusion (RRF).

BM25 scores and cosine similarities live on incompatible scales, so naive weighted
averaging is fragile. RRF fuses ranked lists *by position*, not score: each item gets
``sum(1 / (k + rank))`` across the lists it appears in. It needs no score
normalization and reliably beats either retriever alone.

Reference constant k defaults to 60 (the value from the original RRF paper).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked (id, score) lists into one ranked (id, rrf_score) list.

    Only ranks are used; the per-list scores are ignored by design.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)

    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_n] if top_n else out
