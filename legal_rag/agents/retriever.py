"""Retriever agent — the hybrid retrieval pipeline.

Pipeline per (sub-)query:

    embed query
        │
        ├── dense.search (semantic)  ─┐
        ├── bm25.search  (lexical)   ─┤→ RRF fusion → rerank → top-k children
        │                             ┘
        └── parent expansion → Evidence (citation + parent context)

Design choices realized here:
- **Hybrid + RRF**: combine exact-term (BM25) and paraphrase (dense) recall, fused by
  rank so no score normalization is needed.
- **Metadata pre-filtering**: the planner can constrain by doc_type / clause_type /
  party, which is how cross-document queries (e.g. governing law per agreement) are
  served precisely.
- **Parent-child expansion**: we match the small precise child but return the parent
  section so the synthesizer reasons with full context.
- **Multi sub-query**: decomposed queries (cross-doc / multi-part) are retrieved
  independently and merged, de-duplicated by parent section.

The Retriever does no generation — it only finds and packages evidence, keeping the
retrieval failure mode (recall / document mismatch) isolated and testable.
"""

from __future__ import annotations

from legal_rag.models import Chunk, Evidence
from legal_rag.retrieval import bm25, dense
from legal_rag.retrieval.fusion import reciprocal_rank_fusion
from legal_rag.retrieval.rerank import Reranker
from legal_rag.retrieval.store import VectorStore


def _citation(meta) -> str:
    return f"[{meta.doc_id} §{meta.section_no} {meta.section_heading}]"


class Retriever:
    def __init__(self, store: VectorStore, embedder, reranker: Reranker, settings) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = settings.retrieval

    # ------------------------------------------------------------------ public

    def retrieve(self, queries: str | list[str], filters: dict | None = None,
                 final_k: int | None = None) -> list[Evidence]:
        """Retrieve evidence for one query or a list of sub-queries (merged)."""
        if isinstance(queries, str):
            queries = [queries]
        final_k = final_k or self.cfg.final_k

        # Gather reranked children across all sub-queries, keep the best score per id.
        best: dict[str, float] = {}
        for q in queries:
            for cid, score in self._retrieve_one(q, filters):
                if cid not in best or score > best[cid]:
                    best[cid] = score

        ranked_children = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return self._expand_to_evidence(ranked_children, final_k)

    # ------------------------------------------------------------------ stages

    def _retrieve_one(self, query: str, filters: dict | None) -> list[tuple[str, float]]:
        """Hybrid retrieve + fuse + rerank for a single query."""
        q_vec = self.embedder.embed([query])[0]
        dense_hits = dense.search(self.store, q_vec, self.cfg.dense_top_k, filters)
        bm25_hits = bm25.search(self.store, query, self.cfg.bm25_top_k, filters)

        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits], k=self.cfg.rrf_k, top_n=self.cfg.rerank_top_n
        )
        if not fused:
            return []

        # Rerank on the clause prefixed with its section heading (passage-with-title):
        # the heading is a strong, low-noise relevance signal for both the cross-encoder
        # and the lexical fallback (e.g. "Governing Law", "Uptime Commitment").
        candidates = []
        for cid, _ in fused:
            ch = self.store.get(cid)
            candidates.append((cid, f"{ch.metadata.section_heading}. {ch.text}"))
        return self.reranker.rerank(query, candidates, self.cfg.final_k)

    def _expand_to_evidence(self, ranked_children: list[tuple[str, float]],
                            final_k: int) -> list[Evidence]:
        """Parent-child expansion + de-dup by parent section, capped at final_k."""
        evidence: list[Evidence] = []
        seen_parents: set[str] = set()

        for cid, score in ranked_children:
            child = self.store.get(cid)
            parent = self.store.parent_of(cid) or child
            if parent.chunk_id in seen_parents:
                continue
            seen_parents.add(parent.chunk_id)

            m = child.metadata
            evidence.append(
                Evidence(
                    chunk_id=cid,
                    doc_id=m.doc_id,
                    doc_type=m.doc_type,
                    section_no=m.section_no,
                    section_heading=m.section_heading,
                    clause_type=m.clause_type,
                    child_text=child.text,
                    context_text=parent.text,
                    citation=_citation(m),
                    char_start=m.char_start,
                    char_end=m.char_end,
                    score=float(score),
                )
            )
            if len(evidence) >= final_k:
                break
        return evidence
