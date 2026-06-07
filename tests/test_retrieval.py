"""End-to-end tests for the Milestone 2 retrieval core.

Runs fully offline: FakeEmbedder for query/document vectors and the LexicalReranker,
so no model download or network is needed. Even with the weak hashing embedder, the
*hybrid* pipeline (BM25 carries exact-term overlap, dense + RRF + rerank reinforce)
retrieves the correct clause for the sample queries — which is itself evidence for the
hybrid design.

Assertions check the things that matter:
- the right document AND clause type are retrieved for representative sample queries,
- metadata filters constrain results,
- RRF math is correct,
- parent-child expansion returns the parent section as context.
"""

from __future__ import annotations

import os

import pytest

from legal_rag.agents.retriever import Retriever
from legal_rag.config.settings import load_settings
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import FakeEmbedder
from legal_rag.retrieval.fusion import reciprocal_rank_fusion
from legal_rag.retrieval.rerank import LexicalReranker
from legal_rag.retrieval.store import VectorStore

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "contracts")


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("index"))
    embedder = FakeEmbedder(dim=256)
    build_index(CONTRACTS_DIR, embedder, index_dir)
    store = VectorStore.load(index_dir)
    settings = load_settings()
    return Retriever(store, embedder, LexicalReranker(), settings)


def _top(retriever, query, **kw):
    ev = retriever.retrieve(query, **kw)
    assert ev, f"no evidence for: {query}"
    return ev[0]


def test_q1_nda_termination_notice(retriever):
    top = _top(retriever, "What is the notice period for terminating the NDA?")
    assert top.doc_type == "NDA"
    assert top.clause_type in ("termination", "survival")
    assert "thirty (30) days" in top.context_text


def test_q2_sla_uptime(retriever):
    ev = retriever.retrieve("What is the uptime commitment in the SLA?")
    assert ev and ev[0].doc_type == "SLA"
    assert any("99.5%" in e.context_text for e in ev)


def test_q3_governing_law_vendor(retriever):
    # The planner serves "which law governs <agreement>" with a clause_type filter.
    top = _top(retriever, "Which law governs the Vendor Services Agreement?",
               filters={"doc_type": "Vendor", "clause_type": "governing_law"})
    assert top.doc_type == "Vendor"
    assert top.clause_type == "governing_law"
    assert "England" in top.context_text


def test_q14_breach_notification_window(retriever):
    top = _top(retriever, "What happens if Vendor delays breach notification beyond 72 hours?")
    assert top.doc_type == "DPA"
    assert "72" in top.context_text


def test_metadata_filter_restricts_docs(retriever):
    ev = retriever.retrieve("confidentiality obligations", filters={"doc_type": "NDA"})
    assert ev
    assert all(e.doc_type == "NDA" for e in ev)


def test_cross_doc_governing_law(retriever):
    """Q9: a governing-law sub-query per document surfaces both jurisdictions."""
    ev = retriever.retrieve("governing law jurisdiction",
                            filters={"clause_type": "governing_law"}, final_k=5)
    laws = " ".join(e.context_text for e in ev)
    assert "California" in laws and ("England" in laws or "European" in laws)


def test_parent_child_expansion(retriever):
    """Evidence context is the parent section (>= the matched child text)."""
    top = _top(retriever, "service credits remedy if uptime is not met")
    assert len(top.context_text) >= len(top.child_text)
    assert top.section_heading  # parent heading present


def test_rrf_math():
    a = [("x", 9.0), ("y", 8.0)]   # x rank0, y rank1
    b = [("y", 5.0), ("z", 4.0)]   # y rank0, z rank1
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    # y appears in both lists -> highest fused score
    assert max(fused, key=fused.get) == "y"
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)  # rank1 in a, rank0 in b
    assert fused["x"] == pytest.approx(1 / 61)           # rank0 in a only
