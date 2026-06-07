"""Tests for the Milestone 3 + 4 agent graph (synthesizer, planner, memory, orchestrator).

Runs fully offline: FakeEmbedder + LexicalReranker + FakeLLM + heuristic planner/
verifier. Exercises the end-to-end control flow on the sample queries, including the
out-of-scope refusal (Q16/Q17) and abstention paths.
"""

from __future__ import annotations

import os

import pytest

from legal_rag.agents.planner import Planner
from legal_rag.agents.synthesizer import ABSTAIN, Synthesizer
from legal_rag.app import build_system
from legal_rag.config.settings import load_settings
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.client import FakeLLM
from legal_rag.llm.embeddings import FakeEmbedder
from legal_rag.memory.session import SessionMemory
from legal_rag.models import Evidence

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "contracts")


def _offline_settings(index_dir: str):
    s = load_settings()
    s.index_dir = index_dir
    s.llm.backend = "fake"
    s.embedding.backend = "fake"
    s.retrieval.reranker_backend = "lexical"
    return s


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("idx"))
    build_index(CONTRACTS_DIR, FakeEmbedder(dim=256), index_dir)
    return _offline_settings(index_dir)


def _evidence():
    return [Evidence(
        chunk_id="c", doc_id="NDA_Acme_VendorXYZ", doc_type="NDA", section_no="4",
        section_heading="Term and Termination", clause_type="termination",
        child_text="Either party may terminate by providing thirty (30) days notice.",
        context_text=("4. Term and Termination\nEither party may terminate this "
                      "Agreement by providing thirty (30) days prior written notice."),
        citation="[NDA_Acme_VendorXYZ §4 Term and Termination]",
        char_start=0, char_end=10, score=1.0,
    )]


# ----------------------------------------------------------------- synthesizer

def test_synth_grounded_with_citations():
    syn = Synthesizer(FakeLLM(), load_settings())
    ans = syn.answer("What is the notice period?", _evidence())
    assert not ans.abstained
    assert ans.text.strip()
    assert ans.citations == ["[NDA_Acme_VendorXYZ §4 Term and Termination]"]


def test_synth_abstains_without_evidence():
    syn = Synthesizer(FakeLLM(), load_settings())
    ans = syn.answer("anything", [])
    assert ans.abstained and ans.text == ABSTAIN and ans.citations == []


# ----------------------------------------------------------------- planner

def test_planner_out_of_scope_drafting_and_advice():
    p = Planner(None, load_settings())
    assert p.plan("Can you draft a better NDA for me?")["in_scope"] is False
    assert p.plan("Can you draft a better NDA for me?")["intent"] == "out_of_scope_drafting"
    advice = p.plan("What legal strategy should Acme take against Vendor XYZ?")
    assert advice["in_scope"] is False and advice["intent"] == "out_of_scope_advice"


def test_planner_intents_and_filters():
    p = Planner(None, load_settings())
    p1 = p.plan("What is the notice period for terminating the NDA?")
    assert p1["in_scope"] and p1["filters"].get("doc_type") == "NDA"

    p2 = p.plan("Are there conflicting governing laws across agreements?")
    assert p2["intent"] == "cross_doc_compare"
    assert p2["filters"].get("clause_type") == "governing_law"
    assert "doc_type" not in p2["filters"]  # cross-doc must not pin one document

    p3 = p.plan("Summarize all risks for Acme Corp in one paragraph.")
    assert p3["intent"] == "summary" and p3["needs_risk_agent"]


# ----------------------------------------------------------------- memory

def test_memory_resolves_coreference():
    m = SessionMemory()
    m.add_turn("notice period for the NDA?", {"filters": {"doc_type": "NDA"}}, "...")
    rewritten = m.contextualize("Does it survive termination?")
    assert "NDA" in rewritten


# ----------------------------------------------------------------- orchestrator

def test_orchestrator_factual_answer_with_citation(settings):
    orch = build_system(settings)
    r = orch.handle_turn("What is the notice period for terminating the NDA?")
    assert not r.refused and not r.answer.abstained
    assert r.answer.citations
    assert any("nda" in c.lower() for c in r.answer.citations)


def test_orchestrator_refuses_out_of_scope(settings):
    orch = build_system(settings)
    r = orch.handle_turn("Draft a better NDA for me")
    assert r.refused and not r.answer.citations
    assert "draft" in r.answer.text.lower() or "advice" in r.answer.text.lower()


def test_orchestrator_abstains_when_no_evidence(settings):
    orch = build_system(settings)
    orch.retriever.retrieve = lambda *a, **k: []   # force empty retrieval
    r = orch.handle_turn("What is the office parking policy?")
    assert r.answer.abstained and r.answer.text == ABSTAIN


def test_orchestrator_multiturn_followup(settings):
    orch = build_system(settings)
    orch.handle_turn("What is the notice period for terminating the NDA?")
    r2 = orch.handle_turn("Do confidentiality obligations survive it?")
    # follow-up still resolves to a grounded NDA answer (coreference handled)
    assert not r2.answer.abstained


# ----------------------------------------------------------------- regression fixes

def test_planner_liability_data_breach_routes_to_risk():
    """Q7: 'Is X's liability capped for data breaches?' -> risk_analysis (not interp)."""
    p = Planner(None, load_settings())
    q7 = p.plan("Is Vendor XYZ's liability capped for data breaches?")
    assert q7["intent"] == "risk_analysis" and q7["needs_risk_agent"]
    # Q5 (confidentiality, not data breach) stays interpretation
    q5 = p.plan("Is liability capped for breach of confidentiality?")
    assert q5["intent"] == "interpretation"


def test_verifier_grade_rescues_confident_reranker_score():
    """Zero lexical overlap but a confident reranker score => 'correct' (not abstain)."""
    from legal_rag.agents.verifier import Verifier
    ev_relevant = [Evidence(
        chunk_id="c", doc_id="DPA", doc_type="DPA", section_no="4",
        section_heading="Subprocessors", clause_type="subprocessor",
        child_text="Processor may engage subprocessors with prior written authorization.",
        context_text="...", citation="[DPA §4 Subprocessors]",
        char_start=0, char_end=1, score=5.0)]   # high cross-encoder-style score
    v = Verifier(None, load_settings())
    # query shares no content tokens with the clause, but score is confident
    assert v.grade_retrieval("can they use subcontractors", ev_relevant) == "correct"
    # same clause but zero score (lexical, no overlap) => incorrect
    ev_relevant[0].score = 0.0
    assert v.grade_retrieval("xyzzy floopdoodle quux", ev_relevant) == "incorrect"


def test_planner_and_orchestrator_chitchat(settings):
    from legal_rag.agents.planner import Planner
    p = Planner(None, load_settings())
    assert p.plan("hello there")["intent"] == "chitchat"
    assert p.plan("hi")["intent"] == "chitchat"
    assert p.plan("what can you do")["intent"] == "chitchat"
    # a real question is NOT chitchat
    assert p.plan("What is the notice period for terminating the NDA?")["intent"] != "chitchat"

    orch = build_system(settings)
    r = orch.handle_turn("hello there")
    assert not r.refused and not r.answer.abstained
    assert "contracts" in r.answer.text.lower() and r.answer.citations == []
