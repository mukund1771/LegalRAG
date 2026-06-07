"""Tests for the Milestone 5 risk assessor and Milestone 6 evaluation harness.

Offline (FakeEmbedder + LexicalReranker + FakeLLM + heuristic agents). Asserts the
behaviours that must hold regardless of embedding quality: out-of-scope refusal,
abstention on absent answers, correct routing, and the key risk flags (the
governing-law conflict and the uncapped-liability/confidentiality-cap exposures).
"""

from __future__ import annotations

import os

import pytest

from legal_rag.agents.risk import RiskAssessor
from legal_rag.app import build_system
from legal_rag.config.settings import load_settings
from legal_rag.eval.gold_set import GOLD
from legal_rag.eval.runner import run_eval
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import FakeEmbedder
from legal_rag.models import Evidence

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "contracts")


@pytest.fixture(scope="module")
def orchestrator(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("idx"))
    build_index(CONTRACTS_DIR, FakeEmbedder(dim=256), index_dir)
    s = load_settings()
    s.index_dir = index_dir
    s.llm.backend = "fake"
    s.embedding.backend = "fake"
    s.retrieval.reranker_backend = "lexical"
    return build_system(s)


# ----------------------------------------------------------------- risk unit

def _ev(doc_id, doc_type, clause_type, heading, text):
    return Evidence(chunk_id=f"{doc_id}::x", doc_id=doc_id, doc_type=doc_type,
                    section_no="6", section_heading=heading, clause_type=clause_type,
                    child_text=text, context_text=text,
                    citation=f"[{doc_id} §6 {heading}]", char_start=0, char_end=1, score=1.0)


def test_risk_uncapped_and_confidentiality_cap():
    text = ("each party's total aggregate liability shall not exceed the fees paid. "
            "The foregoing limitation shall not apply to a party's breach of its "
            "confidentiality obligations or data breach obligations, for which "
            "liability shall be uncapped.")
    flags = {f["risk_type"] for f in RiskAssessor().assess(
        [_ev("MSA_x", "MSA", "liability", "Limitation of Liability", text)])}
    assert "uncapped_liability" in flags
    assert "cap_excludes_confidentiality" in flags


def test_risk_governing_law_conflict():
    evs = [
        _ev("NDA_x", "NDA", "governing_law", "Governing Law",
            "governed by the laws of the State of Delaware."),
        _ev("MSA_x", "MSA", "governing_law", "Governing Law",
            "governed by the laws of the State of California."),
    ]
    flags = {f["risk_type"] for f in RiskAssessor().assess(evs)}
    assert "governing_law_conflict" in flags


# ----------------------------------------------------------------- end-to-end

def test_q9_conflict_flag_through_orchestrator(orchestrator):
    r = orchestrator.handle_turn("Are there conflicting governing laws across agreements?")
    assert any(f["risk_type"] == "governing_law_conflict" for f in r.risk_flags)


def test_q12_unlimited_liability_flag(orchestrator):
    r = orchestrator.handle_turn("Is there any unlimited liability in these agreements?")
    assert any(f["risk_type"] == "uncapped_liability" for f in r.risk_flags)


# ----------------------------------------------------------------- harness

def test_eval_runs_and_meets_thresholds(orchestrator):
    report = run_eval(orchestrator, GOLD)
    m = report["metrics"]
    assert report["n"] == len(GOLD)
    assert m["refusal"] == 1.0          # Q16/Q17 always declined
    assert m["abstention"] >= 0.85      # adversarial abstain + answerable answered
    assert m["routing"] >= 0.85         # planner intent accuracy
    assert m["risk_recall"] >= 0.6      # key risk flags raised
