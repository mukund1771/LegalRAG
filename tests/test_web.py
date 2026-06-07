"""Web service tests — exercise the testable core (answer_query) offline.

No FastAPI/HTTP needed: we call the service layer directly with the FakeEmbedder +
lexical reranker + FakeLLM via env vars, against a temp index over the fixture corpus.
"""

from __future__ import annotations

import os

import pytest

from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import FakeEmbedder

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "contracts")


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    index_dir = str(tmp_path_factory.mktemp("idx"))
    build_index(CONTRACTS_DIR, FakeEmbedder(dim=256), index_dir)
    os.environ["LEGALRAG_BACKEND"] = "fake"
    os.environ["LEGALRAG_RERANKER"] = "lexical"
    os.environ["LEGALRAG_INDEX_DIR"] = index_dir
    from legal_rag.web import service as svc
    svc.reset()
    yield svc
    svc.reset()
    for k in ("LEGALRAG_BACKEND", "LEGALRAG_RERANKER", "LEGALRAG_INDEX_DIR"):
        os.environ.pop(k, None)


def test_factual_answer(service):
    out = service.answer_query("What is the notice period for terminating the NDA?", "s1")
    assert not out["refused"] and not out["abstained"]
    assert out["citations"] and any("nda" in c.lower() for c in out["citations"])


def test_out_of_scope_refusal(service):
    out = service.answer_query("Draft a better NDA for me", "s2")
    assert out["refused"] and out["citations"] == []


def test_risk_query_returns_flags(service):
    out = service.answer_query("Are there conflicting governing laws across agreements?", "s3")
    assert any(f["risk_type"] == "governing_law_conflict" for f in out["risk_flags"])


def test_multiturn_session(service):
    service.answer_query("What is the notice period for terminating the NDA?", "s4")
    out = service.answer_query("Do confidentiality obligations survive it?", "s4")
    assert not out["abstained"]
