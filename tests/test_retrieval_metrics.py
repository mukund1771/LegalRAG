"""Unit tests for the ranking metrics + a smoke test of the retrieval-eval harness.

The metric math is verified against hand-computed values; the harness test confirms it
runs end-to-end and that qrels/gold IDs line up (using the offline FakeEmbedder).
"""

from __future__ import annotations

import math
import os

import pytest

from legal_rag.agents.planner import Planner
from legal_rag.agents.retriever import Retriever
from legal_rag.config.settings import load_settings
from legal_rag.eval.gold_set import GOLD
from legal_rag.eval.qrels import QRELS
from legal_rag.eval.retrieval_eval import evaluate_retrieval
from legal_rag.eval.retrieval_metrics import (
    hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k,
)
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import FakeEmbedder
from legal_rag.retrieval.rerank import LexicalReranker
from legal_rag.retrieval.store import VectorStore

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "contracts")


# ----------------------------------------------------------------- metric math

def test_mrr():
    assert mrr([False, True, False]) == pytest.approx(0.5)
    assert mrr([True]) == 1.0
    assert mrr([False, False]) == 0.0


def test_recall_and_precision():
    rel = [True, False, True, False]   # 2 relevant retrieved
    assert recall_at_k(rel, 4, n_relevant=3) == pytest.approx(2 / 3)
    assert recall_at_k(rel, 1, n_relevant=3) == pytest.approx(1 / 3)
    assert precision_at_k(rel, 4) == pytest.approx(0.5)
    assert hit_at_k(rel, 1) == 1.0
    assert hit_at_k([False, False], 2) == 0.0


def test_ndcg_perfect_and_imperfect():
    # perfect ranking: relevant items first
    assert ndcg_at_k([True, True, False], 3, n_relevant=2) == pytest.approx(1.0)
    # one relevant at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k([False, True], 2, n_relevant=1) == pytest.approx(expected)


def test_qrels_cover_gold_answerable():
    answerable = {it["id"] for it in GOLD
                  if it.get("in_scope", True) and not it.get("expect_abstain")}
    assert set(QRELS).issubset(answerable)


# ----------------------------------------------------------------- harness

def test_evaluate_retrieval_runs(tmp_path):
    index_dir = str(tmp_path / "idx")
    build_index(CONTRACTS_DIR, FakeEmbedder(dim=256), index_dir)
    store = VectorStore.load(index_dir)
    settings = load_settings()
    retriever = Retriever(store, FakeEmbedder(dim=256), LexicalReranker(), settings)
    planner = Planner(None, settings)

    report = evaluate_retrieval(retriever, planner)
    m = report["metrics"]
    assert report["n"] == len(QRELS)
    assert set(m) >= {"MRR", "recall@1", "recall@5", "nDCG@5", "hit@5"}
    # even the toy embedder should retrieve *something* relevant for most queries
    assert 0.0 <= m["recall@5"] <= 1.0
    assert m["hit@10"] >= 0.5
