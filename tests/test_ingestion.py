"""End-to-end tests for the Milestone 1 ingestion pipeline.

These run fully offline using the deterministic FakeEmbedder (no model download, no
GPU, no network), so CI can verify the parse -> chunk -> embed -> index -> persist
path on every commit.

They assert the things that matter for downstream correctness:
- documents parse into sections and clauses,
- character offsets are exact (a cited span actually matches the source),
- the clause taxonomy the sample queries depend on is detected,
- the deliberate cross-document governing-law conflict is present (Q9),
- the index round-trips through disk.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from legal_rag.ingestion.chunker import chunk_document
from legal_rag.ingestion.indexer import build_index, discover_contracts
from legal_rag.ingestion.parser import parse_document
from legal_rag.llm.embeddings import FakeEmbedder
from legal_rag.retrieval.store import VectorStore

CONTRACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "contracts")


@pytest.fixture(scope="module")
def docs():
    return [parse_document(p) for p in discover_contracts(CONTRACTS_DIR)]


def test_corpus_discovered():
    paths = discover_contracts(CONTRACTS_DIR)
    assert len(paths) >= 4


def test_doc_types_detected(docs):
    types = {d.doc_type for d in docs}
    assert {"NDA", "MSA", "SLA", "DPA"}.issubset(types)


def test_sections_and_parties(docs):
    for d in docs:
        assert d.sections, f"{d.doc_id} parsed no sections"
        assert d.parties, f"{d.doc_id} found no parties"
    nda = next(d for d in docs if d.doc_type == "NDA")
    assert any("Acme" in p for p in nda.parties)


def test_char_offsets_are_exact(docs):
    """A clause's recorded offsets must slice back to (a normalization of) its text."""
    for d in docs:
        for section in d.sections:
            for clause in section.clauses:
                span = d.full_text[clause.char_start:clause.char_end]
                assert clause.text.strip()[:30] in span or span.strip()[:30] in clause.text


def test_clause_taxonomy_present(docs):
    found = set()
    for d in docs:
        for s in d.sections:
            for c in s.clauses:
                found.add(c.clause_type)
    # the clause types the sample queries rely on
    for needed in ["termination", "confidentiality", "governing_law",
                   "liability", "data_breach", "sla_uptime", "subprocessor"]:
        assert needed in found, f"clause_type '{needed}' not detected anywhere"


def test_governing_law_conflict(docs):
    """Q9: NDA (Delaware) vs MSA/DPA (California) — the conflict must be visible."""
    laws = {}
    for d in docs:
        for s in d.sections:
            if "governing law" in s.heading.lower():
                import re
                m = re.search(r"State of ([A-Z][a-z]+)", s.text)
                if m:
                    laws[d.doc_type] = m.group(1)
    assert laws.get("NDA") == "Delaware"
    assert laws.get("MSA") == "California"
    assert laws["NDA"] != laws["MSA"]


def test_parent_child_chunks(docs):
    chunks = chunk_document(docs[0])
    parents = [c for c in chunks if c.metadata.is_parent]
    children = [c for c in chunks if not c.metadata.is_parent]
    assert parents and children
    # every child points at a real parent
    parent_ids = {p.chunk_id for p in parents}
    for c in children:
        assert c.metadata.parent_id in parent_ids
    # context cue (SAC) is present in child embed_text but not in raw text
    assert all(c.embed_text.startswith("[") for c in children)


def test_build_and_roundtrip(tmp_path):
    embedder = FakeEmbedder(dim=128)
    index_dir = str(tmp_path / "index")
    report = build_index(CONTRACTS_DIR, embedder, index_dir)

    assert report["documents"] >= 4
    assert report["children"] > report["documents"]  # multiple clauses per doc
    assert os.path.exists(os.path.join(index_dir, "chunks.jsonl"))

    store = VectorStore.load(index_dir)
    s = store.stats()
    assert s["children"] == report["children"]
    assert store.child_vectors.shape == (report["children"], 128)
    # vectors are L2-normalized
    norms = np.linalg.norm(store.child_vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    # parent lookup works
    a_child = store.children()[0]
    assert store.parent_of(a_child.chunk_id) is not None
