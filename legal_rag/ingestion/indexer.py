"""Ingestion entry point: corpus directory -> parsed -> chunked -> embedded -> index.

``build_index`` is the offline pipeline behind ``python main.py --ingest``. It walks
the contracts directory, parses and chunks each document, embeds only the *child*
chunks (parents are stored for context, not searched), writes the index to disk, and
returns a human-readable report used by the CLI.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from legal_rag.ingestion.chunker import chunk_document
from legal_rag.ingestion.parser import parse_document
from legal_rag.llm.embeddings import Embedder
from legal_rag.models import Chunk, ParsedDoc
from legal_rag.retrieval.store import VectorStore

SUPPORTED_EXT = (".md", ".txt", ".pdf", ".docx")


def discover_contracts(contracts_dir: str) -> list[str]:
    """Return sorted paths of supported contract files in the directory."""
    paths = []
    for name in sorted(os.listdir(contracts_dir)):
        if name.lower().endswith(SUPPORTED_EXT) and name.lower() != "readme.md":
            paths.append(os.path.join(contracts_dir, name))
    return paths


def build_index(contracts_dir: str, embedder: Embedder, index_dir: str) -> dict:
    """Parse + chunk + embed + persist the corpus. Returns an ingestion report."""
    paths = discover_contracts(contracts_dir)
    if not paths:
        raise FileNotFoundError(f"No contracts ({SUPPORTED_EXT}) found in {contracts_dir}")

    parsed_docs: list[ParsedDoc] = []
    all_chunks: list[Chunk] = []
    for path in paths:
        doc = parse_document(path)
        parsed_docs.append(doc)
        all_chunks.extend(chunk_document(doc))

    children = [c for c in all_chunks if not c.metadata.is_parent]
    child_vectors = (
        embedder.embed([c.embed_text for c in children])
        if children
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )

    store = VectorStore(dim=embedder.dim, embedder_name=embedder.name)
    store.add(all_chunks, child_vectors)
    store.build_bm25()
    store.persist(index_dir)

    clause_counts = Counter(c.metadata.clause_type for c in children)
    return {
        "documents": len(parsed_docs),
        "doc_types": {d.doc_id: d.doc_type for d in parsed_docs},
        "governing_law": {d.doc_id: _find_governing_law(d) for d in parsed_docs},
        "parents": sum(1 for c in all_chunks if c.metadata.is_parent),
        "children": len(children),
        "clause_types": dict(sorted(clause_counts.items(), key=lambda kv: -kv[1])),
        "index_dir": index_dir,
        "embedder": embedder.name,
    }


def _find_governing_law(doc: ParsedDoc) -> str:
    """Best-effort extraction of the jurisdiction for the ingestion report/demo."""
    import re
    for section in doc.sections:
        if any(c.clause_type == "governing_law" for c in section.clauses) or \
                "governing law" in section.heading.lower():
            m = re.search(r"laws of the State of ([A-Z][a-z]+)", section.text)
            if m:
                return m.group(1)
    return "—"
