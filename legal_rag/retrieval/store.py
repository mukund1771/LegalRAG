"""On-disk index for the contract corpus (dev backing store).

For a small corpus a brute-force NumPy cosine over child vectors plus an in-memory
BM25 index is exact, dependency-light, and fast — no FAISS/Qdrant needed yet. The
class is the single seam the rest of the system goes through, so swapping in an ANN
store (Qdrant / pgvector) for the 10k+ scaling path is a localized change.

Persistence layout (under ``index_dir``):
- ``chunks.jsonl``       — every chunk (parents + children) as JSON
- ``child_vectors.npy``  — dense matrix aligned to the child chunk order
- ``manifest.json``      — dims, counts, embedder name

BM25 is rebuilt from child texts on load (cheap), so it is not serialized.

This module owns *storage and index building* (Milestone 1). Query methods
(dense/BM25 search, RRF, rerank) are added in Milestone 2 on top of this store.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from legal_rag.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class VectorStore:
    """Holds all chunks, the child dense matrix, and a BM25 index over children."""

    def __init__(self, dim: int, embedder_name: str = "") -> None:
        self.dim = dim
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self.child_ids: list[str] = []
        self.child_vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._bm25 = None  # built lazily / on load (used by Milestone 2)

    # ---------------------------------------------------------------- building

    def add(self, chunks: list[Chunk], child_vectors: np.ndarray) -> None:
        """Add chunks and the dense matrix for the child subset (order must match)."""
        children = [c for c in chunks if not c.metadata.is_parent]
        if child_vectors.shape[0] != len(children):
            raise ValueError(
                f"child_vectors rows ({child_vectors.shape[0]}) != "
                f"num children ({len(children)})"
            )
        for c in chunks:
            self.chunks.append(c)
            self._by_id[c.chunk_id] = c
        self.child_ids.extend(c.chunk_id for c in children)
        self.child_vectors = (
            child_vectors.astype(np.float32)
            if self.child_vectors.size == 0
            else np.vstack([self.child_vectors, child_vectors.astype(np.float32)])
        )

    def build_bm25(self) -> None:
        """Fit a BM25 index over child chunk texts (rank_bm25)."""
        from rank_bm25 import BM25Okapi  # lazy import
        corpus = [_tokenize(self.get(cid).embed_text) for cid in self.child_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def ensure_bm25(self):
        """Return the BM25 index, building it lazily if needed."""
        if self._bm25 is None:
            self.build_bm25()
        return self._bm25

    # ---------------------------------------------------------------- filtering

    @staticmethod
    def matches_filters(chunk: Chunk, filters: dict | None) -> bool:
        """True if a chunk satisfies metadata filters (doc_type/clause_type/doc_id/party).

        Filters are AND-ed. ``party`` matches if any party name contains the value
        (case-insensitive). Empty / falsy filter values are ignored, so a planner can
        emit partial filters without over-constraining retrieval.
        """
        if not filters:
            return True
        m = chunk.metadata
        for key, val in filters.items():
            if not val:
                continue
            if key == "doc_type" and m.doc_type.lower() != str(val).lower():
                return False
            if key == "clause_type" and m.clause_type.lower() != str(val).lower():
                return False
            if key == "doc_id" and m.doc_id.lower() != str(val).lower():
                return False
            if key == "party" and not any(str(val).lower() in p.lower() for p in m.parties):
                return False
        return True

    def child_index_filter(self, filters: dict | None) -> list[int]:
        """Indices (into child_ids/child_vectors) of children passing the filters."""
        return [
            i for i, cid in enumerate(self.child_ids)
            if self.matches_filters(self.get(cid), filters)
        ]

    # ---------------------------------------------------------------- accessors

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def parent_of(self, chunk_id: str) -> Chunk | None:
        pid = self.get(chunk_id).metadata.parent_id
        return self._by_id.get(pid) if pid else None

    def children(self) -> list[Chunk]:
        return [self._by_id[cid] for cid in self.child_ids]

    def stats(self) -> dict:
        parents = sum(1 for c in self.chunks if c.metadata.is_parent)
        return {
            "total_chunks": len(self.chunks),
            "parents": parents,
            "children": len(self.child_ids),
            "dim": self.dim,
            "embedder": self.embedder_name,
        }

    # ---------------------------------------------------------------- persistence

    def persist(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_json()) + "\n")
        np.save(os.path.join(index_dir, "child_vectors.npy"), self.child_vectors)
        with open(os.path.join(index_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"dim": self.dim, "embedder": self.embedder_name,
                 "child_ids": self.child_ids, **self.stats()},
                fh, indent=2,
            )

    @classmethod
    def load(cls, index_dir: str) -> "VectorStore":
        with open(os.path.join(index_dir, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        store = cls(dim=manifest["dim"], embedder_name=manifest.get("embedder", ""))
        chunks: list[Chunk] = []
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(Chunk.from_json(json.loads(line)))
        vectors = np.load(os.path.join(index_dir, "child_vectors.npy"))
        store.add(chunks, vectors)
        store.build_bm25()
        return store
