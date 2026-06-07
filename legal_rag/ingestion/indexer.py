"""Embed child chunks and write them to the hybrid index (BM25 + dense + metadata)."""
from __future__ import annotations


def build_index(chunks: list["Chunk"]) -> None:
    """Embed and persist chunks to the vector store. Idempotent / incremental."""
    raise NotImplementedError
