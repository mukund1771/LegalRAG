"""Structure-aware, parent-child chunking with metadata + context augmentation.

- child  = single clause / sub-clause (embedded & searched; small, precise)
- parent = enclosing section (returned to the LLM for context)
- each child's embedding text is prefixed with a doc/section context cue (SAC)
  to fight document-level retrieval mismatch.
"""
from __future__ import annotations


def chunk_document(parsed: "ParsedDoc") -> list["Chunk"]:
    """Produce child chunks (with parent_id + metadata) ready for indexing."""
    raise NotImplementedError
