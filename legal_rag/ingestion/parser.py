"""Parse raw contracts (PDF/DOCX) into clean text + a section/clause tree.

Layout-aware; detects numbered sections and clause boundaries so the chunker can
build the parent-child hierarchy. Scanned docs route through an OCR pre-stage.
"""
from __future__ import annotations


def parse_document(path: str) -> "ParsedDoc":
    """Return a ParsedDoc with ordered sections and detected clause spans."""
    raise NotImplementedError
