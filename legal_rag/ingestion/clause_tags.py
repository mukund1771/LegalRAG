"""Heuristic clause-type and document-type tagging.

Why heuristics (keywords) rather than an LLM classifier at ingestion:
- The clause taxonomy the sample queries care about (termination, confidentiality,
  liability, governing_law, data_breach, sla_uptime, subprocessor, indemnification,
  survival) is small and lexically distinctive in contracts.
- Keyword tagging is deterministic, fast, free, and dependency-light — ideal for v1.
- ``clause_type`` is metadata used for retrieval *pre-filtering*, not for the final
  answer, so a few mis-tags degrade ranking slightly rather than producing wrong
  answers. An LLM tagger is a clean upgrade later behind the same function signature.

Heading text is weighted more heavily than body text, because section headings in
contracts are strong, low-noise signals of clause type.
"""

from __future__ import annotations

import re

# Ordered by priority: the first type whose pattern matches the heading wins.
# Each value is a list of regex fragments (case-insensitive).
CLAUSE_PATTERNS: dict[str, list[str]] = {
    "governing_law": [r"governing law", r"governed by", r"choice of law", r"jurisdiction"],
    "data_breach": [r"data breach", r"personal data breach", r"breach notification",
                    r"security incident", r"\b72 hours?\b"],
    "subprocessor": [r"subprocessor", r"sub-processor", r"subcontractor"],
    "sla_uptime": [r"uptime", r"availability", r"service level", r"service credit",
                   r"\b99\.\d+%"],
    "confidentiality": [r"confidential", r"non-disclosure", r"nondisclosure"],
    "survival": [r"survival", r"shall survive", r"survive the termination"],
    "termination": [r"termination", r"terminate", r"term and termination", r"non-renewal"],
    "indemnification": [r"indemnif"],
    "liability": [r"limitation of liability", r"liability", r"liable", r"\bcap\b"],
    "fees": [r"fees", r"payment", r"invoice"],
    "security": [r"technical and organizational", r"security measures"],
}

# Document type by filename hint first, then content keywords.
DOC_TYPE_FILENAME: dict[str, str] = {
    "nda": "NDA",
    "msa": "MSA",
    "sla": "SLA",
    "dpa": "DPA",
    "vendor": "Vendor",
}

DOC_TYPE_CONTENT: list[tuple[str, str]] = [
    (r"non-disclosure agreement", "NDA"),
    (r"data processing agreement", "DPA"),
    (r"service level agreement", "SLA"),
    (r"master (vendor )?services? agreement", "MSA"),
    (r"vendor services? agreement", "Vendor"),
]


def _matches(patterns: list[str], text: str) -> int:
    """Count how many patterns hit ``text`` (case-insensitive)."""
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def tag_clause(heading: str, body: str) -> str:
    """Return the best clause_type for a clause.

    Strategy: a heading match is decisive (return immediately, in priority order).
    Otherwise fall back to the body type with the most keyword hits; default 'general'.
    """
    heading = heading or ""
    for clause_type, patterns in CLAUSE_PATTERNS.items():
        if _matches(patterns, heading):
            return clause_type

    best_type, best_score = "general", 0
    for clause_type, patterns in CLAUSE_PATTERNS.items():
        score = _matches(patterns, body)
        if score > best_score:
            best_type, best_score = clause_type, score
    return best_type


def detect_doc_type(filename: str, text: str) -> str:
    """Infer NDA/MSA/SLA/DPA/Vendor from filename hints, then document title/body."""
    name = filename.lower()
    for hint, doc_type in DOC_TYPE_FILENAME.items():
        if hint in name:
            return doc_type

    head = text[:600].lower()
    for pattern, doc_type in DOC_TYPE_CONTENT:
        if re.search(pattern, head, flags=re.IGNORECASE):
            return doc_type
    return "Unknown"
