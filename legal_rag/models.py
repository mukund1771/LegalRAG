"""Core data models shared across ingestion and retrieval.

These are deliberately plain dataclasses (no framework coupling) so they are easy
to serialize to JSON for the on-disk index and easy to assert on in tests.

The chunking strategy is parent-child:
- A ``Section`` is a top-level numbered section of a contract (the *parent*).
- A ``Clause`` is a sub-unit within a section (the *child*) that gets embedded and
  searched. When a section has no sub-clauses, the section body itself is the single
  child.
- A ``Chunk`` is the indexed unit. ``is_parent`` distinguishes the large parent
  (returned to the LLM for context) from the small child (embedded and searched).

Every unit carries character offsets into ``ParsedDoc.full_text`` so that answers can
cite an exact span — character-level citation is what makes a legal answer verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Clause:
    """A child unit (sub-clause / paragraph) within a section."""

    clause_no: str           # e.g. "4" or "4.1" or "4(a)"
    text: str
    char_start: int          # offset into ParsedDoc.full_text
    char_end: int
    clause_type: str = "general"


@dataclass
class Section:
    """A parent unit: one top-level numbered section of the contract."""

    section_no: str          # e.g. "4"
    heading: str             # e.g. "Term and Termination"
    text: str                # full section body (parent context)
    char_start: int
    char_end: int
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class ParsedDoc:
    """A parsed contract: clean text plus its section/clause structure + metadata."""

    doc_id: str              # stable id, e.g. "NDA_Acme_VendorXYZ"
    source_path: str
    doc_type: str            # NDA | MSA | SLA | DPA | Vendor | Unknown
    title: str
    parties: list[str]
    full_text: str
    sections: list[Section] = field(default_factory=list)

    def slice(self, start: int, end: int) -> str:
        """Return the exact source span — used to validate citations."""
        return self.full_text[start:end]


@dataclass
class ChunkMetadata:
    """Filterable metadata attached to every chunk (the backbone of precise retrieval)."""

    doc_id: str
    doc_type: str
    parties: list[str]
    section_no: str
    section_heading: str
    clause_type: str
    char_start: int
    char_end: int
    is_parent: bool
    parent_id: str | None = None


@dataclass
class Chunk:
    """The indexed unit.

    ``text`` is the raw clause/section text (what we show the user / LLM).
    ``embed_text`` is the context-augmented text we actually embed and tokenize for
    BM25 — it is prefixed with a document/section cue (Summary-Augmented Chunking) to
    fight document-level retrieval mismatch.
    """

    chunk_id: str
    text: str
    embed_text: str
    metadata: ChunkMetadata

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Chunk":
        meta = ChunkMetadata(**d["metadata"])
        return Chunk(
            chunk_id=d["chunk_id"],
            text=d["text"],
            embed_text=d["embed_text"],
            metadata=meta,
        )


@dataclass
class Evidence:
    """A retrieved piece of evidence handed to the synthesizer/risk agents.

    The match happens on the small ``child`` chunk, but ``context_text`` is the larger
    parent section (parent-child expansion) so the LLM reasons with full context. The
    ``citation`` is a human-readable, verifiable reference to the source span.
    """

    chunk_id: str            # the matched child chunk id
    doc_id: str
    doc_type: str
    section_no: str
    section_heading: str
    clause_type: str
    child_text: str          # the precise matched clause
    context_text: str        # the parent section (what the LLM reads)
    citation: str            # e.g. "[NDA_Acme_VendorXYZ §4 Term and Termination]"
    char_start: int
    char_end: int
    score: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """The synthesized response shown to the user.

    Citations are attached *programmatically* from the retrieved evidence (not parsed
    out of the LLM text), which guarantees every displayed citation is real and
    verifiable. ``abstained`` is True when the system declined to answer for lack of
    grounded evidence.
    """

    text: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class TurnResult:
    """Everything the orchestrator produced for one conversational turn."""

    answer: Answer
    plan: dict
    refused: bool = False
    risk_flags: list[dict] = field(default_factory=list)
