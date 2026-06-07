"""Parent-child chunking with context-augmented embedding text.

For each section we emit:
- one **parent** chunk (the whole section) — returned to the LLM for context, not
  searched directly;
- one **child** chunk per clause — small and precise, this is what gets embedded and
  BM25-indexed.

Each child's ``embed_text`` is prefixed with a document/section *context cue*
(Summary-Augmented Chunking / contextual retrieval). This cue — e.g.
``[NDA between Acme, Vendor | §5 Survival | survival]`` — disambiguates which
agreement and section the clause belongs to and is the cheapest, highest-leverage
defense against document-level retrieval mismatch (pulling the right clause from the
wrong contract), the dominant failure mode in legal RAG.
"""

from __future__ import annotations

from legal_rag.models import Chunk, ChunkMetadata, ParsedDoc


def _context_cue(doc: ParsedDoc, section_no: str, heading: str, clause_type: str) -> str:
    parties = ", ".join(doc.parties) if doc.parties else "parties"
    return f"[{doc.doc_type} between {parties} | §{section_no} {heading} | {clause_type}]"


def _parent_id(doc_id: str, section_no: str) -> str:
    return f"{doc_id}::s{section_no}"


def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    """Produce parent + child chunks (with metadata and embed text) for one document."""
    chunks: list[Chunk] = []

    for section in doc.sections:
        pid = _parent_id(doc.doc_id, section.section_no)
        # parent clause_type: derive from the heading via the first child, else 'general'
        parent_type = section.clauses[0].clause_type if section.clauses else "general"

        parent_meta = ChunkMetadata(
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            parties=doc.parties,
            section_no=section.section_no,
            section_heading=section.heading,
            clause_type=parent_type,
            char_start=section.char_start,
            char_end=section.char_end,
            is_parent=True,
            parent_id=None,
        )
        # Parents are stored for context; we still give them an embed_text for
        # completeness, but the retriever searches children only.
        chunks.append(
            Chunk(
                chunk_id=pid,
                text=section.text,
                embed_text=section.text,
                metadata=parent_meta,
            )
        )

        for idx, clause in enumerate(section.clauses):
            cid = f"{pid}::c{idx}"
            cue = _context_cue(doc, section.section_no, section.heading, clause.clause_type)
            child_meta = ChunkMetadata(
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                parties=doc.parties,
                section_no=section.section_no,
                section_heading=section.heading,
                clause_type=clause.clause_type,
                char_start=clause.char_start,
                char_end=clause.char_end,
                is_parent=False,
                parent_id=pid,
            )
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=clause.text,
                    embed_text=f"{cue}\n{clause.text}",
                    metadata=child_meta,
                )
            )

    return chunks
