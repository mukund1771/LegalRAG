
from __future__ import annotations

import os
import re

from legal_rag.models import Clause, ParsedDoc, Section
from legal_rag.ingestion.clause_tags import detect_doc_type, tag_clause

# A child clause is kept small for precise retrieval; long paragraphs are split.
MAX_CHILD_CHARS = 400

# Top-level section header, e.g. "4. Term and Termination" on its own line.
_SECTION_RE = re.compile(r"^[ \t]*(\d+)\.[ \t]+(.+?)[ \t]*$", re.MULTILINE)


# --------------------------------------------------------------------------- IO

def _read_text(path: str) -> str:
    """Extract digital text from .md/.txt/.pdf/.docx. No OCR."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    if ext == ".pdf":
        from pypdf import PdfReader  # lazy import
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx  # python-docx, lazy import
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {ext} ({path})")


def _normalize(text: str) -> str:
    """Light normalization that preserves character offsets meaningfully."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ----------------------------------------------------------------- metadata

def _detect_title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "Untitled"


def _detect_parties(text: str) -> list[str]:
    """Extract party names from the intro: legal names before a quoted defined term."""
    intro = text[:800]
    parties: list[str] = []
    # e.g. 'Acme Corp ("Acme")' or 'Vendor XYZ Inc. ("Vendor")'
    for m in re.finditer(r"([A-Z][A-Za-z0-9&.,\-]*(?:\s+[A-Z][A-Za-z0-9&.,\-]*){0,4})\s*\(\"", intro):
        name = m.group(1).strip(" ,.")
        if name and name not in parties and len(name) > 2:
            parties.append(name)
    return parties


# ------------------------------------------------------------- child splitting

def _split_children(body: str, base: int) -> list[tuple[str, int, int]]:
    """Split a section body into child spans (text, abs_start, abs_end).

    Paragraphs become children; an over-long paragraph is further split into
    sentence-grouped children so each child stays small and precise.
    """
    units: list[tuple[str, int, int]] = []
    for para in re.finditer(r"[^\n].*?(?=\n[ \t]*\n|\Z)", body, re.S):
        ptext, pstart = para.group(0), base + para.start()
        if len(ptext) <= MAX_CHILD_CHARS:
            units.append((ptext.strip(), pstart, pstart + len(ptext)))
            continue
        # sentence-group long paragraphs
        cursor = 0
        buf, buf_start = "", None
        for sent in re.finditer(r".+?(?:\.\s|\.$|$)", ptext, re.S):
            s_abs = pstart + sent.start()
            if buf_start is None:
                buf_start = s_abs
            buf += sent.group(0)
            if len(buf) >= MAX_CHILD_CHARS:
                units.append((buf.strip(), buf_start, buf_start + len(buf)))
                buf, buf_start = "", None
        if buf.strip():
            units.append((buf.strip(), buf_start, buf_start + len(buf)))
    return [u for u in units if u[0]]


# ----------------------------------------------------------------- main entry

def parse_document(path: str) -> ParsedDoc:
    """Parse one contract file into a ParsedDoc (sections + clauses + offsets)."""
    raw = _read_text(path)
    full_text = _normalize(raw)

    doc_id = os.path.splitext(os.path.basename(path))[0]
    title = _detect_title(full_text)
    parties = _detect_parties(full_text)
    doc_type = detect_doc_type(os.path.basename(path), full_text)

    # Locate section headers; each section runs until the next header (or EOF).
    headers = list(_SECTION_RE.finditer(full_text))
    sections: list[Section] = []
    for i, h in enumerate(headers):
        section_no = h.group(1)
        heading = h.group(2).strip()
        body_start = h.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        body = full_text[body_start:body_end].strip("\n")
        # absolute offsets: account for stripped leading newlines
        lead = len(full_text[body_start:body_end]) - len(full_text[body_start:body_end].lstrip("\n"))
        abs_body_start = body_start + lead

        clauses: list[Clause] = []
        for ctext, cstart, cend in _split_children(body, abs_body_start):
            clauses.append(
                Clause(
                    clause_no=section_no,
                    text=ctext,
                    char_start=cstart,
                    char_end=cend,
                    clause_type=tag_clause(heading, ctext),
                )
            )
        sections.append(
            Section(
                section_no=section_no,
                heading=heading,
                text=full_text[h.start():body_end].strip(),
                char_start=h.start(),
                char_end=body_end,
                clauses=clauses,
            )
        )

    return ParsedDoc(
        doc_id=doc_id,
        source_path=path,
        doc_type=doc_type,
        title=title,
        parties=parties,
        full_text=full_text,
        sections=sections,
    )
