"""Conversational memory — rolling history + structured session state.

Maintains the entities the conversation is 'about' (last document type, clause type,
party) so a history-aware rewrite can turn a follow-up like 'Does it survive
termination?' into a standalone query ('Does it survive termination? (regarding the
NDA)'), which retrieval needs to work across turns.

Heuristic rewrite by default; an LLM rewrite is a drop-in upgrade at ``contextualize``.
"""

from __future__ import annotations

import re

_PRONOUN_RE = re.compile(r"\b(it|its|they|them|their|this|that|the agreement|the contract)\b",
                         re.IGNORECASE)
_DOC_MENTION_RE = re.compile(r"\b(nda|dpa|sla|msa|vendor|service|agreement|contract)\b",
                             re.IGNORECASE)


class SessionMemory:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.last_doc_type: str | None = None
        self.last_clause_type: str | None = None
        self.last_party: str | None = None

    def contextualize(self, user_input: str) -> str:
        """Resolve coreference into a standalone query using session state."""
        text = user_input.strip()
        has_pronoun = bool(_PRONOUN_RE.search(text))
        names_doc = bool(re.search(r"\b(nda|dpa|sla|msa|vendor services|services agreement)\b",
                                   text, re.IGNORECASE))
        if has_pronoun and not names_doc and self.last_doc_type:
            return f"{text} (regarding the {self.last_doc_type})"
        return text

    def add_turn(self, user_input: str, plan: dict, answer_text: str) -> None:
        filters = plan.get("filters", {}) if plan else {}
        if filters.get("doc_type"):
            self.last_doc_type = filters["doc_type"]
        if filters.get("clause_type"):
            self.last_clause_type = filters["clause_type"]
        if filters.get("party"):
            self.last_party = filters["party"]
        self.history.append({"user": user_input, "answer": answer_text})
