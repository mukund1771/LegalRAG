"""Planner / Router agent — intent, scope guardrail, decomposition, retrieval filters.

Two modes behind one interface:
- ``heuristic`` (default): deterministic keyword rules. Reliable, free, fully testable,
  and strong on this small, lexically-distinctive corpus.
- ``llm``: prompt the model for JSON (PLANNER template), falling back to the heuristic
  on any parse/again error. This is the path that generalizes to messier corpora.

Output contract (dict):
  intent, in_scope (bool), sub_queries (list[str]),
  filters ({doc_type?, clause_type?, party?}), needs_risk_agent (bool)

The scope guardrail lives here: drafting / legal-strategy requests (sample queries 16
and 17) are classified out of scope so the orchestrator can refuse safely.
"""

from __future__ import annotations

import json
import re

from legal_rag.agents.prompts.templates import PLANNER, PLANNER_SYSTEM
from legal_rag.ingestion.clause_tags import tag_clause

_DOC_TYPE_HINTS = [
    (r"\bnda\b|non-disclosure", "NDA"),
    (r"\bdpa\b|data processing", "DPA"),
    (r"\bsla\b|uptime|service level|availability", "SLA"),
    (r"vendor services? agreement|master .*agreement|\bmsa\b|services agreement", "Vendor"),
]

_DRAFTING_RE = re.compile(
    r"\b(draft|write|create|generate|compose|rewrite|redraft)\b.*"
    r"(nda|agreement|contract|clause|policy|better)", re.IGNORECASE)
_GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "sup", "howdy", "hola", "namaste",
              "greetings", "thanks", "thank", "thx", "ty", "tysm", "bye", "goodbye",
              "cheers", "gm"}
_META_RE = re.compile(
    r"\b(what can you do|what can i ask|who are you|what are you|"
    r"how do(es)? (you|this) work|what is this(?: app| tool)?$|what do you do)\b",
    re.IGNORECASE)
_ADVICE_RE = re.compile(
    r"\b(legal strategy|strategy should|should .*(take|sue|do)|advise|recommend|"
    r"what should .* do)\b", re.IGNORECASE)


class Planner:
    def __init__(self, llm=None, settings=None) -> None:
        self.llm = llm
        self.mode = getattr(settings.llm, "planner_mode", "heuristic") if settings else "heuristic"

    def plan(self, query: str) -> dict:
        if self.mode == "llm" and self.llm is not None:
            try:
                return self._validate(self._llm_plan(query), query)
            except Exception:
                pass  # fall back to deterministic heuristic
        return self._heuristic(query)


    def _llm_plan(self, query: str) -> dict:
        raw = self.llm.complete(PLANNER_SYSTEM, PLANNER.format(question=query),
                                temperature=0.0, json_mode=True, max_tokens=256)
        return json.loads(raw)

    def _validate(self, plan: dict, query: str) -> dict:
        base = self._heuristic(query)
        base.update({k: v for k, v in plan.items() if v is not None})
        return base

    def _heuristic(self, query: str) -> dict:
        q = query.lower()

        # 0) greetings / meta ("hi there", "what can you do") -> friendly chitchat
        _toks = q.replace("!", " ").replace(".", " ").replace("?", " ").split()
        _is_greeting = bool(_toks) and _toks[0] in _GREETINGS and len(_toks) <= 4
        _is_greeting = _is_greeting or (
            len(_toks) >= 2 and _toks[0] == "good"
            and _toks[1] in {"morning", "afternoon", "evening"})
        if _is_greeting or (len(_toks) <= 6 and _META_RE.search(q)):
            return {"intent": "chitchat", "in_scope": True, "sub_queries": [],
                    "filters": {}, "needs_risk_agent": False}

        if _DRAFTING_RE.search(q) or "better nda" in q:
            return self._scope_out("out_of_scope_drafting")
        if _ADVICE_RE.search(q):
            return self._scope_out("out_of_scope_advice")

        if "summar" in q:
            intent = "summary"
        elif any(w in q for w in ("conflict", "across", "all agreements", "all contracts",
                                  "which agreement", "governing laws", "in the contracts",
                                  "these agreements", "these contracts")):
            intent = "cross_doc_compare"
        elif (any(w in q for w in ("risk", "unlimited", "financial", "exposure"))
              or ("liabilit" in q and ("data breach" in q or "data breaches" in q))):
            # liability-for-data-breach is a risk-analysis question (Q7), not a plain
            # yes/no interpretation like Q5 (liability vs confidentiality).
            intent = "risk_analysis"
        elif q.startswith("what happens if") or "what remedies" in q or "if " in q and "beyond" in q:
            intent = "conditional"
        elif q.startswith(("do ", "does ", "is ", "are ", "can ")):
            intent = "interpretation"
        else:
            intent = "single_fact"

        filters = self._filters(query, intent)
        needs_risk = (
            intent in ("risk_analysis", "summary")
            or "conflict" in q
            or any(w in q for w in ("subcontractor", "subprocessor", "unlimited",
                                    "exposure", "financial risk"))
            or ("liabilit" in q and any(w in q for w in ("cap", "unlimited", "breach", "data"))))
        sub_queries = [query]

        return {
            "intent": intent,
            "in_scope": True,
            "sub_queries": sub_queries,
            "filters": filters,
            "needs_risk_agent": needs_risk,
        }

    def _filters(self, query: str, intent: str) -> dict:
        q = query.lower()
        filters: dict = {}
        for pattern, doc_type in _DOC_TYPE_HINTS:
            if re.search(pattern, q):
                filters["doc_type"] = doc_type
                break
        ct = tag_clause("", query)
        if ct != "general" and intent in ("single_fact", "cross_doc_compare"):
            filters["clause_type"] = ct
        if intent == "cross_doc_compare":
            filters.pop("doc_type", None)
        return filters

    @staticmethod
    def _scope_out(intent: str) -> dict:
        return {"intent": intent, "in_scope": False, "sub_queries": [],
                "filters": {}, "needs_risk_agent": False}
