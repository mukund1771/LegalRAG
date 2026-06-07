from __future__ import annotations

import re

from legal_rag.models import Evidence

DEFAULT_PARTY = "Acme Corp"

_STATE_RE = re.compile(r"laws of the State of ([A-Z][a-z]+)")


def _flag(risk_type: str, severity: str, ev: Evidence, rationale: str,
          party: str = DEFAULT_PARTY) -> dict:
    return {
        "risk_type": risk_type,
        "severity": severity,
        "affected_party": party,
        "rationale": rationale,
        "citation": ev.citation,
    }


class RiskAssessor:
    def __init__(self, llm=None, settings=None) -> None:
        self.llm = llm

    def assess(self, evidence: list[Evidence]) -> list[dict]:
        flags: list[dict] = []
        for ev in evidence:
            flags.extend(self._detect_clause(ev))
        flags.extend(self._detect_governing_law_conflict(evidence))

        seen, unique = set(), []
        for f in flags:
            key = (f["risk_type"], f["citation"])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        order = {"high": 0, "medium": 1, "low": 2}
        unique.sort(key=lambda f: order.get(f["severity"], 3))
        return unique

    def _detect_clause(self, ev: Evidence) -> list[dict]:
        text = ev.context_text.lower()
        out: list[dict] = []

        if re.search(r"\buncapped\b|\bunlimited\b", text) and "liab" in text:
            out.append(_flag("uncapped_liability", "high", ev,
                             "Liability is expressly uncapped for certain breaches."))

        if ("liab" in text and ("cap" in text or "limitation" in text)
                and re.search(r"(not apply|shall not apply).{0,80}confidential"
                              r"|breach of (its )?confidentiality", text)):
            out.append(_flag("cap_excludes_confidentiality", "high", ev,
                             "The liability cap excludes confidentiality breaches, so "
                             "that exposure is uncapped."))

        if ev.clause_type == "data_breach" and re.search(r"72|seventy-two", text):
            out.append(_flag("breach_notification_window", "medium", ev,
                             "Breach must be notified within 72 hours; late notice "
                             "triggers liability for fines and penalties."))

        if ev.clause_type == "subprocessor" or "subprocessor" in text or "subcontractor" in text:
            consent = "prior written consent" in text
            out.append(_flag(
                "subprocessor_data_sharing", "low" if consent else "high", ev,
                "Subprocessors may process data " +
                ("only with prior written consent." if consent
                 else "without explicit customer consent.")))

        if ev.clause_type == "indemnification" or "indemnif" in text:
            mutual = "each party" in text or "mutual" in text
            if not mutual:
                out.append(_flag("one_sided_indemnification", "medium", ev,
                                 "Indemnification appears one-sided rather than mutual."))

        if re.search(r"renews automatically|automatic(ally)? renew|auto-renew", text):
            out.append(_flag("auto_renewal", "low", ev,
                             "Agreement auto-renews unless notice of non-renewal is given."))

        return out
    
    def _detect_governing_law_conflict(self, evidence: list[Evidence]) -> list[dict]:
        laws: dict[str, str] = {}
        cites: dict[str, str] = {}
        for ev in evidence:
            if ev.clause_type == "governing_law":
                m = _STATE_RE.search(ev.context_text)
                if m:
                    laws[ev.doc_id] = m.group(1)
                    cites[ev.doc_id] = ev.citation

        distinct = set(laws.values())
        if len(distinct) > 1:
            detail = ", ".join(f"{d}: {j}" for d, j in laws.items())
            return [{
                "risk_type": "governing_law_conflict",
                "severity": "high",
                "affected_party": "all parties",
                "rationale": f"Agreements specify conflicting governing law ({detail}).",
                "citation": "; ".join(cites.values()),
            }]
        return []
