from __future__ import annotations

import json
import re

from legal_rag.agents.prompts.templates import VERIFIER_POSTGEN, VERIFIER_POSTGEN_SYSTEM
from legal_rag.models import Answer, Evidence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
         "shall", "this", "that", "be", "by", "with", "as", "any", "its", "it"}


def _content(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


class Verifier:
    def __init__(self, llm=None, settings=None) -> None:
        self.llm = llm
        self.mode = getattr(settings.llm, "verifier_mode", "heuristic") if settings else "heuristic"
        self.min_groundedness = 0.18
        # a confident reranker score (e.g. cross-encoder) counts as relevant
        # even when the clause shares few tokens with the query
        self.min_rerank_relevance = 0.0

    def grade_retrieval(self, query: str, evidence: list[Evidence]) -> str:
        """Return 'correct' | 'ambiguous' | 'incorrect'."""
        if not evidence:
            return "incorrect"
        q = _content(query)
        if not q:
            return "correct"
        top = _content(evidence[0].child_text + " " + evidence[0].section_heading)
        coverage = len(q & top) / len(q)
        if coverage > 0:
            return "correct"
        # Zero lexical overlap, but a confident reranker (cross-encoder) still indicates
        # relevance -> don't falsely abstain on dense / paraphrase matches (e.g. Q13
        # "subcontractors" vs the "subprocessor" clause). For the lexical reranker this
        # branch is inert (its score is ~0 when overlap is 0), so offline behaviour is
        # unchanged and adversarial no-answer queries still abstain.
        if evidence[0].score > self.min_rerank_relevance:
            return "correct"
        return "incorrect"

    def check_faithfulness(self, answer: Answer, evidence: list[Evidence]) -> dict:
        """Return {'verdict': 'pass'|'abstain', 'groundedness': float}."""
        if answer.abstained or not evidence:
            return {"verdict": "abstain", "groundedness": 0.0}

        if self.mode == "llm" and self.llm is not None:
            try:
                return self._llm_faithfulness(answer, evidence)
            except Exception:
                pass

        ev_tokens: set[str] = set()
        for e in evidence:
            ev_tokens |= _content(e.context_text)
        ans = _content(answer.text)
        if not ans:
            return {"verdict": "abstain", "groundedness": 0.0}
        grounded = len(ans & ev_tokens) / len(ans)
        verdict = "pass" if grounded >= self.min_groundedness else "abstain"
        return {"verdict": verdict, "groundedness": round(grounded, 3)}

    def _llm_faithfulness(self, answer: Answer, evidence: list[Evidence]) -> dict:
        context = "\n\n".join(f"{e.citation}\n{e.context_text}" for e in evidence)
        raw = self.llm.complete(
            VERIFIER_POSTGEN_SYSTEM,
            VERIFIER_POSTGEN.format(context=context, answer=answer.text),
            temperature=0.0, json_mode=True, max_tokens=256,
        )
        data = json.loads(raw)
        verdict = "abstain" if data.get("verdict") == "abstain" else "pass"
        return {"verdict": verdict, "groundedness": 1.0 if verdict == "pass" else 0.0}
