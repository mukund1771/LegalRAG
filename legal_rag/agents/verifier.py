"""Verifier / Critic agent — the anti-hallucination spine.

pre-generation (CRAG-style):  grade whether retrieved evidence is sufficient; the
orchestrator uses the grade to trigger a corrective re-retrieval or abstain.

post-generation (Self-RAG-style):  check the drafted answer is grounded in the cited
evidence; an unsupported answer is downgraded to abstention.

Both have a deterministic heuristic mode (default) and an optional LLM mode.
"""

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

    # ------------------------------------------------------ pre-generation (CRAG)

    def grade_retrieval(self, query: str, evidence: list[Evidence]) -> str:
        """Return 'correct' | 'ambiguous' | 'incorrect'."""
        if not evidence:
            return "incorrect"
        # crude confidence: does the top evidence share content with the query?
        q = _content(query)
        if not q:
            return "correct"
        top = _content(evidence[0].child_text + " " + evidence[0].section_heading)
        coverage = len(q & top) / len(q)
        if coverage == 0:
            return "ambiguous"
        return "correct"

    # ----------------------------------------------------- post-generation (Self-RAG)

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
