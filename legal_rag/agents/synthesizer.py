from __future__ import annotations

from legal_rag.agents.prompts.templates import SYNTHESIZER, SYNTH_SYSTEM
from legal_rag.models import Answer, Evidence

ABSTAIN = "Not found in the provided contracts."


class Synthesizer:
    def __init__(self, llm, settings) -> None:
        self.llm = llm
        self.settings = settings

    def answer(self, query: str, evidence: list[Evidence]) -> Answer:
        if not evidence:
            return Answer(text=ABSTAIN, citations=[], abstained=True, evidence=[])

        context = "\n\n".join(f"{e.citation}\n{e.context_text}" for e in evidence)
        user = SYNTHESIZER.format(context=context, question=query)
        text = self.llm.complete(
            SYNTH_SYSTEM, user, temperature=0.1, max_tokens=512,
        )

        abstained = ABSTAIN.lower() in text.lower()
        citations = [] if abstained else list(dict.fromkeys(e.citation for e in evidence))
        return Answer(text=text, citations=citations, abstained=abstained, evidence=evidence)
