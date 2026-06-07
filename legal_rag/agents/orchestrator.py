"""Orchestrator — owns the per-turn control flow and conversation memory.

Flow (risk agent arrives in Milestone 5):

  contextualize(history) -> plan(intent/scope/filters)
      |--- out of scope ----------------------> safe refusal
      v
  retrieve(sub_queries, filters)
      |--- verifier.grade == 'incorrect' ----> corrective re-retrieval (relax filters),
      |                                          up to max_corrective_loops, else abstain
      v
  synthesize(grounded answer + citations)
      |--- verifier.faithfulness == 'abstain' -> downgrade to abstention
      v
  assemble TurnResult, update memory
"""

from __future__ import annotations

from legal_rag.agents.prompts.templates import REFUSAL_ADVICE, REFUSAL_DRAFTING
from legal_rag.agents.synthesizer import ABSTAIN
from legal_rag.models import Answer, TurnResult


class Orchestrator:
    def __init__(self, planner, retriever, synthesizer, verifier, memory, settings) -> None:
        self.planner = planner
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.verifier = verifier
        self.memory = memory
        self.max_loops = settings.retrieval.max_corrective_loops

    def handle_turn(self, user_input: str) -> TurnResult:
        query = self.memory.contextualize(user_input)
        plan = self.planner.plan(query)

        # 1) scope guardrail (sample queries 16, 17)
        if not plan.get("in_scope", True):
            refusal = (REFUSAL_DRAFTING if plan["intent"] == "out_of_scope_drafting"
                       else REFUSAL_ADVICE)
            ans = Answer(text=refusal, citations=[], abstained=False)
            self.memory.add_turn(user_input, plan, refusal)
            return TurnResult(answer=ans, plan=plan, refused=True)

        # 2) retrieve (+ CRAG corrective loop)
        evidence = self._retrieve_with_correction(query, plan)

        # 3) abstain if still no evidence
        if not evidence:
            ans = Answer(text=ABSTAIN, citations=[], abstained=True)
            self.memory.add_turn(user_input, plan, ABSTAIN)
            return TurnResult(answer=ans, plan=plan)

        # 4) synthesize + post-gen faithfulness check
        ans = self.synthesizer.answer(query, evidence)
        faith = self.verifier.check_faithfulness(ans, evidence)
        if faith["verdict"] == "abstain" and not ans.abstained:
            ans = Answer(text=ABSTAIN, citations=[], abstained=True, evidence=evidence)

        self.memory.add_turn(user_input, plan, ans.text)
        return TurnResult(answer=ans, plan=plan)

    # ------------------------------------------------------------------ internals

    def _retrieve_with_correction(self, query: str, plan: dict) -> list:
        filters = dict(plan.get("filters") or {})
        sub_queries = plan.get("sub_queries") or [query]

        evidence = self.retriever.retrieve(sub_queries, filters)
        grade = self.verifier.grade_retrieval(query, evidence)

        loops = 0
        while grade == "incorrect" and loops < self.max_loops:
            loops += 1
            filters = self._relax(filters)        # progressively widen
            evidence = self.retriever.retrieve(sub_queries, filters)
            grade = self.verifier.grade_retrieval(query, evidence)
        return evidence

    @staticmethod
    def _relax(filters: dict) -> dict:
        """Drop the most specific filter first (clause_type), then doc_type."""
        relaxed = dict(filters)
        if "clause_type" in relaxed:
            relaxed.pop("clause_type")
        elif "doc_type" in relaxed:
            relaxed.pop("doc_type")
        else:
            relaxed = {}
        return relaxed
