from __future__ import annotations
from legal_rag.agents.prompts.templates import (
    CHITCHAT, REFUSAL_ADVICE, REFUSAL_DRAFTING,
)
from legal_rag.agents.synthesizer import ABSTAIN
from legal_rag.models import Answer, TurnResult

RISK_CLAUSE_TYPES = ["liability", "indemnification", "data_breach",
                     "subprocessor", "governing_law", "termination"]


class Orchestrator:
    def __init__(self, planner, retriever, synthesizer, verifier, risk, memory,
                 settings) -> None:
        self.planner = planner
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.verifier = verifier
        self.risk = risk
        self.memory = memory
        self.max_loops = settings.retrieval.max_corrective_loops

    def handle_turn(self, user_input: str) -> TurnResult:
        query = self.memory.contextualize(user_input)
        plan = self.planner.plan(query)

        # 0) greetings / meta -> friendly capability message (no retrieval)
        if plan.get("intent") == "chitchat":
            ans = Answer(text=CHITCHAT, citations=[], abstained=False)
            self.memory.add_turn(user_input, plan, ans.text)
            return TurnResult(answer=ans, plan=plan)

        if not plan.get("in_scope", True):
            refusal = (REFUSAL_DRAFTING if plan["intent"] == "out_of_scope_drafting"
                       else REFUSAL_ADVICE)
            ans = Answer(text=refusal, citations=[], abstained=False)
            self.memory.add_turn(user_input, plan, refusal)
            return TurnResult(answer=ans, plan=plan, refused=True)

        risk_intent = plan["intent"] in ("risk_analysis", "summary")
        if risk_intent:
            evidence = self._gather_risk_evidence(query)
        else:
            evidence = self._retrieve_with_correction(query, plan)

        if not evidence:
            ans = Answer(text=ABSTAIN, citations=[], abstained=True)
            self.memory.add_turn(user_input, plan, ABSTAIN)
            return TurnResult(answer=ans, plan=plan)

        ans = self.synthesizer.answer(query, evidence)
        faith = self.verifier.check_faithfulness(ans, evidence)
        if faith["verdict"] == "abstain" and not ans.abstained:
            ans = Answer(text=ABSTAIN, citations=[], abstained=True, evidence=evidence)

        flags = []
        if not ans.abstained and (risk_intent or plan.get("needs_risk_agent")):
            flags = self.risk.assess(evidence)

        self.memory.add_turn(user_input, plan, ans.text)
        return TurnResult(answer=ans, plan=plan, risk_flags=flags)

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
        # unreliable retrieval after correction -> abstain (better than a wrong answer)
        return [] if grade == "incorrect" else evidence

    def _gather_risk_evidence(self, query: str) -> list:
        merged: dict[str, object] = {}
        for clause_type in RISK_CLAUSE_TYPES:
            for ev in self.retriever.retrieve([query], {"clause_type": clause_type},
                                              final_k=2):
                merged.setdefault(ev.chunk_id, ev)
        return list(merged.values())

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
