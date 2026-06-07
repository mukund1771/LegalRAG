"""Planner / Router — intent classification, scope guardrail, decomposition, filters.

Emits {intent, in_scope, sub_queries, filters, needs_risk_agent} as JSON (temp 0).
Out-of-scope drafting/advice (Q16/Q17) is diverted to a safe refusal here.
"""
class Planner:
    def plan(self, standalone_query: str) -> dict: raise NotImplementedError
