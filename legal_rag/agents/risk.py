"""Risk Assessor — map clauses to the risk taxonomy -> structured flags.

Flag = {risk_type, severity, affected_party, rationale, citation}. No invented risk.
"""
class RiskAssessor:
    def assess(self, evidence: list) -> list[dict]: raise NotImplementedError
