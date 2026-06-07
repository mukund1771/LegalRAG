"""Verifier / Critic — the anti-hallucination spine.

pre_gen:  CRAG-style retrieval sufficiency grading + corrective re-retrieval loop.
post_gen: Self-RAG faithfulness — every answer claim must be entailed by a citation.
"""
class Verifier:
    def grade_retrieval(self, query: str, evidence: list) -> str: raise NotImplementedError   # correct|ambiguous|incorrect
    def check_faithfulness(self, answer, evidence: list) -> dict: raise NotImplementedError    # pass|revise|abstain
