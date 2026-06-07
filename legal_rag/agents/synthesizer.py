"""Synthesizer — grounded answer strictly from retrieved evidence, with citations.

Uses brief IRAC for yes/no interpretation; abstains if evidence is absent.
"""
class Synthesizer:
    def answer(self, query: str, evidence: list) -> "Answer": raise NotImplementedError
