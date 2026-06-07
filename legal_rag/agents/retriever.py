"""Retriever agent — hybrid search -> RRF -> cross-encoder rerank -> parent expansion.

Runs sub-queries in parallel and applies metadata pre-filters from the planner.
"""
class Retriever:
    def retrieve(self, sub_queries: list[str], filters: dict) -> list: raise NotImplementedError
