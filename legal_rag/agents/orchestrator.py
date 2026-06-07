"""Orchestrator — owns the multi-turn loop, memory, and final assembly.

Resolves coreference into a standalone query, routes through the agent graph,
and renders Answer | Citations | Risk flags back to the console.
"""
from __future__ import annotations


class Orchestrator:
    def __init__(self, planner, retriever, synthesizer, risk, verifier, memory):
        ...

    def handle_turn(self, user_input: str) -> "TurnResult":
        raise NotImplementedError
