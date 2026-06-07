"""Conversational memory: rolling history + structured session state.

Tracks resolved entities (which agreement, party, last clause_type) so follow-ups
like 'Does it survive termination?' rewrite into standalone queries.
"""
class SessionMemory:
    def add_turn(self, user: str, answer) -> None: raise NotImplementedError
    def contextualize(self, user_input: str) -> str: raise NotImplementedError
