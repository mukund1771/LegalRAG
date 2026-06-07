"""Provider-agnostic LLM client. One seam over Ollama / vLLM / OpenAI.

All agents call this; switching dev (Ollama) -> prod (vLLM) is a config change.
Centralizes determinism controls: temperature, seed, top_p, max_tokens, JSON mode.
"""
from __future__ import annotations


class LLMClient:
    def __init__(self, settings): self.settings = settings

    def complete(self, system: str, user: str, *, model: str | None = None,
                 temperature: float = 0.0, json_mode: bool = False,
                 max_tokens: int = 512) -> str:
        """Single completion with explicit determinism/verbosity controls."""
        raise NotImplementedError
