"""Provider-agnostic LLM client — one seam over Ollama (and an offline fake).

All agents call ``complete``; switching dev (Ollama) -> prod (vLLM/OpenAI, both
OpenAI-compatible) is a config change. This module also centralizes the determinism
and verbosity controls the brief asks for: temperature, fixed seed, JSON mode, and a
max-token cap, applied per call by each agent.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, *, model: str | None = None,
                 temperature: float = 0.0, json_mode: bool = False,
                 max_tokens: int = 512) -> str:
        ...


class OllamaClient:
    """Calls a local Ollama server's chat endpoint."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.base_url = settings.llm.base_url.rstrip("/")
        self.name = f"ollama:{settings.llm.reasoning_model}"

    def complete(self, system: str, user: str, *, model: str | None = None,
                 temperature: float = 0.0, json_mode: bool = False,
                 max_tokens: int = 512) -> str:
        import requests  # lazy import
        model = model or self.settings.llm.reasoning_model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": self.settings.llm.seed,     # determinism
                "num_predict": max_tokens,          # verbosity cap
            },
        }
        if json_mode:
            payload["format"] = "json"              # constrained JSON output
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class FakeLLM:
    """Deterministic offline client for tests/demo (no Ollama needed).

    For prose answers it returns an *extractive* grounded sentence pulled from the
    CONTEXT block of the prompt, so offline runs still demonstrate grounding. For JSON
    requests it returns an empty object (agents that need JSON offline use heuristics).
    """

    name = "fake-llm"

    def complete(self, system: str, user: str, *, model: str | None = None,
                 temperature: float = 0.0, json_mode: bool = False,
                 max_tokens: int = 512) -> str:
        if json_mode:
            return "{}"
        m = re.search(r"CONTEXT:\s*(.+?)\n\s*QUESTION:", user, re.S)
        ctx = (m.group(1) if m else user).strip()
        # drop the bracketed citation lines, keep clause prose
        lines = [ln for ln in ctx.splitlines() if ln.strip() and not ln.strip().startswith("[")]
        body = " ".join(lines)
        sents = re.split(r"(?<=[.])\s+", body)
        return " ".join(sents[:2]).strip() or "Not found in the provided contracts."


def get_llm(settings) -> LLMClient:
    """Factory: FakeLLM when llm.backend='fake', else the Ollama client."""
    if getattr(settings.llm, "backend", "ollama") == "fake":
        return FakeLLM()
    return OllamaClient(settings)
