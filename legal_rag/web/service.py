"""Web service layer — the testable core behind the HTTP API (no FastAPI import here).

Loads the heavy, stateless components (vector store, embedder, cross-encoder reranker,
LLM client, and the agents) ONCE, and keeps a lightweight per-session ``Orchestrator``
that only differs by its conversation memory. This makes multi-turn chat work without
re-loading the cross-encoder on every request.

Configuration is env-driven so the same code runs locally and in the RunPod container:
- LEGALRAG_BACKEND      ollama (default) | fake     (embeddings + LLM)
- LEGALRAG_RERANKER     cross_encoder (default) | lexical
- LEGALRAG_INDEX_DIR    path to the prebuilt index (default: settings.index_dir)
"""

from __future__ import annotations

import os

from legal_rag.agents.orchestrator import Orchestrator
from legal_rag.agents.planner import Planner
from legal_rag.agents.retriever import Retriever
from legal_rag.agents.risk import RiskAssessor
from legal_rag.agents.synthesizer import Synthesizer
from legal_rag.agents.verifier import Verifier
from legal_rag.config.settings import load_settings
from legal_rag.llm.client import get_llm
from legal_rag.llm.embeddings import get_embedder
from legal_rag.memory.session import SessionMemory
from legal_rag.retrieval.rerank import get_reranker
from legal_rag.retrieval.store import VectorStore

_shared: dict | None = None
_sessions: dict[str, Orchestrator] = {}


def _settings_from_env():
    s = load_settings()
    backend = os.getenv("LEGALRAG_BACKEND")
    if backend:
        s.llm.backend = backend
        s.embedding.backend = backend
    rr = os.getenv("LEGALRAG_RERANKER")
    if rr:
        s.retrieval.reranker_backend = rr
    idx = os.getenv("LEGALRAG_INDEX_DIR")
    if idx:
        s.index_dir = idx
    return s


def get_shared() -> dict:
    """Build (once) the shared, stateless components used by every session."""
    global _shared
    if _shared is None:
        s = _settings_from_env()
        store = VectorStore.load(str(s.index_dir))
        embedder = get_embedder(s)
        reranker = get_reranker(s)
        llm = get_llm(s)
        _shared = {
            "settings": s,
            "retriever": Retriever(store, embedder, reranker, s),
            "planner": Planner(llm, s),
            "synthesizer": Synthesizer(llm, s),
            "verifier": Verifier(llm, s),
            "risk": RiskAssessor(llm, s),
            "info": {"embedder": embedder.name, "reranker": reranker.name, "llm": llm.name},
        }
    return _shared


def _orchestrator(session_id: str) -> Orchestrator:
    if session_id not in _sessions:
        sh = get_shared()
        _sessions[session_id] = Orchestrator(
            sh["planner"], sh["retriever"], sh["synthesizer"],
            sh["verifier"], sh["risk"], SessionMemory(), sh["settings"],
        )
    return _sessions[session_id]


def answer_query(query: str, session_id: str = "default") -> dict:
    """Run one conversational turn; return a JSON-serializable response."""
    result = _orchestrator(session_id).handle_turn(query)
    ans = result.answer
    return {
        "answer": ans.text,
        "citations": ans.citations,
        "risk_flags": result.risk_flags,
        "intent": result.plan.get("intent"),
        "abstained": ans.abstained,
        "refused": result.refused,
    }


def system_info() -> dict:
    return get_shared()["info"]


def reset() -> None:
    """Drop caches (used by tests to re-read env)."""
    global _shared
    _shared = None
    _sessions.clear()
