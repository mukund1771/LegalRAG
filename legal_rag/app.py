"""Application wiring — build the full agent graph from settings.

One place that assembles the index, retriever, LLM, and agents into an Orchestrator.
Used by the console and the tests, so both exercise the same composition.
"""

from __future__ import annotations

from legal_rag.agents.orchestrator import Orchestrator
from legal_rag.agents.planner import Planner
from legal_rag.agents.retriever import Retriever
from legal_rag.agents.risk import RiskAssessor
from legal_rag.agents.synthesizer import Synthesizer
from legal_rag.agents.verifier import Verifier
from legal_rag.llm.client import get_llm
from legal_rag.llm.embeddings import get_embedder
from legal_rag.memory.session import SessionMemory
from legal_rag.retrieval.rerank import get_reranker
from legal_rag.retrieval.store import VectorStore


def build_system(settings) -> Orchestrator:
    store = VectorStore.load(str(settings.index_dir))
    embedder = get_embedder(settings)
    reranker = get_reranker(settings)
    retriever = Retriever(store, embedder, reranker, settings)

    llm = get_llm(settings)
    synthesizer = Synthesizer(llm, settings)
    planner = Planner(llm, settings)
    verifier = Verifier(llm, settings)
    risk = RiskAssessor(llm, settings)
    memory = SessionMemory()

    return Orchestrator(planner, retriever, synthesizer, verifier, risk, memory, settings)
