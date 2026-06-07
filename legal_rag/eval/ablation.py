"""Ablation runner — compare embedder / reranker combinations on retrieval metrics.

Re-ingests the corpus once per embedding backend (different embedders => different
vectors), then evaluates each reranker on top. Prints one comparison row per config so
you can see, on real numbers, what bge-m3 buys over the baseline and what the
cross-encoder reranker adds.

Configs whose backend isn't available (Ollama not running, sentence-transformers not
installed) are skipped with a reason rather than crashing the whole run.
"""

from __future__ import annotations

import tempfile

from legal_rag.agents.planner import Planner
from legal_rag.agents.retriever import Retriever
from legal_rag.config.settings import load_settings
from legal_rag.eval.retrieval_eval import evaluate_retrieval
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import get_embedder
from legal_rag.retrieval.rerank import get_reranker
from legal_rag.retrieval.store import VectorStore

# (embedding_backend, reranker_backend)
DEFAULT_MATRIX = [
    ("fake", "lexical"),            # baseline: no real model
    ("ollama", "lexical"),          # real embeddings, light reranker
    ("ollama", "cross_encoder"),    # real embeddings + cross-encoder (bonus)
]


def _settings_for(emb_backend: str, rr_backend: str):
    s = load_settings()
    s.embedding.backend = emb_backend
    s.llm.backend = "fake"  # ablation scores retrieval only; no generation needed
    s.retrieval.reranker_backend = rr_backend
    return s


def run_ablation(contracts_dir: str = "data/contracts",
                 matrix=None, ks=(1, 3, 5, 10)) -> list[dict]:
    matrix = matrix or DEFAULT_MATRIX
    index_cache: dict[str, str] = {}
    results: list[dict] = []

    for emb_backend, rr_backend in matrix:
        label = f"{emb_backend}/{rr_backend}"
        try:
            s = _settings_for(emb_backend, rr_backend)
            if emb_backend not in index_cache:
                idx = tempfile.mkdtemp(prefix=f"abl_{emb_backend}_")
                build_index(contracts_dir, get_embedder(s), idx)
                index_cache[emb_backend] = idx
            store = VectorStore.load(index_cache[emb_backend])
            retriever = Retriever(store, get_embedder(s), get_reranker(s), s)
            planner = Planner(None, s)
            rep = evaluate_retrieval(retriever, planner, ks=ks)
            results.append({"config": label, "metrics": rep["metrics"], "error": None})
        except Exception as exc:  # backend unavailable, etc.
            results.append({"config": label, "metrics": None, "error": str(exc)[:90]})
    return results


def format_ablation(results: list[dict], cols=("recall@1", "recall@5", "MRR", "nDCG@5")) -> str:
    lines = ["", "=== Ablation: retrieval metrics by config ===", ""]
    header = f"{'config':<26}" + "".join(f"{c:>12}" for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        if r["error"]:
            lines.append(f"{r['config']:<26}  skipped: {r['error']}")
            continue
        m = r["metrics"]
        lines.append(f"{r['config']:<26}" + "".join(f"{m.get(c, 0):>12}" for c in cols))
    lines.append("")
    return "\n".join(lines)
