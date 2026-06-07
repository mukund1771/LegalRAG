from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel


class LLMSettings(BaseModel):
    provider: str = "ollama"            # ollama | vllm | openai  (provider-agnostic seam)
    backend: str = "ollama"             # ollama | fake (offline tests/demo)
    reasoning_model: str = "qwen2.5:14b-instruct"
    fast_model: str = "llama3.1:8b-instruct"
    base_url: str = "http://localhost:11434"
    seed: int = 42                      # determinism
    planner_mode: str = "heuristic"     # heuristic | llm
    verifier_mode: str = "heuristic"    # heuristic | llm


class EmbeddingSettings(BaseModel):
    # backend: ollama (local runtime) | sentence_transformers | fake (tests/offline)
    backend: str = "ollama"
    model: str = "bge-m3"               # local default; swap to legal-tuned in prod
    dim: int = 1024
    fake_dim: int = 256                 # dimension used by the FakeEmbedder in tests


class RetrievalSettings(BaseModel):
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60                     # RRF constant
    rerank_top_n: int = 32             # finalists into the cross-encoder
    final_k: int = 6                   # passages handed to the synthesizer
    reranker_backend: str = "cross_encoder"  # cross_encoder | lexical (offline/tests)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_corrective_loops: int = 2      # CRAG re-retrieval budget


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    data_dir: Path = Path("data/contracts")
    index_dir: Path = Path("data/index")


def load_settings(path: str | None = None) -> Settings:
    """Build settings from defaults, then apply environment overrides.

    Env vars (so ingest, serve, and the web app all agree — important when the index
    lives on a mounted volume in production):
      LEGALRAG_INDEX_DIR   where the index is read/written (e.g. a RunPod volume)
      LEGALRAG_DATA_DIR    the corpus directory
      LEGALRAG_BACKEND     ollama | fake   (embeddings + LLM)
      LEGALRAG_RERANKER    cross_encoder | lexical
      LEGALRAG_OLLAMA_URL  Ollama base URL (default http://localhost:11434)
    """
    s = Settings()
    if os.getenv("LEGALRAG_INDEX_DIR"):
        s.index_dir = Path(os.environ["LEGALRAG_INDEX_DIR"])
    if os.getenv("LEGALRAG_DATA_DIR"):
        s.data_dir = Path(os.environ["LEGALRAG_DATA_DIR"])
    if os.getenv("LEGALRAG_BACKEND"):
        s.llm.backend = s.embedding.backend = os.environ["LEGALRAG_BACKEND"]
    if os.getenv("LEGALRAG_RERANKER"):
        s.retrieval.reranker_backend = os.environ["LEGALRAG_RERANKER"]
    if os.getenv("LEGALRAG_OLLAMA_URL"):
        s.llm.base_url = os.environ["LEGALRAG_OLLAMA_URL"]
    return s
