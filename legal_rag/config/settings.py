"""Typed configuration loaded from config.yaml + environment.

Single source of truth for model names, retrieval k's, temperatures and paths,
so every component reads from one place (readable configuration management).
"""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel


class LLMSettings(BaseModel):
    provider: str = "ollama"            # ollama | vllm | openai  (provider-agnostic seam)
    reasoning_model: str = "qwen2.5:14b-instruct"
    fast_model: str = "llama3.1:8b-instruct"
    base_url: str = "http://localhost:11434"
    seed: int = 42                      # determinism


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
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    max_corrective_loops: int = 2      # CRAG re-retrieval budget


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    data_dir: Path = Path("data/contracts")
    index_dir: Path = Path("data/index")


def load_settings(path: str | None = None) -> Settings:
    """Load settings from YAML (falling back to defaults). TODO: merge yaml/env."""
    return Settings()
