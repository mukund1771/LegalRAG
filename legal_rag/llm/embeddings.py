
from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of strings into an (n, dim) float32 matrix."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class FakeEmbedder:
    """Deterministic hashing embedder for tests / offline runs (no downloads).

    Each token is hashed into a fixed bucket; vectors are L2-normalized so cosine
    similarity is meaningful. This is NOT semantically strong — it exists purely to
    exercise the pipeline deterministically. Real retrieval quality comes from the
    Ollama / SentenceTransformer backends.
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = "fake-hash"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2_normalize(out)


class OllamaEmbedder:
    """Calls a local Ollama server's embedding endpoint (default model: bge-m3)."""

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434",
                 dim: int = 1024) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests  # lazy import
        vectors: list[list[float]] = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        mat = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    """Runs BGE-M3 (or any SentenceTransformer) in-process."""

    def __init__(self, model: str = "BAAI/bge-m3", dim: int = 1024) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension() or dim

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(mat, dtype=np.float32)


def get_embedder(settings) -> Embedder:
    """Factory: build the embedder named by ``settings.embedding.backend``."""
    emb = settings.embedding
    backend = getattr(emb, "backend", "ollama")
    if backend == "fake":
        return FakeEmbedder(dim=getattr(emb, "fake_dim", 256))
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model=emb.model, dim=emb.dim)
    return OllamaEmbedder(
        model=emb.model if "/" not in emb.model else "bge-m3",
        base_url=settings.llm.base_url,
        dim=emb.dim,
    )
