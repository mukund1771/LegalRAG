"""Vector-store abstraction. FAISS/Chroma in dev -> Qdrant/pgvector in prod.

Keeps a single seam so the backing store can be swapped without touching agents.
Also exposes document-level routing hooks for large-corpus scaling (see DESIGN/Scaling).
"""
def get_store(settings): raise NotImplementedError
