"""Entry point for the interactive legal-contract RAG console.

Usage:
    python main.py                 # start the interactive console (Milestone 3+)
    python main.py --ingest        # (re)build the index from data/contracts
    python main.py --ingest --backend fake   # offline ingest (no model download)
    python main.py --eval          # run the evaluation harness (Milestone 6)
"""

from __future__ import annotations

import argparse

from legal_rag.config.settings import load_settings
from legal_rag.ingestion.indexer import build_index
from legal_rag.llm.embeddings import get_embedder


def _run_ingest(settings, backend: str | None) -> None:
    if backend:
        settings.embedding.backend = backend
    embedder = get_embedder(settings)
    report = build_index(str(settings.data_dir), embedder, str(settings.index_dir))

    print("\n=== Ingestion complete ===")
    print(f"Embedder      : {report['embedder']}")
    print(f"Documents     : {report['documents']}")
    print(f"Parents/Child : {report['parents']} / {report['children']}")
    print("Doc types     :")
    for doc_id, dt in report["doc_types"].items():
        gl = report["governing_law"].get(doc_id, "—")
        print(f"  - {doc_id:32s} {dt:6s}  governing law: {gl}")
    print("Clause types  :")
    for ct, n in report["clause_types"].items():
        print(f"  - {ct:18s} {n}")
    print(f"Index written : {report['index_dir']}\n")


def _run_search(settings, query: str, backend: str | None, reranker_backend: str | None,
                k: int) -> None:
    from legal_rag.retrieval.store import VectorStore
    from legal_rag.retrieval.rerank import get_reranker
    from legal_rag.agents.retriever import Retriever

    if backend:
        settings.embedding.backend = backend
    if reranker_backend:
        settings.retrieval.reranker_backend = reranker_backend

    store = VectorStore.load(str(settings.index_dir))
    embedder = get_embedder(settings)
    reranker = get_reranker(settings)
    retriever = Retriever(store, embedder, reranker, settings)

    evidence = retriever.retrieve(query, final_k=k)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(evidence)} passages "
          f"(embedder={embedder.name}, reranker={reranker.name}):\n")
    for i, ev in enumerate(evidence, 1):
        print(f"{i}. {ev.citation}  [{ev.clause_type}]  score={ev.score:.3f}")
        snippet = ev.child_text.replace("\n", " ")
        print(f"   {snippet[:160]}{'…' if len(snippet) > 160 else ''}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalRAG console")
    parser.add_argument("--ingest", action="store_true", help="build the index")
    parser.add_argument("--search", metavar="QUERY", default=None,
                        help="retrieve evidence for a query (Milestone 2)")
    parser.add_argument("--eval", action="store_true", help="run evaluation")
    parser.add_argument("--backend", default=None,
                        help="embedding backend override: ollama | sentence_transformers | fake")
    parser.add_argument("--reranker", default=None,
                        help="reranker backend override: cross_encoder | lexical")
    parser.add_argument("-k", type=int, default=6, help="number of passages to return")
    args = parser.parse_args()

    settings = load_settings()

    if args.ingest:
        _run_ingest(settings, args.backend)
    elif args.search:
        _run_search(settings, args.search, args.backend, args.reranker, args.k)
    elif args.eval:
        print("Evaluation harness lands in Milestone 6.")
    else:
        print("Interactive console lands in Milestone 3. Run with --ingest / --search for now.")


if __name__ == "__main__":
    main()
