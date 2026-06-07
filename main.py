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
    parser.add_argument("--eval", action="store_true", help="run system evaluation")
    parser.add_argument("--retrieval-eval", action="store_true",
                        help="retrieval metrics (recall@k/MRR/nDCG) vs qrels")
    parser.add_argument("--ablation", action="store_true",
                        help="compare embedder/reranker configs on retrieval metrics")
    parser.add_argument("--serve", action="store_true", help="run the FastAPI web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
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
        from legal_rag.app import build_system
        from legal_rag.eval.runner import format_report, run_eval
        if args.backend:
            settings.llm.backend = args.backend
            settings.embedding.backend = args.backend
        if args.reranker:
            settings.retrieval.reranker_backend = args.reranker
        orchestrator = build_system(settings)
        print(format_report(run_eval(orchestrator)))
    elif args.retrieval_eval:
        from legal_rag.agents.planner import Planner
        from legal_rag.agents.retriever import Retriever
        from legal_rag.eval.retrieval_eval import evaluate_retrieval, format_retrieval_report
        from legal_rag.llm.embeddings import get_embedder
        from legal_rag.retrieval.rerank import get_reranker
        from legal_rag.retrieval.store import VectorStore
        if args.backend:
            settings.embedding.backend = args.backend
        if args.reranker:
            settings.retrieval.reranker_backend = args.reranker
        store = VectorStore.load(str(settings.index_dir))
        retriever = Retriever(store, get_embedder(settings), get_reranker(settings), settings)
        planner = Planner(None, settings)
        print(format_retrieval_report(evaluate_retrieval(retriever, planner)))
    elif args.ablation:
        from legal_rag.eval.ablation import format_ablation, run_ablation
        print(format_ablation(run_ablation(str(settings.data_dir))))
    elif args.serve:
        import uvicorn
        uvicorn.run("legal_rag.web.app:app", host=args.host, port=args.port)
    else:
        from legal_rag.cli.console import repl
        if args.backend:
            settings.llm.backend = args.backend
            settings.embedding.backend = args.backend
        if args.reranker:
            settings.retrieval.reranker_backend = args.reranker
        repl(settings)


if __name__ == "__main__":
    main()
