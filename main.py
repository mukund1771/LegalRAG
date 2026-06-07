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


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalRAG console")
    parser.add_argument("--ingest", action="store_true", help="build the index")
    parser.add_argument("--eval", action="store_true", help="run evaluation")
    parser.add_argument("--backend", default=None,
                        help="embedding backend override: ollama | sentence_transformers | fake")
    args = parser.parse_args()

    settings = load_settings()

    if args.ingest:
        _run_ingest(settings, args.backend)
    elif args.eval:
        print("Evaluation harness lands in Milestone 6.")
    else:
        print("Interactive console lands in Milestone 3. Run with --ingest for now.")


if __name__ == "__main__":
    main()
