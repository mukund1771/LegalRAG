"""Entry point for the interactive legal-contract RAG console.

Usage:
    python main.py            # start the interactive console
    python main.py --ingest   # (re)build the index from data/contracts
    python main.py --eval      # run the evaluation harness
"""
from legal_rag.config.settings import load_settings


def main() -> None:
    settings = load_settings()
    # TODO: wire ingestion / console / eval entrypoints
    print("Legal-RAG — see DESIGN.md. Scaffold only; agents not yet implemented.")


if __name__ == "__main__":
    main()
