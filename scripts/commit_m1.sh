#!/usr/bin/env bash
# Commit + push Milestone 1 (run on your machine, where git can write).
set -e
cd "$(dirname "$0")/.."

rm -f .git/index.lock          # clear the stale lock left by the sandbox
git reset -q                   # start from a clean staging area

# you should already be on feature/m1-ingestion; create it if not
git rev-parse --verify feature/m1-ingestion >/dev/null 2>&1 \
  && git checkout feature/m1-ingestion \
  || git checkout -b feature/m1-ingestion

# Commit 1 — doc/scope finalization
git add DESIGN.md DESIGN_RATIONALE.md README.md docs/diagram_1_ingestion_pipeline.svg .gitignore
git commit -m "docs: finalize design rationale and drop OCR from v1 scope"

# Commit 2 — Milestone 1 ingestion pipeline
git add legal_rag/ data/contracts/*.md tests/ pyproject.toml scripts/ main.py requirements.txt
git commit -m "feat(ingestion): Milestone 1 — parser, parent-child chunker, embeddings, index

- structure-aware parser (digital-text PDF/DOCX/MD/TXT) with char offsets
- parent-child chunking + Summary-Augmented context cues
- clause-type + doc-type tagging heuristics
- Embedder interface: Ollama (bge-m3) / SentenceTransformers / Fake (offline tests)
- NumPy + BM25 vector store with JSON/npy persistence
- python main.py --ingest; 8 passing offline tests"

git push -u origin feature/m1-ingestion
echo
echo "Now open the PR: base=main  compare=feature/m1-ingestion"
