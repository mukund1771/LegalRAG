#!/usr/bin/env bash
# Commit + push Milestone 2 (run on your machine). Assumes Milestone 1 is already
# committed; this branches off it so the M2 PR shows only retrieval changes.
set -e
cd "$(dirname "$0")/.."

rm -f .git/index.lock

git rev-parse --verify feature/m2-retrieval >/dev/null 2>&1 \
  && git checkout feature/m2-retrieval \
  || git checkout -b feature/m2-retrieval

git add legal_rag/retrieval/ legal_rag/agents/retriever.py legal_rag/models.py \
        legal_rag/config/settings.py main.py tests/test_retrieval.py README.md scripts/
git commit -m "feat(retrieval): Milestone 2 — hybrid search + RRF + rerank

- dense (cosine) and BM25 search with metadata pre-filtering
- Reciprocal Rank Fusion of the two ranked lists
- reranker interface: cross-encoder (bge-reranker-v2-m3) + offline lexical fallback
- Retriever agent: embed -> dense+BM25 -> RRF -> rerank -> parent-child expansion,
  multi sub-query merge, Evidence with verifiable citations
- python main.py --search; 16 passing offline tests"

git push -u origin feature/m2-retrieval
echo
echo "Open the PR: base=feature/m1-ingestion (or main if M1 is merged)  compare=feature/m2-retrieval"
