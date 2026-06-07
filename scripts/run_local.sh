#!/usr/bin/env bash
# Run LegalRAG locally with real Ollama models (bge-m3 + qwen2.5).
# Prereqs: Ollama installed and running (`ollama serve` or the desktop app).
set -e
cd "$(dirname "$0")/.."

# Use whichever Python exists (macOS usually has python3, not python), and install
# deps into THAT SAME interpreter so there is no pip/python mismatch.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then echo "No python3 found on PATH."; exit 1; fi
echo "==> using interpreter: $PY"

echo "==> checking Ollama"
curl -sf http://localhost:11434/api/tags >/dev/null || {
  echo "Ollama not reachable on :11434 — start it (ollama serve) and retry."; exit 1; }

echo "==> pulling models (skips if present)"
ollama pull bge-m3
ollama pull qwen2.5:14b-instruct

echo "==> python deps (into $PY)"
"$PY" -m pip install -q -r requirements.txt

echo "==> ingest real corpus with bge-m3 embeddings"
"$PY" main.py --ingest --backend ollama

echo
echo "Done. Try:"
echo "  $PY main.py --backend ollama --reranker lexical            # interactive console"
echo "  $PY main.py --eval --backend ollama --reranker lexical      # system eval"
echo "  $PY main.py --retrieval-eval --backend ollama --reranker lexical   # recall@k/MRR/nDCG"
echo "  $PY main.py --ablation                                     # compare configs"
echo
echo "For the cross-encoder reranker (bonus, ~600MB download):"
echo "  $PY -m pip install sentence-transformers torch"
echo "  $PY main.py --eval --backend ollama        # default reranker = cross_encoder"
