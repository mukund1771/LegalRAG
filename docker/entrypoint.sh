#!/usr/bin/env bash
# Start Ollama, pull models (cached on the mounted volume), build the index, serve.
set -e

EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:14b-instruct}"

echo "==> starting ollama"
ollama serve &
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 1; done

echo "==> pulling models (cached in /root/.ollama if a volume is mounted there)"
ollama pull "$EMBED_MODEL"
ollama pull "$LLM_MODEL"

# Build the index once (skip if a persisted index already exists on the volume)
if [ ! -f "data/index/manifest.json" ]; then
  echo "==> ingesting corpus"
  python3 main.py --ingest --backend ollama
else
  echo "==> reusing existing index at data/index"
fi

echo "==> warming cross-encoder + starting web app on :8000"
exec python3 -m uvicorn legal_rag.web.app:app --host 0.0.0.0 --port 8000
