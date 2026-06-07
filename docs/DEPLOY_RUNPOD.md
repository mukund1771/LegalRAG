# Deploying LegalRAG on RunPod (GPU)

The app needs a GPU to run `qwen2.5:14b` (LLM), `bge-m3` (embeddings) and the
`bge-reranker-v2-m3` cross-encoder. The provided `Dockerfile` bundles **Ollama + the
cross-encoder + the FastAPI web app** into one image, so a single GPU pod serves
everything on port `8000`.

## What runs where

```
RunPod GPU pod  ── container (this image) ──────────────────────────────┐
                   ollama serve  (:11434)  →  bge-m3 + qwen2.5            │
                   uvicorn       (:8000)   →  FastAPI /ask + chat UI      │
                   cross-encoder (in-process, sentence-transformers)      │
   network volume → /root/.ollama  (model cache, persists across restarts)│
                    /root/.cache/huggingface (cross-encoder cache)        │
────────────────────────────────────────────────────────────────────────┘
```

## Sizing

- **GPU:** 1× 24 GB (e.g. RTX 4090 / A5000) is comfortable for `qwen2.5:14b` (Q4) +
  `bge-m3` + the reranker. 16 GB works if you drop the LLM to `qwen2.5:7b` (set
  `LLM_MODEL` and `reasoning_model`). 12 GB → use `llama3.1:8b`.
- **Disk / volume:** ~15 GB for models (qwen 14B ≈ 9 GB, bge-m3 ≈ 2 GB, reranker ≈ 2 GB).
  Put it on a **network volume** so models aren't re-pulled on every restart.

## Option A — Custom container (recommended)

1. **Build & push the image** (from the repo root, on a machine with Docker):
   ```bash
   docker build -t <your-dockerhub-user>/legalrag:latest .
   docker push <your-dockerhub-user>/legalrag:latest
   ```

2. **Create the pod on RunPod:**
   - Template → *Custom* → Container image: `<your-dockerhub-user>/legalrag:latest`
   - **Expose HTTP port:** `8000`
   - **Volume:** create/attach a network volume, mount path `/root/.ollama`
     (so the 11 GB of models persist between restarts).
   - Optional env overrides: `LLM_MODEL`, `EMBED_MODEL`, `LEGALRAG_RERANKER`
     (`cross_encoder` default; set `lexical` to skip the cross-encoder).

3. **First boot** pulls the models and builds the index (a few minutes — watch the pod
   logs). Subsequent boots reuse the cached models and the persisted `data/index`.

4. Open the pod's **HTTP 8000** proxy URL → the chat UI. API at `POST /ask`,
   health at `/health`, OpenAPI at `/docs`.

## Option B — No Docker (quick, on a RunPod PyTorch/Ollama template)

SSH into a GPU pod that already has CUDA + Python, then:

```bash
git clone <your-repo> && cd LegalRAG
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-deploy.txt

# start ollama (if not already running) and pull models
ollama serve & 
ollama pull bge-m3 && ollama pull qwen2.5:14b-instruct

python main.py --ingest --backend ollama
python main.py --serve --port 8000        # or: uvicorn legal_rag.web.app:app --host 0.0.0.0 --port 8000
```

Expose port 8000 in the pod settings and open the proxy URL.

## Configuration (env vars)

| var | default | purpose |
|---|---|---|
| `LEGALRAG_BACKEND` | `ollama` | embeddings + LLM backend (`ollama` \| `fake`) |
| `LEGALRAG_RERANKER` | `cross_encoder` | `cross_encoder` (best) or `lexical` (no torch) |
| `LEGALRAG_INDEX_DIR` | `data/index` | prebuilt index location |
| `LLM_MODEL` | `qwen2.5:14b-instruct` | Ollama LLM tag (entrypoint) |
| `EMBED_MODEL` | `bge-m3` | Ollama embedding tag (entrypoint) |

## Updating the corpus

Add/replace files in `data/contracts/`, then re-ingest and restart:
```bash
python main.py --ingest --backend ollama && # restart the web process
```

## Production hardening (beyond this demo)

- Put the app behind the RunPod HTTPS proxy (done) or a reverse proxy with auth.
- Swap the in-memory session store for Redis if you run multiple replicas.
- For higher throughput, serve the LLM with **vLLM** instead of Ollama (same OpenAI-style
  API; see DESIGN.md §9) and move the vector index to Qdrant/pgvector (see README scaling).
- Pin model digests and bake them into the image (or a baked volume snapshot) to remove
  cold-start pulls.
