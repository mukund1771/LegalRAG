# LegalRAG — self-contained image: Ollama (bge-m3 + qwen2.5) + cross-encoder + web app.
# Built on the official Ollama image (ships the Ollama server + CUDA runtime).
FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive \
    LEGALRAG_BACKEND=ollama \
    LEGALRAG_RERANKER=cross_encoder \
    OLLAMA_HOST=0.0.0.0:11434 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-deploy.txt ./
# torch + sentence-transformers (cross-encoder) + app deps
RUN pip3 install --no-cache-dir --break-system-packages \
        -r requirements.txt -r requirements-deploy.txt

COPY . .

EXPOSE 8000
# override the base image's ollama entrypoint with our launcher
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
