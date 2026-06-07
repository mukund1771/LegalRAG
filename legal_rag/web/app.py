"""FastAPI app: REST endpoint + chat UI over the LegalRAG orchestrator.

Run locally:   uvicorn legal_rag.web.app:app --host 0.0.0.0 --port 8000
or:            python main.py --serve
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from legal_rag.web import service

app = FastAPI(title="LegalRAG", version="1.0")

_STATIC = os.path.join(os.path.dirname(__file__), "static")


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    risk_flags: list[dict]
    intent: str | None
    abstained: bool
    refused: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    try:
        return service.system_info()
    except Exception as exc:  # index not built yet
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.query.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    return service.answer_query(req.query, req.session_id)


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))
