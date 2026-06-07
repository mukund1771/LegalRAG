# LegalRAG — Multi-Agent RAG for Legal Contract Analysis

A console-based, multi-turn assistant that answers natural-language questions about a
corpus of legal contracts (NDAs, vendor/service agreements, DPAs) with **grounded,
cited answers** and **risk flags**. Built around the realities of legal RAG: every
claim is traceable to a specific clause, the system **abstains** when the corpus
doesn't contain the answer, and it **refuses** out-of-scope requests (drafting,
legal strategy).

> Full design rationale, research citations, prompt templates and the per-query
> walkthrough live in **[DESIGN.md](./DESIGN.md)**. Diagrams are in **[docs/](./docs/)**.

## Architecture at a glance

Six role-separated agents (each owns a distinct failure mode — no decorative agents):

1. **Orchestrator** — multi-turn loop, conversation memory, history-aware query rewrite.
2. **Planner / Router** — intent classification, scope guardrail, query decomposition, metadata filters.
3. **Retriever** — hybrid (BM25 + dense) → RRF fusion → cross-encoder rerank → parent-child expansion.
4. **Synthesizer** — grounded answer strictly from retrieved evidence, with inline citations.
5. **Risk Assessor** — maps clauses to a risk taxonomy → structured flags (type, severity, party, citation).
6. **Verifier** — pre-gen retrieval grading + corrective loop (CRAG) and post-gen faithfulness check (Self-RAG).

```
User → Orchestrator → Planner → Retriever → Verifier(pre) → Synthesizer → Risk → Verifier(post) → Console
                          └─ out-of-scope → safe refusal      └─ corrective re-retrieval / abstain ─┘
```

See `docs/diagram_2_agent_flow.svg` for the full flow and `docs/diagram_1_ingestion_pipeline.svg` for ingestion.

## RAG choices (summary)

| Decision | Default (local) | Why |
|---|---|---|
| Chunking | structure-aware, parent-child, context-augmented (SAC) | search small/precise child, read large parent; fights document-level mismatch |
| Embeddings | `BAAI/bge-m3` | dense+sparse from one model, 8k ctx, open/private; swap to legal-tuned in prod |
| LLM | `qwen2.5:14b-instruct` (+ `llama3.1:8b` for fast ops) via **Ollama** | reasoning + structured output, runs locally |
| Retrieval | hybrid BM25+dense → RRF → cross-encoder rerank | exact terms + semantics; rank fusion needs no score normalization |
| Determinism | temp 0–0.1, fixed seed, JSON mode, token caps | reproducible, low-verbosity legal answers |

## Setup

```bash
# 1. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local models (Ollama)
ollama pull qwen2.5:14b-instruct
ollama pull llama3.1:8b-instruct
ollama pull bge-m3                # or use sentence-transformers for BGE-M3

# 3. Add contracts
cp your_contracts/*.pdf data/contracts/
```

## Run

```bash
python main.py --ingest    # build the hybrid index from data/contracts
python main.py             # start the interactive console
python main.py --eval      # run the evaluation harness
```

## Project structure

```
legal_rag/
  config/      typed settings + config.yaml
  ingestion/   parser → chunker (parent-child + metadata + SAC) → indexer
  retrieval/   bm25 · dense · fusion(rrf) · rerank · store
  llm/         provider-agnostic client (ollama | vllm | openai)
  agents/      orchestrator · planner · retriever · synthesizer · risk · verifier · prompts/
  memory/      session state + history-aware rewrite
  eval/        gold set (17 sample queries + adversarial) + RAGAS runner
  cli/         interactive console
main.py        entry point
docs/          architecture diagrams
DESIGN.md      full design document
```

## Evaluation

Measured on a gold set (the 17 sample queries + adversarial no-answer cases):
retrieval (context precision/recall, document-level accuracy), generation
(faithfulness, citation correctness, answer relevancy via RAGAS), risk-flag
precision/recall, and **abstention/refusal accuracy**. In legal QA a confident wrong
answer is worse than "I don't know," so abstention and verifiable citation are
first-class metrics. See DESIGN.md §5 for limitations.

## Scaling — from 4 documents to 10k+

The current corpus is tiny (4 docs), so v1 retrieves over a flat index. The design
**anticipates scale**, and the path to 10k+ documents splits cleanly by query type:

**Point / clause queries** (Q1–7, 13, 14) scale almost for free: swap the dev store
(FAISS/Chroma) for an ANN vector DB with HNSW + payload filtering
(**Qdrant / pgvector / Weaviate**), shard, and add a **document-level routing**
stage — first narrow to candidate documents via metadata + doc-level summary
embeddings, *then* run clause-level hybrid retrieval. This keeps the reranker's
candidate pool small and latency flat as the corpus grows.

**Corpus-wide analytical queries are the real scaling challenge** (Q9 "conflicting
governing laws *across agreements*", Q10/Q12 "any unlimited liability", Q15
"summarize *all* risks"). At 4 docs these can brute-force iterate every document; at
10k that is impossible and unreliable — you cannot stuff 10k clauses into a context
window. The fix is to **move this work from query time to ingestion time** with a
**structured knowledge layer**:

- At ingestion, extract key fields into a queryable schema per document —
  `governing_law`, `liability_cap`, `breach_notice_window`, `indemnification_type`,
  `subprocessor_sharing`, plus **pre-computed risk flags**. Think "contract data
  warehouse" alongside the vector index.
- Aggregate/analytical questions then become **metadata aggregation** (a `GROUP BY`
  over a structured table), and the LLM only reasons over the small aggregated
  result set — not thousands of raw clauses. "Conflicting governing laws" becomes
  `SELECT governing_law, COUNT(*) ... GROUP BY` then a short LLM judgment; "summarize
  all risks for Acme" reads a precomputed risk index filtered by party.

This is the mature takeaway: **pure retrieval-time RAG does not scale to corpus-wide
analytics** — large-scale legal analysis needs *RAG (point lookup) + a structured
extraction layer (aggregation)*, with document-level routing in front of both.

Other production concerns at scale (detailed in DESIGN.md §9): async/incremental/
idempotent ingestion with dedup; embedding + context-augmentation cost management;
**vLLM** serving (continuous batching, ~6–20× throughput) behind the same API;
semantic + exact response caching; observability + online RAGAS + drift monitoring;
multi-tenant **RBAC / entitlement filtering** (10k docs implies many matters/clients);
and human-in-the-loop review for high-severity risk flags.

## Known limitations

No licensed-attorney validation — output is decision-support, **not legal advice**.
Quality is bounded by text-extraction accuracy on **digital-text** PDFs (scanned/image-only docs and OCR are out of scope for v1). The risk taxonomy is finite.
Evaluation is directional on a tiny corpus. See DESIGN.md §8.
