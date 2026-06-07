# Multi-Agent RAG System for Legal Contract Analysis — System Design

> **Status:** Design document (pre-implementation). This defines the architecture, the RAG choices and their justifications, agent responsibilities, prompt strategy, evaluation plan, trade-offs, and the path to production. Code follows once this is agreed.

---

## 1. Problem Statement & Goals

Build a **console-based, multi-turn** assistant that lets a user ask natural-language questions about a small corpus of legal contracts (NDAs, vendor/service agreements, DPAs) and returns answers that are:

- **Grounded** — every claim traceable to a specific clause in a specific document.
- **Precise & well-cited** — show the referenced sections, not a vague summary.
- **Risk-aware** — surface risk flags (e.g. uncapped liability, short breach-notification windows) where applicable.
- **Honest about limits** — abstain when the corpus doesn't contain the answer, and refuse out-of-scope requests (drafting, legal strategy) with a clear disclaimer.

### Why legal RAG is *not* a generic RAG problem

Three domain facts shape every decision below:

1. **Document-faithful retrieval is the core reliability metric.** Pulling a clause from a *similar but different* contract silently produces a wrong-but-plausible answer. The dominant failure mode in legal RAG is **document-level retrieval mismatch** — retrieving the right *kind* of clause from the *wrong* agreement. ([Towards Reliable Retrieval in RAG for Large Legal Datasets](https://arxiv.org/html/2510.06999v1))
2. **Hallucination is dangerous and common.** A Stanford HAI study found even purpose-built legal AI tools hallucinate on **17–34%** of queries; general LLMs reach **69–88%** on legal questions. Lawyers have been sanctioned for citing fabricated authority. The mitigation that works is the combination of *RAG + structured prompting + post-hoc verification + character-level citation*, never a single guardrail. ([Stanford via Spellbook](https://www.spellbook.legal/learn/ai-hallucination-in-law), [Kallam](https://www.kallam.ai/blog/legal-ai-trust-reducing-hallucinations))
3. **A confident wrong answer is worse than "I don't know."** In legal QA, *abstention* and *verifiable citation* are first-class success criteria, not nice-to-haves.

### The query taxonomy (derived from the 17 sample queries)

The sample set was clearly crafted to stress different capabilities. Clustering them reveals the capabilities the architecture must support:

| # | Sample query | Query class | Capability stressed |
|---|---|---|---|
| 1,2,3 | notice period / uptime / governing law | **Single-clause fact lookup** | Precise retrieval + extraction |
| 4,5 | confidentiality survives termination? liability capped for confidentiality breach? | **Clause interpretation (yes/no + reasoning)** | Retrieval + grounded reasoning |
| 6,14 | remedies if SLA missed; what if breach notice > 72h | **Conditional / consequence reasoning** | Multi-clause stitching |
| 8,9 | which agreement governs breach timelines; conflicting governing laws | **Cross-document comparison** | Decomposition + per-doc retrieval + synthesis |
| 7,10,11,12,13 | liability cap for data breach; legal/financial risk; unlimited liability; subcontractor data sharing | **Risk analysis** | Risk taxonomy mapping + severity |
| 15 | summarize all risks in one paragraph | **Aggregation / summarization** | Corpus-wide gather + compress |
| 16,17 | draft a better NDA; what legal strategy | **Out-of-scope (drafting / advice)** | Guardrail + safe refusal |

This taxonomy is what justifies a **planner/router** and a **dedicated risk agent** — not aesthetics. Roughly a third of the benchmark is risk-centric, and two queries are deliberate traps that a mature system must *decline*.

---

## 2. Architecture Overview

The system is an **agentic RAG pipeline**: instead of a fixed "retrieve → generate" line, control flows through specialized agents that plan, retrieve, synthesize, assess risk, and *verify*, with a corrective loop when retrieval is weak. This mirrors the **decomposer → retriever → decider/critic** pattern that recurs across the agentic-RAG literature. ([Agentic RAG: A Survey, arXiv:2501.09136](https://arxiv.org/abs/2501.09136))

### Design principle: agents are *roles*, not headcount

> **Avoiding over-agentification.** Each agent below owns a distinct, testable responsibility and a distinct failure mode. Several share the same underlying LLM with different system prompts and tools — an "agent" here is a *(role + prompt + tools + I/O contract)*, not necessarily a separate model or process. I explicitly chose **not** to split, e.g., "citation agent," "formatting agent," or "coreference agent" into their own nodes, because they have no independent decision to make.

### The six agents

| Agent | Responsibility | Distinct failure it owns | LLM |
|---|---|---|---|
| **1. Orchestrator** | Owns the multi-turn loop, conversation memory/session state, history-aware query rewrite, calls agents, assembles console output. | Lost context across turns; broken coreference. | Small (8B) |
| **2. Planner / Router** | Classifies intent (the taxonomy above), enforces the scope guardrail, decomposes complex queries into sub-queries, emits a retrieval plan (target docs, metadata filters). | Wrong routing; missed decomposition of cross-doc queries. | Medium (14B) |
| **3. Retriever** | Executes hybrid (BM25 + dense) retrieval → RRF fusion → cross-encoder rerank → parent-child expansion, with metadata filtering. Runs sub-queries in parallel. | Document-level retrieval mismatch; low recall. | none (search) + reranker |
| **4. Synthesizer** | Generates the grounded answer **strictly** from retrieved evidence, with inline citations and controlled verbosity. | Ungrounded / over-confident generation. | Medium (14B) |
| **5. Risk Assessor** | Maps retrieved clauses to a risk taxonomy → structured flags (type, severity, party, rationale, citation). Invoked for risk/summary intents or when synthesis surfaces risk-bearing clauses. | Missed or fabricated risk; wrong severity. | Medium (14B) |
| **6. Verifier / Critic** | (a) **Pre-gen:** grades retrieval sufficiency (CRAG evaluator) and can trigger a corrective re-retrieval; (b) **Post-gen:** checks every answer claim is supported by a cited chunk (Self-RAG faithfulness), strips/blocks unsupported claims. | Hallucination slips through. | Medium (14B) |

The **scope guardrail** (handling queries 16/17 and the "not a lawyer" disclaimer) is a **cross-cutting policy enforced by the Planner and Synthesizer**, deliberately *not* its own agent — it's a rule, not a reasoner.

This decomposer (Planner) → parallel retriever → synthesizer → critic (Verifier) shape, with a specialized analyst (Risk), is exactly the centralized "manager + role-specialized workers" pattern the survey identifies as the workhorse multi-agent RAG topology. ([Agentic RAG Survey](https://arxiv.org/html/2501.09136v4))

### Control flow (happy path + corrective loop)

```
User turn
  │
  ▼
[Orchestrator]  ── load history, resolve coreference → standalone query
  │
  ▼
[Planner/Router]  ── intent? scope-ok? → sub-queries + retrieval plan
  │            └─(out-of-scope: drafting/advice)──► Safe refusal + disclaimer ─► User
  ▼
[Retriever]  ── hybrid search → RRF → rerank → parent expansion (per sub-query, parallel)
  │
  ▼
[Verifier — pre-gen / CRAG]  ── evidence sufficient?
  │      ├─ NO  ──► corrective re-retrieval (relax filters / rewrite / broaden)  ──┐
  │      │                                                                          │
  │      └─ still NO after N tries ──► "Insufficient evidence in the corpus" ─► User
  ▼                                                                                 │
[Synthesizer]  ── grounded answer + inline citations  ◄──────────────────────────-┘
  │
  ▼
[Risk Assessor]  ── (if risk/summary intent) clause → risk flags
  │
  ▼
[Verifier — post-gen / Self-RAG]  ── every claim supported by a citation?
  │      └─ unsupported claims ──► strip / regenerate / downgrade to abstention
  ▼
[Orchestrator]  ── assemble: Answer | Citations | Risk flags  ─► console + update memory
```

---

## 3. RAG Design (Required Choices & Justifications)

### 3a. Document Chunking Strategy — *structure-aware, parent-child, context-augmented*

**Choice:** Parse each contract into its native hierarchy (Document → Article/Section → Clause → sub-clause) and chunk **clause-aware** with a **parent-child** index, plus **per-chunk context augmentation**.

- **Child chunk** = a single clause / sub-clause (small, ~256–512 tokens) → this is what we *embed and search*. Small chunks give precise matches.
- **Parent chunk** = the full enclosing section (up to ~2k tokens) → this is what we *return to the LLM*. Clauses derive meaning from their section (e.g. a penalty figure means nothing without the obligation it attaches to). On retrieval we match the child but hand the synthesizer the parent. ([Parent-Child Chunking](https://www.sandgarden.com/learn/parent-child-chunking))
- **Why not fixed-size chunking:** contracts have *real* structure (numbered clauses, headings). A header-aware splitter preserves boundaries that fixed-size windows destroy, and it stops a single clause being split mid-sentence across two chunks. ([Chunking strategies for RAG](https://sureprompts.com/blog/chunking-strategies-for-rag))
- **Context augmentation (Summary-Augmented Chunking / contextual retrieval):** prepend a short, LLM-generated one-line context to each child chunk's *embedding text* — e.g. *"[NDA between Acme Corp and Vendor XYZ, Section 7 — Term & Termination]"*. This directly attacks **document-level retrieval mismatch**, the #1 legal-RAG failure: it disambiguates which agreement and section a clause belongs to. ([SAC, arXiv:2510.06999](https://arxiv.org/html/2510.06999v1))

**Metadata attached to every chunk** (the backbone of precise retrieval and citation):

```
doc_id, doc_type (NDA|MSA|SLA|DPA|Vendor), parties[], effective_date,
section_no, section_heading, clause_type (termination|confidentiality|
indemnification|liability|governing_law|data_breach|sla_uptime|...),
char_start, char_end, parent_id
```

`char_start/char_end` enable **character-level citation** — the single biggest factor in making hallucination *verifiable rather than invisible* in contract review. ([GC AI](https://gc.ai/blog/ai-contract-review))

### 3b. Embedding Model — *local-first, legal-aware, hybrid-capable*

**Default (local / Ollama-aligned): `BGE-M3`.** Rationale:

- Produces **dense + sparse (lexical)** vectors from one model → powers hybrid search without a second system.
- **8k context** comfortably holds long clauses/parents.
- Open weights → contract data never leaves the machine (privacy is non-negotiable for legal).
- Runs locally via SentenceTransformers/FastEmbed (or `ollama pull bge-m3`).

**Key finding that drives the production swap:** *general-purpose embedding quality ≠ legal retrieval quality.* On the **Massive Legal Embedding Benchmark (MLEB)**, rankings reshuffle vs MTEB — Gemini Embedding is #1 on MTEB but only #7 on legal; the top legal performers (Kanon 2, Voyage-3-large, Voyage-3.5, voyage-law-2) are all **domain-adapted**. ([MLEB, arXiv:2510.19365](https://arxiv.org/html/2510.19365v1); [voyage-law-2](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/))

→ **Decision:** the embedding model sits behind an interface; **default `BGE-M3` locally**, config-swappable to **`voyage-law-2` / `voyage-3-large`** (or a fine-tuned in-house model) in production for the legal-retrieval accuracy gain.

### 3c. LLM Choice (Ollama) + Prompting + Determinism

**Models (via Ollama):** a two-tier setup balances quality and latency.

| Role | Model | Why |
|---|---|---|
| Reasoning agents (Planner, Synthesizer, Risk, Verifier) | **`qwen2.5:14b-instruct`** | Strong instruction-following + structured/JSON output + solid reasoning; good faithfulness at a size that runs on a single workstation GPU. |
| Cheap/fast ops (query rewrite, intent classification) | **`llama3.1:8b-instruct`** | Low latency for short, well-bounded tasks. |
| Re-ranker | **`bge-reranker-v2-m3`** (cross-encoder) | Not a chat LLM; see 3e. |

**Prompting strategy per agent** (full templates in §4). Common spine for every grounded agent:
1. **Role + scope** ("You analyze provided contract excerpts. You are not a lawyer.").
2. **Hard grounding rule** — "Answer **only** from the CONTEXT. If the answer is not in the context, say *'Not found in the provided contracts.'* Never use outside knowledge."
3. **Mandatory citation** — every sentence must cite `[doc_id §section]`.
4. **Structured output** — JSON where downstream agents consume it (Planner, Risk, Verifier); clean prose for the final user answer.
5. **Reasoning scaffold** for interpretation queries — lightweight **IRAC** (Issue → Rule(cited clause) → Application → Conclusion), which is associated with lower hallucination in legal drafting. ([Legal AI comparison](https://thelegalprompts.com/blog/claude-vs-gemini-lawyers-legal-work))

**Determinism & verbosity control (per-agent):**

| Agent | temperature | output | max tokens | notes |
|---|---|---|---|---|
| Planner / Router | 0.0 | JSON (`format=json`) | 256 | deterministic routing |
| Retriever | n/a | — | — | no generation |
| Synthesizer | 0.1 | prose + citations | 512 | tight, factual |
| Risk Assessor | 0.0 | JSON flags | 512 | deterministic taxonomy |
| Verifier | 0.0 | JSON verdict | 256 | strict grading |
| Query rewrite | 0.0 | text | 128 | — |

Plus: **fixed seed**, **low top_p (≈0.3)** on factual agents, **JSON-mode / grammar-constrained decoding** for structured agents (eliminates parse failures), and **token caps** to bound verbosity. Higher temperature is allowed *only* for the final phrasing of summaries (query 15), and even then over already-grounded content.

**Production note:** Ollama is the right call for **dev / single-user** (~62 tok/s on Llama-3.1-8B, sequential request handling). For multi-user production, the same OpenAI-compatible API is served by **vLLM** (PagedAttention + continuous batching → ~6–20× aggregate throughput under concurrency). Because all agents call a thin **provider-agnostic LLM interface**, moving Ollama → vLLM (or OpenAI/Bedrock) is a config change, not a code change. ([Red Hat: Ollama vs vLLM](https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking))

### 3d. Retrieval Mechanism — *hybrid + RRF + metadata filtering*

A three-stage pipeline (the production-standard shape):

1. **Hybrid candidate generation.** Run **BM25** (sparse) and **dense** (BGE-M3) in parallel.
   - BM25 nails exact tokens that matter enormously in contracts — *"72 hours," "Section 7.2," party names, "99.9%"* — which dense search can under-weight.
   - Dense nails paraphrase — *"can they share data with subcontractors"* ↔ *"disclosure to third-party processors."*
   - ([Hybrid search reference](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026))
2. **Fusion via Reciprocal Rank Fusion (RRF).** RRF merges the two ranked lists *by rank, not score* — no score normalization needed across incompatible BM25/cosine scales. It reliably beats either retriever alone on NDCG. ([RRF](https://avchauzov.github.io/blog/2025/hybrid-retrieval-rrf-rank-fusion/))
3. **Cross-encoder rerank** (see 3e) → final top-k → **parent-child expansion** (return parents of the winning children).

**Metadata pre-filtering** is the legal-specific lever. The Planner can constrain retrieval: `doc_type=NDA`, or `clause_type=governing_law GROUP BY doc_id`. This is how cross-document queries work:

- *Q9 "conflicting governing laws"* → filter `clause_type=governing_law`, retrieve the clause **from each document**, then compare in synthesis/risk.
- *Q8 "which agreement governs breach notification timelines"* → filter `clause_type=data_breach`, return per-doc, identify the controlling one.

**Vector store:** in-memory / **FAISS** or **Chroma** for the dev corpus; **Qdrant / pgvector / Weaviate** in production (native hybrid + metadata filtering + persistence).

### 3e. Re-ranking (bonus)

After RRF produces ~32 finalists, a **cross-encoder (`bge-reranker-v2-m3`)** scores each `(query, chunk)` pair jointly and selects the final ~5–8. Cross-encoders are meaningfully more accurate than bi-encoder similarity at the cost of latency, which is acceptable on a small finalist set. ([Cross-encoder reranking](https://appscale.blog/en/blog/hybrid-search-and-reranking-production-rag-bm25-dense-cross-encoder-2026))

> **Design alternative considered:** *Ranking-Free RAG* (replace the reranker with an LLM **selection** step) is proposed specifically for sensitive domains and is worth A/B-testing, but a deterministic cross-encoder is cheaper and more reproducible for v1. ([Ranking-Free RAG, arXiv:2505.16014](https://arxiv.org/pdf/2505.16014))

### 3f. Verification & the corrective loop (the anti-hallucination spine)

This is where legal RAG earns trust, so it gets its own subsystem rather than being folded into generation.

- **Pre-generation (CRAG retrieval evaluator):** grade whether the retrieved evidence actually answers the query (confidence: *correct / ambiguous / incorrect*). On low confidence, take a **corrective action** — rewrite/broaden the query, relax metadata filters, increase k — for up to *N* tries, then **abstain**. (In a production system with a trusted external legal KB, "incorrect" could trigger an authorized external lookup; we do **not** fall back to open web for legal facts.) ([CRAG, arXiv:2401.15884](https://arxiv.org/html/2401.15884v3))
- **Post-generation (Self-RAG faithfulness):** decompose the draft answer into atomic claims and check each is entailed by a cited chunk. Unsupported claims are stripped or trigger a regenerate; if the core claim is unsupported, the answer is **downgraded to abstention**. ([Self-RAG](https://arxiv.org/html/2401.15884v3))

This directly implements the "RAG + structured prompting + post-hoc verification + character-level citation" stack the legal-AI literature says is required to push hallucination toward zero.

---

## 4. Prompt Design

Prompts are versioned templates with explicit I/O contracts. Abridged examples:

**Planner / Router (JSON, temp 0):**
```
You route questions about a fixed corpus of legal contracts. You are NOT a lawyer.
Classify the user's standalone question and produce a retrieval plan.

Return JSON:
{
  "intent": "single_fact | interpretation | conditional | cross_doc_compare |
             risk_analysis | summary | out_of_scope_drafting | out_of_scope_advice",
  "in_scope": true|false,
  "sub_queries": ["..."],            // decompose cross-doc/multi-part questions
  "filters": {"doc_type": "...", "clause_type": "...", "party": "..."},
  "needs_risk_agent": true|false
}
Rules:
- Drafting new documents or giving legal strategy/advice => out_of_scope_*, in_scope=false.
- "conflicting/across agreements" => cross_doc_compare; one sub_query per relevant clause_type per doc.
```

**Synthesizer (prose + citations, temp 0.1):**
```
You answer ONLY from the CONTEXT below (retrieved contract excerpts). You are not a lawyer.
RULES:
1. Use only the CONTEXT. If the answer is not present, reply exactly:
   "Not found in the provided contracts."
2. Cite every factual sentence as [doc_id §section].
3. For yes/no interpretation, use IRAC briefly: state the rule (quote the clause),
   apply it, then conclude.
4. Be concise. No outside knowledge, no speculation.

CONTEXT:
{retrieved_parents_with_metadata}

QUESTION: {standalone_query}
```

**Risk Assessor (JSON flags, temp 0):**
```
Identify legal/financial risks ONLY from the provided clauses. You are not a lawyer.
For each risk return:
{ "risk_type": "uncapped_liability | cap_excludes_confidentiality |
                 breach_notice_window_too_long | subcontractor_data_sharing |
                 one_sided_indemnification | auto_renewal | governing_law_conflict |
                 missing_data_breach_clause",
  "severity": "high|medium|low",
  "affected_party": "...",
  "rationale": "...",
  "citation": "[doc_id §section]" }
If a clause is favorable or neutral, do not invent a risk. Empty list is valid.
```

**Verifier — post-gen (JSON, temp 0):**
```
Given an ANSWER and the CONTEXT it cited, check each claim.
Return {"supported": [...], "unsupported": [...], "verdict": "pass|revise|abstain"}.
A claim is supported only if a cited chunk explicitly entails it. Quotation > paraphrase.
```

**Out-of-scope response (queries 16, 17):** scoped, useful refusal — decline the generative/advisory ask, then offer the in-scope alternative, with disclaimer. E.g. for *"draft a better NDA"*: "I analyze the contracts you've provided; I don't draft new agreements or give legal advice. I **can** point out weak or risky clauses in your current NDA with citations — want that?"

---

## 5. Evaluation

**What we build:** a small **gold set** = the 17 sample queries, each annotated with (expected answer, the correct source clause(s), expected risk flags) — **plus** adversarial additions: (a) questions whose answer is *absent* from the corpus (must abstain), and (b) cross-doc traps. The harness runs per-turn and aggregates.

**What we measure & why it matters:**

| Layer | Metric | Why it matters here |
|---|---|---|
| **Retrieval** | Context precision, context recall, hit-rate/MRR@k, **document-level retrieval accuracy** | Catches the #1 legal failure — right clause type, wrong contract. ([RAGAS](https://docs.ragas.io/en/v0.1.21/concepts/metrics/)) |
| **Generation** | **Faithfulness/groundedness**, answer relevancy, **citation correctness** (do cited spans actually contain the claim), answer correctness vs gold | Faithfulness + verifiable citation are the metrics that map to real legal harm. |
| **Risk** | Precision/recall of risk flags vs annotated flags; severity agreement | Risk flagging is a first-class requirement (1/3 of queries). |
| **Safety** | **Abstention accuracy** (correctly says "not found"), **refusal accuracy** on out-of-scope (16/17) | A confident wrong answer is the worst outcome; abstention is success. |

**How:** **RAGAS** (LLM-as-judge) for faithfulness/relevancy/context metrics (no human labels needed for those); **deterministic exact/regex checks** for hard facts ("72 hours," governing-law state, "99.9%"); **LLM-judge with a rubric** for risk/severity; and the small hand-labeled gold set for correctness and citation checks. Run as a CI gate so regressions are caught.

**Limitations (stated honestly):**
- **Tiny corpus / 17 queries → high variance**; results are directional, not statistically significant.
- **LLM-as-judge is itself imperfect** — non-deterministic, biased toward verbose/own-style answers, and adds cost; we pin judge model + temperature and spot-check against human labels.
- **Faithfulness ≠ legal correctness.** A perfectly grounded answer can still be legally naive; we have **no licensed-attorney ground truth.**
- Doesn't yet cover **adversarial paraphrase robustness** or **long-context degradation** at scale.

---

## 6. Code Structure (planned)

```
legal_rag/
├── config/            # pydantic-settings + YAML; models, k, temps, paths
├── ingestion/         # parse → structure-aware chunk → context-augment → embed → index
│   ├── parser.py
│   ├── chunker.py     # clause/parent-child + metadata
│   └── indexer.py
├── retrieval/         # bm25.py, dense.py, fusion(rrf).py, rerank.py, store.py
├── llm/               # provider-agnostic client: ollama | vllm | openai
├── agents/            # orchestrator, planner, retriever, synthesizer, risk, verifier
│   └── prompts/       # versioned prompt templates
├── memory/            # session state + history-aware rewrite + summarization
├── eval/              # gold set, ragas runner, metric reports
└── cli/               # interactive console (multi-turn loop, rendering)
```
Clean separation of concern, typed config, docstrings, an `LLMClient` seam for provider swap, and a `Retriever` seam for store swap.

---

## 7. Key Trade-offs (explicit)

- **More agents vs. simplicity** → chose 6 role-separated agents, each owning a distinct failure mode; refused decorative agents (citation/format/coreference) that make no decision.
- **Local privacy vs. peak accuracy** → default fully local (BGE-M3 + Qwen-2.5-14B); contract data never leaves the box. Production can swap to legal-tuned embeddings/larger models for accuracy, accepting a data-governance review.
- **Latency vs. faithfulness** → the verifier/corrective loop adds LLM calls and latency; justified because in legal, a wrong answer is far costlier than a slow one. Loop is bounded (N tries) and skippable for low-stakes single-fact queries via config.
- **Small precise chunks vs. context** → resolved by parent-child (search small, read big).
- **Ollama vs. vLLM** → Ollama for dev ergonomics now; vLLM behind the same API for production throughput.
- **Cross-encoder rerank vs. LLM selection** → deterministic cross-encoder for v1 reproducibility; LLM-selection (Ranking-Free RAG) flagged as an A/B for later.

---

## 8. Known Limitations

- No licensed-attorney validation; output is decision-support, **not legal advice** (enforced disclaimer).
- Quality is bounded by parser accuracy on messy/scanned PDFs (OCR + layout errors propagate). Scanned contracts need an OCR pre-stage.
- Risk taxonomy is curated and finite; novel risk patterns outside it won't be flagged.
- Cross-document reasoning depends on correct `clause_type` tagging at ingestion; mis-tagging hides conflicts.
- Evaluation is directional on this corpus size (see §5).

---

## 9. Production Scaling (path beyond the console)

| Concern | Dev (this assignment) | Production |
|---|---|---|
| LLM serving | Ollama, single user | **vLLM** (continuous batching, PagedAttention), autoscaled GPU pool, same OpenAI-compatible API |
| Embeddings | BGE-M3 local | Legal-tuned (voyage-law-2 / fine-tuned), GPU batch endpoint |
| Vector store | FAISS / Chroma | Qdrant / pgvector / Weaviate — hybrid + metadata filters + HA |
| Ingestion | one-shot script | async, incremental, versioned; OCR + layout parsing; per-doc-type pipelines |
| Agents | in-process | stateless services + task queue; horizontal scale |
| Caching | none | semantic + exact response cache |
| Observability | logs | tracing (Langfuse), **online RAGAS**, drift monitoring |
| Trust/governance | disclaimer | **human-in-the-loop review for high-severity flags**, RBAC, PII handling, audit log, eval CI gate |

---

## Appendix A — Walkthrough of all 17 sample queries

Tracing each benchmark query through the design confirms the architecture (not just hand-waving). `intent` is the Planner label; the rest is how the query is served.

| # | Query | Intent | How the design serves it |
|---|---|---|---|
| 1 | NDA termination notice period | single_fact | filter `doc_type=NDA, clause_type=termination`; BM25 catches "notice"/days; synthesizer extracts the period + cite. |
| 2 | SLA uptime commitment | single_fact | filter `clause_type=sla_uptime`; BM25 nails "99.x%"; extract + cite. |
| 3 | Governing law of Vendor Services Agreement | single_fact | filter `doc_type=Vendor, clause_type=governing_law`; return jurisdiction + cite. |
| 4 | Do confidentiality obligations survive NDA termination? | interpretation | retrieve confidentiality + survival clauses; IRAC yes/no with quoted survival language. |
| 5 | Is liability capped for breach of confidentiality? | interpretation | retrieve liability cap **and** its carve-outs; key check: does the cap *exclude* confidentiality? cite both. |
| 6 | Remedies if SLA uptime not met | conditional | stitch SLA + remedies/service-credit clauses; enumerate remedies + cite. |
| 7 | Is Vendor XYZ's liability capped for data breaches? | risk_analysis | retrieve liability + data_breach clauses; Risk agent checks cap vs data-breach carve-out → flag if uncapped. |
| 8 | Which agreement governs breach-notification timelines? | cross_doc_compare | filter `clause_type=data_breach` across docs; identify the controlling doc + cite. |
| 9 | Conflicting governing laws across agreements? | cross_doc_compare | per-doc `governing_law` retrieval; compare jurisdictions; Risk flag `governing_law_conflict` if mismatched. |
| 10 | Legal risks from liability exposure | risk_analysis | gather liability/indemnification clauses corpus-wide; Risk taxonomy → flags w/ severity + cite. |
| 11 | Clauses posing financial risk to Acme Corp | risk_analysis | metadata `party=Acme`; Risk agent scores uncapped liability, one-sided indemnity, penalties. |
| 12 | Any unlimited liability? | risk_analysis | search liability caps; flag any clause with no cap / explicit "unlimited" → high severity. |
| 13 | Can Vendor XYZ share Acme's data with subcontractors? | interpretation/risk | retrieve subprocessor/disclosure clauses; answer + `subcontractor_data_sharing` flag if permitted without consent. |
| 14 | What if breach notice > 72h? | conditional | retrieve breach-notification + remedy/penalty clauses; reason consequence; flag if window long/unspecified. |
| 15 | Summarize all risks for Acme in one paragraph | summary | fan-out retrieval across risk clause types → Risk agent aggregates → synthesizer compresses to one cited paragraph. |
| 16 | Draft a better NDA | out_of_scope_drafting | guardrail off-ramp: decline drafting, offer clause-weakness analysis instead + disclaimer. |
| 17 | What legal strategy vs Vendor XYZ? | out_of_scope_advice | guardrail off-ramp: decline advice; offer factual risk summary + "consult counsel" disclaimer. |

Plus the eval-only adversarial cases (answer absent from corpus) exercise the **abstention** path through the pre-gen Verifier.

## Sources

- Towards Reliable Retrieval in RAG for Large Legal Datasets (SAC, document-level mismatch) — https://arxiv.org/html/2510.06999v1
- Agentic Retrieval-Augmented Generation: A Survey — https://arxiv.org/abs/2501.09136 · https://arxiv.org/html/2501.09136v4
- CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review — https://ar5iv.labs.arxiv.org/html/2103.06268
- The Massive Legal Embedding Benchmark (MLEB) — https://arxiv.org/html/2510.19365v1
- voyage-law-2 (domain-specific legal embeddings) — https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/
- Parent-Child Chunking — https://www.sandgarden.com/learn/parent-child-chunking
- Chunking Strategies for RAG — https://sureprompts.com/blog/chunking-strategies-for-rag
- Hybrid Search (BM25 + Vector + Reranking) reference — https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026
- Hybrid retrieval with Reciprocal Rank Fusion — https://avchauzov.github.io/blog/2025/hybrid-retrieval-rrf-rank-fusion/
- Hybrid Search & Reranking in Production RAG — https://appscale.blog/en/blog/hybrid-search-and-reranking-production-rag-bm25-dense-cross-encoder-2026
- Ranking-Free RAG (selection vs rerank for sensitive domains) — https://arxiv.org/pdf/2505.16014
- Corrective RAG (CRAG) — https://arxiv.org/html/2401.15884v3
- RAGAS metrics — https://docs.ragas.io/en/v0.1.21/concepts/metrics/
- AI hallucination in law (Stanford 17–34%) — https://www.spellbook.legal/learn/ai-hallucination-in-law
- Reducing hallucinations in legal AI — https://www.kallam.ai/blog/legal-ai-trust-reducing-hallucinations
- AI Contract Review (character-level citation, entity grounding) — https://gc.ai/blog/ai-contract-review
- Ollama vs vLLM benchmarking — https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking
