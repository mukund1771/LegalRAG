# Approach — LegalRAG

How this system was designed, built, evaluated, and hardened. This is the narrative;
**[docs/DESIGN.md](./docs/DESIGN.md)** is the full design, **[docs/DESIGN_RATIONALE.md](./docs/DESIGN_RATIONALE.md)**
maps every decision to the brief, and **[findings.md](./findings.md)** is the evaluation log.

## 1. The problem, framed honestly

Build a multi-agent RAG system that answers natural-language questions over a small
corpus of contracts (NDA, vendor/service agreement, SLA, DPA) with grounded, cited
answers and risk flags. The non-obvious part is that **legal RAG is not generic RAG**:

1. The dominant failure is **document-level mismatch** — retrieving the right *kind* of
   clause from the *wrong* contract, which yields a wrong-but-plausible answer.
2. **Hallucination is dangerous and common** in legal tools; the only mitigation that
   works is *RAG + structured prompting + post-hoc verification + character-level
   citation*, never a single guardrail.
3. **A confident wrong answer is worse than "I don't know."** Abstention and verifiable
   citation are first-class success criteria, not nice-to-haves.

Every design decision below follows from these three facts.

## 2. Design decisions (and why)

- **Six agents, each owning a distinct failure mode** — Orchestrator (memory /
  coreference), Planner (intent + scope guardrail + filters), Retriever, Synthesizer,
  Risk Assessor, Verifier (pre-gen CRAG grading + post-gen Self-RAG faithfulness). We
  deliberately *rejected* decorative agents (citation/format) that make no independent
  decision — over-agentification was an explicit anti-goal.
- **Chunking: structure-aware parent-child + context cue (SAC).** Search the small,
  precise *child* clause; hand the LLM the larger *parent* section. Each child's
  embedding text is prefixed with a doc/section cue (e.g. `[NDA | §3 Term & Termination
  | termination]`) to fight document-level mismatch. This also makes character-level
  citation possible.
- **Hybrid retrieval + RRF + cross-encoder.** BM25 catches exact tokens ("72 hours",
  party names); dense (`bge-m3`) catches paraphrase ("subcontractors" ≈ "subprocessor").
  Reciprocal Rank Fusion merges them by rank (no score normalization). A cross-encoder
  reranks the finalists for top-rank precision.
- **Metadata filters as the cross-doc lever.** "Conflicting governing laws across
  agreements" = filter `clause_type=governing_law`, retrieve one per document, compare.
- **Verification as a first-class subsystem.** Pre-gen grading can trigger corrective
  re-retrieval or abstention; post-gen faithfulness downgrades unsupported answers. This
  is what earns the words "grounded" and "well-cited."
- **Determinism & verbosity control.** temp 0–0.1, fixed seed, JSON mode for structured
  agents, token caps. Legal answers must be reproducible and conservative.
- **Provider-agnostic seams.** One `LLMClient` (Ollama / fake), one `Embedder`
  interface, one `VectorStore` — so Ollama→vLLM and FAISS→Qdrant are config changes.

## 3. How it was built

Six milestones, each a runnable, tested slice shipped as its own PR:
M1 ingestion → M2 retrieval core → M3 answer path → M4 agent graph + control →
M5 risk assessor → M6 evaluation harness. Then: real-corpus integration, retrieval
metrics + ablation, two evaluation-driven fixes, a FastAPI web app, and a RunPod deploy.

A deliberate engineering choice runs through all of it: a **deterministic offline path**
(`FakeEmbedder` + lexical reranker + `FakeLLM` + heuristic planner/verifier) so the
entire pipeline — parse → chunk → retrieve → route → answer → risk → eval — is unit-
testable with no GPU, no downloads, no network. **41 tests** run in well under a second.

## 4. Evaluation — and what it taught us

We don't trust vibes. `eval/` has labeled **`qrels`** (the gold clause per query),
pure-function ranking metrics, a gold set (17 sample queries + adversarial no-answer
cases), and an ablation runner.

The eval loop changed the system, which is the point:

- **Run 1 (bge-m3 + lexical)** came out *tied with the baseline*. Diagnosis: the
  **lexical reranker was the bottleneck** — it re-sorts by token overlap and undoes
  dense retrieval's semantic win (Q13 "subcontractors" → "subprocessor" stayed at
  recall 0). The eval also exposed two real bugs: the planner mis-routed a
  liability-for-data-breach question, and the verifier graded retrieval on lexical
  overlap (false abstentions on dense matches).
- **Fixes** (both validated): route "liability … data breach" → risk_analysis, and make
  the verifier's grade reranker-agnostic (a confident cross-encoder score counts as
  relevant). Offline routing went 0.947 → **1.0**; abstention and risk-recall improved.
- **Run 2 (bge-m3 + cross-encoder)** finally beat the baseline where it matters —
  **recall@1 0.624 and the best MRR (0.867)** — confirming the thesis: *the embedder
  only pays off once the reranker can exploit it.*

We also re-diagnosed an apparent failure honestly: abstaining on "what happens if breach
notice is delayed >72h" is **correct** for this corpus, because the real DPA has no
consequences clause. We changed the expectation, not the behavior.

## 5. From demo to product

The system is decision-support today. The path to production is documented, not hand-
waved (DESIGN §9): vLLM behind the same API for throughput; an ANN vector DB with
document-level routing; a **structured knowledge layer** built at ingestion so
corpus-wide analytical queries become metadata aggregation rather than stuffing
thousands of clauses into context; incremental ingestion; caching; multi-tenant RBAC;
and human-in-the-loop review for high-severity risk flags. A CI threshold gate on the
eval metrics (routing ≥ 0.95, refusal = 1.0, recall@1 ≥ 0.55, MRR ≥ 0.80) turns the
evaluation harness into a regression guard.

## 6. Honest limitations

No licensed-attorney validation — **not legal advice**. Digital-text documents only (no
OCR in v1). The clause tagger is English-keyword; the risk taxonomy is curated and
finite, so jurisdiction-specific risks (e.g. Indian-law statutory specifics) aren't
auto-flagged. Evaluation is directional on a 4-document corpus. The system's discipline
— grounding, citation, abstention, refusal — is what makes it trustworthy *within* those
bounds.
