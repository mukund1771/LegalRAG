# Design Rationale — Why Every Decision, Tied to the Problem Statement

This document does one thing: it **re-reads the original assignment**, confirms the
design covers every stated requirement, and gives the **reasoning and benefit behind
each decision** — including the alternatives we rejected and *why*. Where a decision
exists because of a specific requirement or sample query, that requirement is quoted.

It complements `DESIGN.md` (the full design) and `README.md` (the summary). It ends
with the **v1 implementation scope** — what we build first.

---

## 0. Scope assumptions (locked)

- **Digital-text contracts only.** Provided PDFs/DOCX have a selectable text layer.
  Scanned/image-only docs and OCR are **out of scope for v1**. *Benefit:* removes a
  noisy, error-propagating stage; the parser is a clean `pypdf`/`python-docx` text +
  structure extractor. OCR can drop in later at the same `parser.py` seam without
  touching anything downstream.
- **Small corpus (≈4 docs).** v1 retrieves over a flat index; the scaling path to
  10k+ is documented (README → "Scaling") but not built now.
- **Local-first via Ollama**, per the chosen backend, with a provider seam so prod
  can swap to vLLM/OpenAI by config.

---

## 1. Re-reading the problem statement → requirement checklist

The assignment states the system *"should allow users to ask natural language
questions about contracts and receive grounded, precise, and well-cited answers,
along with risk indicators where applicable,"* and *"must run as a console-based
interactive system supporting multi-turn conversations."* Every explicit requirement,
and where the design satisfies it:

| # | Requirement (from the problem statement) | Where it's satisfied | Status |
|---|---|---|---|
| R1 | **Interactive console**, accept queries via CLI | `cli/console.py` REPL; `main.py` entry | ✅ designed |
| R2 | **Multi-turn** + maintain conversational context | `memory/session.py` + Orchestrator history-aware rewrite | ✅ designed |
| R3 | Display **final answer** | Synthesizer → Orchestrator render | ✅ designed |
| R4 | Display **referenced sections/clauses** | char-level citations from chunk metadata `[doc_id §section]` | ✅ designed |
| R5 | Display **risk flags (if applicable)** | Risk Assessor agent → structured flags | ✅ designed |
| R6 | **Multi-agent**, responsibilities clearly separated; *"over-agentification will be penalized"* | 6 agents, each owns one failure mode; decorative agents explicitly rejected (§2.1) | ✅ designed |
| R7 | RAG: **(a) chunking strategy** justified | parent-child + structure-aware + SAC (§3) | ✅ |
| R8 | RAG: **(b) embedding model** justified | BGE-M3, with MLEB evidence (§4) | ✅ |
| R9 | RAG: **(c) LLM via Ollama** — model name, **prompting per agent**, **determinism/verbosity control** | Qwen2.5-14B + Llama-3.1-8B; per-agent prompts; per-agent temp/seed/JSON table (§5) | ✅ |
| R10 | RAG: **(d) retrieval mechanism** justified | hybrid BM25+dense → RRF (§6) | ✅ |
| R11 | RAG: **(e) re-ranking (optional bonus)** | cross-encoder bge-reranker-v2-m3 (§7) | ✅ bonus |
| R12 | **Prompt design** (*"major evaluation criterion"*) | versioned templates, grounding/citation/refusal rules (§5, DESIGN §4) | ✅ |
| R13 | **Evaluation** — what / why / limitations | RAGAS + deterministic checks + safety metrics (§9) | ✅ designed |
| R14 | **Code quality**: modular, separation of concern, docstrings, readable config | package layout, typed `Settings`, docstringed stubs | ✅ scaffolded |
| R15 | **README** with all mandated sections | `README.md` | ✅ |
| R16 | Handle the **17 sample queries** incl. cross-doc, risk, and the out-of-scope traps (16, 17) | query taxonomy drives routing; guardrail for 16/17 (§8 + DESIGN App. A) | ✅ designed |

Nothing in the problem statement is unaddressed. The rest of this doc explains *why
each choice*, not just *that it exists*.

---

## 2. Why a multi-agent design — and why exactly these six

### 2.1 The core tension the assignment sets up

The brief rewards a multi-agent system but warns: *"Agents must be meaningful.
Over-agentification without clear responsibility separation will be penalized."* So
the design question is not "how many agents" but **"what are the genuinely distinct
responsibilities?"** Our test for admitting an agent: *does it own a distinct decision
and a distinct failure mode that can be tested in isolation?* If splitting something
out doesn't introduce a new decision, it stays a function, not an agent.

### 2.2 Why the responsibilities decompose the way they do

The 17 sample queries are not homogeneous — they deliberately stress different
capabilities (the brief calls them *"deliberately crafted to stress different aspects
of the system"*). Clustering them reveals the seams:

- *Q1–3* are **fact lookups** → need precise retrieval + extraction.
- *Q4–6, 14* need **reasoning over clauses** (survival, conditionals).
- *Q8, 9* are **cross-document** ("across agreements") → need decomposition + per-doc retrieval + comparison.
- *Q7, 10–13* are **risk** → need a risk taxonomy, not prose.
- *Q15* is **aggregation**.
- *Q16, 17* are **out of scope** (drafting, advice) → need a refusal, not an answer.

Each cluster implies a decision point. Mapping decisions → agents:

| Agent | The distinct decision it owns | The distinct failure it prevents | Why not fold it elsewhere |
|---|---|---|---|
| **Orchestrator** | "What is the user *actually* asking, given history?" | losing context across turns (breaks R2) | resolving "does *it* survive?" needs dialogue state the retriever shouldn't hold |
| **Planner/Router** | "What kind of query is this, is it in scope, and what's the retrieval plan?" | wrong route; missing decomposition of cross-doc queries (Q9) | routing is a classification decision distinct from retrieval execution |
| **Retriever** | "Which clauses are the evidence?" | document-level mismatch / low recall | search mechanics are orthogonal to generation |
| **Synthesizer** | "What is the grounded answer + citations?" | ungrounded generation | generation from fixed evidence is a separate skill from finding it |
| **Risk Assessor** | "Which clauses map to which risks, at what severity?" | missed/invented risk (R5 + 1/3 of queries) | risk classification is a different task with a different prompt + output schema than Q&A |
| **Verifier** | "Is retrieval good enough? Is the answer faithful?" | hallucination slipping through | a generator grading itself is the conflict-of-interest the literature flags |

**Why we *rejected* finer agents** (the over-agentification trap): a "Citation
agent," "Formatting agent," or "Coreference agent" would each make **no independent
decision** — citation is a constraint on the Synthesizer's output, formatting is
rendering, coreference is part of the Orchestrator's rewrite. Splitting them would be
agentification for show. **Benefit of restraint:** fewer LLM hops = lower latency and
cost, fewer failure points, and a system that's honest about where the real decisions
are.

**Why we *kept* the Verifier separate** even though it's "just checking": legal RAG's
defining risk is confident hallucination (Stanford HAI measured 17–34% even in
purpose-built legal tools). The brief demands *"grounded, precise, and well-cited"*
answers — verification is the mechanism that earns those adjectives. A self-grading
generator is unreliable, so the critic is a first-class, separately-promptable role.
This is the decomposer→retriever→**critic** pattern the agentic-RAG survey identifies
as the workhorse topology.

> **Net benefit:** the agent boundaries map 1:1 to the capability clusters the sample
> queries stress, each is unit-testable, and we can defend every one against "why is
> this its own agent?"

---

## 3. Chunking — why structure-aware parent-child + context augmentation

**Decision:** parse contracts into their native hierarchy; embed/search the **child**
(a single clause), but hand the LLM the **parent** (the enclosing section); prefix
each child's embedding text with a doc/section context cue.

**Why, against the requirement.** R4 demands answers reference *"document sections or
clauses."* Citation is only possible if a chunk *is* a clause with a section label and
character offsets — so chunking and the citation requirement are the same decision.
- *Why not fixed-size chunks?* They'd split a clause mid-sentence and lose the section
  number, making R4 (clause references) and precise answers impossible. Contracts have
  *real* structure (numbered sections) — throwing it away is strictly worse here.
- *Why parent-child?* A clause's meaning depends on its section (a penalty figure is
  meaningless without the obligation it attaches to). Searching the small child gives
  precision; returning the parent gives the LLM enough context to answer Q5/Q6
  correctly. Best of both: precise match, sufficient context.
- *Why context augmentation (SAC)?* The #1 legal-RAG failure is retrieving the right
  *kind* of clause from the *wrong* contract — fatal when the corpus has several
  agreements between overlapping parties (exactly our setup; Q3, Q8, Q9 name specific
  agreements). Prefixing *"[NDA between Acme and Vendor XYZ, §7 Termination]"* to the
  embedded text disambiguates the source document at retrieval time.

**Benefit:** directly enables R4 (clause-level citation), makes Q5/Q6-style reasoning
possible (full section in context), and attacks the dominant legal-RAG error before it
happens.

---

## 4. Embedding model — why BGE-M3 (local), legal-tuned in prod

**Decision:** default `BAAI/bge-m3`; behind an interface so prod can swap to a
legal-tuned model.

**Why.** R8 asks us to choose and justify an embedding model.
- *Why local/open?* Contracts are confidential; local embeddings mean **no document
  text leaves the machine** — a hard requirement for legal data and consistent with
  the Ollama-first backend you chose.
- *Why BGE-M3 specifically?* It emits **dense + sparse vectors from one model**, which
  powers our hybrid retrieval (§6) without standing up a second system; it has an **8k
  context** that comfortably holds long clauses/parents; and it's strong and open.
- *Why a swap path, not just BGE-M3?* The MLEB benchmark shows **general embedding
  skill ≠ legal retrieval skill** (models reshuffle badly between MTEB and legal
  benchmarks; the legal leaders are all domain-adapted). So for production accuracy the
  honest move is a legal-tuned model (voyage-law-2 / fine-tuned) — but only behind the
  interface, after a data-governance review, since it may mean leaving the box.

**Benefit:** privacy + hybrid-from-one-model now; a clear, evidence-backed accuracy
upgrade later without code churn.

---

## 5. LLM, prompting, and determinism — the explicit R9 sub-requirements

R9 is the most prescriptive requirement: *"LLM Model Choice (via Ollama or OpenAI) —
Model name; Prompting strategy per agent; How determinism or verbosity is
controlled."* We answer all three.

**(a) Model name + why.** `qwen2.5:14b-instruct` for the reasoning agents (Planner,
Synthesizer, Risk, Verifier); `llama3.1:8b-instruct` for cheap ops (rewrite,
classification).
- *Why Qwen2.5-14B?* Strong instruction-following and **reliable structured/JSON
  output** (our Planner/Risk/Verifier emit JSON), good reasoning, and it runs on a
  single workstation GPU — the right quality/footprint point for local.
- *Why a 8B second tier?* Query rewrite and intent classification are short, bounded
  tasks; paying 14B latency for them is waste. Two tiers = better latency without
  hurting the hard steps.

**(b) Prompting strategy *per agent*** (the requirement is explicit about "per
agent"). Every grounded agent shares a spine — role + scope ("you are not a lawyer"),
a **hard grounding rule** ("answer only from CONTEXT; if absent, say 'Not found'"),
**mandatory citation**, and **structured output** where a downstream agent consumes
it. On top of the spine, each agent's prompt is specialized: the Planner emits a
routing+plan JSON; the Synthesizer uses brief **IRAC** for yes/no interpretation
(Q4/Q5) which is associated with lower legal hallucination; the Risk agent is
constrained to a fixed taxonomy and told *not to invent risk*; the Verifier does
claim-by-claim entailment. Full templates are in `DESIGN.md §4` / `agents/prompts/`.
- *Why grounding + citation in every prompt?* Because R-level goals ("grounded,
  well-cited") are enforced at the prompt boundary, not hoped for.
- *Why a refusal prompt?* Q16/Q17 require declining drafting/advice gracefully — that's
  a designed prompt, not an afterthought.

**(c) Determinism & verbosity control** (named requirement). A per-agent table:
`temperature 0` for routing/risk/verification, `0.1` for synthesis; **fixed seed**;
**low top_p** on factual agents; **JSON-mode / grammar-constrained decoding** for
structured agents (no parse failures); **max-token caps** to bound verbosity. Higher
temperature is allowed *only* for final summary phrasing (Q15), over already-grounded
content.
- *Why temp 0 nearly everywhere?* Legal answers must be reproducible and conservative;
  creativity is a liability here. *Why caps?* The brief wants *precise* answers — token
  caps + "be concise" prevent rambling.

**Benefit:** every sub-clause of the most demanding requirement is met explicitly and
visibly, which the brief flags as a *"major evaluation criterion."*

---

## 6. Retrieval — why hybrid BM25 + dense, fused with RRF

**Decision:** run BM25 (sparse) and BGE-M3 dense in parallel, fuse with Reciprocal
Rank Fusion, apply planner-supplied metadata filters.

**Why, against the sample queries.** R10 asks us to justify retrieval.
- *Why BM25 in the mix?* The queries are full of **exact tokens** that semantic search
  under-weights: *"72 hours"* (Q14), *"99.9%"* uptime (Q2), party names like *"Vendor
  XYZ"* (Q7, Q13), section references. BM25 nails these.
- *Why dense too?* Other queries are **paraphrase**: *"share data with subcontractors"*
  (Q13) must match "disclosure to third-party processors." Dense handles that.
- *Why RRF to combine?* BM25 scores and cosine similarities live on incompatible
  scales; naive weighted averaging is fragile. RRF fuses **by rank**, needs no score
  normalization, and reliably beats either retriever alone.
- *Why metadata filtering?* It's how cross-document queries become tractable: Q9
  ("conflicting governing laws") = filter `clause_type=governing_law`, retrieve **one
  per document**, compare; Q8 = filter `clause_type=data_breach` per doc. Without
  filters these degrade into guesswork.

**Benefit:** robust across the deliberately mixed query set — exact-term queries and
paraphrase queries both retrieve well, and cross-doc queries get the per-document
evidence they need.

---

## 7. Re-ranking — why a cross-encoder (the optional bonus, R11)

**Decision:** after RRF yields ~32 finalists, a cross-encoder (`bge-reranker-v2-m3`)
scores each `(query, clause)` pair jointly and selects the final ~6.

**Why.** Bi-encoder similarity (what the vector search uses) embeds query and clause
*separately*; a cross-encoder reads them *together*, so it judges relevance far more
accurately — the difference between "mentions liability" and "actually answers whether
liability is capped for *this* breach" (Q5/Q7). It's slower, but only on a tiny
finalist set, so the cost is negligible.
- *Why deterministic cross-encoder over an LLM-selection step?* Reproducibility — a
  cross-encoder gives the same ranking every run; the LLM-selection alternative
  (Ranking-Free RAG) is noted as a later A/B.

**Benefit:** sharper final evidence → more precise, better-grounded answers, which is
exactly what the bonus is meant to demonstrate.

---

## 8. Multi-turn, memory, risk flags, and the out-of-scope guardrail

**Multi-turn + memory (R2).** The Orchestrator keeps rolling history + structured
session state (which agreement, party, last clause_type) and **rewrites** follow-ups
into standalone queries. *Why:* "Does it survive termination?" (a natural Q4 follow-up)
is unanswerable without resolving "it" — and retrieval needs a self-contained query.
*Benefit:* satisfies R2 literally and makes retrieval reliable across turns.

**Risk flags (R5).** A curated **risk taxonomy** (uncapped liability, cap excludes
confidentiality, breach-notice window too long/unspecified, subcontractor data sharing
without consent, one-sided indemnification, governing-law conflict, …) → each flag is
`{type, severity, party, rationale, citation}`. *Why structured, not prose:* R5 wants
risk *indicators*; structure makes severity sortable and citations checkable, and lets
Q15 ("summarize all risks") aggregate cleanly. *Benefit:* directly serves a third of
the benchmark (Q7, Q10–13, Q15).

**Out-of-scope guardrail (Q16, Q17).** The brief's last two queries are traps: "draft
a better NDA" and "what legal strategy should Acme take" are **not** what an analysis/QA
system should do. The Planner routes these to a **safe, useful refusal**: decline the
generative/advisory ask, offer the in-scope alternative ("I can flag weak clauses in
your current NDA with citations"), and attach the not-a-lawyer disclaimer.
- *Why refuse rather than attempt?* Drafting/strategy is unverifiable against the
  corpus → maximal hallucination and liability risk. Declining well is the mature,
  higher-scoring behavior. *Why a policy, not its own agent?* It's a rule, not a
  reasoner (consistent with §2's anti-over-agentification stance).
- *Why abstention generally?* The Verifier abstains ("Not found in the provided
  contracts") when evidence is missing — in legal QA a confident wrong answer is worse
  than "I don't know." *Benefit:* turns Q16/Q17 and no-answer cases from failures into
  demonstrations of judgment.

---

## 9. Evaluation — what / why / limitations (R13)

R13 requires *"at least basic evaluation"* and an explanation of **what**, **why**, and
**limitations**. Our plan:

**What we evaluate** (on a gold set = the 17 sample queries + adversarial no-answer
cases):
- *Retrieval:* context precision/recall, MRR@k, and **document-level retrieval
  accuracy** (did we pull from the *right contract*).
- *Generation:* **faithfulness/groundedness**, answer relevancy, **citation
  correctness** (do cited spans actually contain the claim), answer correctness.
- *Risk:* flag precision/recall + severity agreement.
- *Safety:* **abstention accuracy** and **out-of-scope refusal accuracy** (Q16/Q17).

**Why these matter here.** They map to *real legal harm*: document-level accuracy
catches the wrong-contract error; faithfulness + citation correctness are the metrics
behind "grounded and well-cited"; abstention/refusal accuracy rewards the system for
knowing its limits. Generic accuracy alone would hide the failures that matter most.

**How.** RAGAS (LLM-as-judge) for faithfulness/relevancy/context; deterministic
exact/regex checks for hard facts ("72 hours", governing-law state, "99.9%"); a small
hand-labeled gold set for correctness, citations, and risk flags. Run as a CI gate.

**Limitations (stated honestly, as the brief asks).** Tiny corpus / 17 queries → high
variance, directional not significant; LLM-as-judge is itself imperfect and non-
deterministic (we pin judge model+temp and spot-check); **faithfulness ≠ legal
correctness** (a grounded answer can be legally naive); no licensed-attorney ground
truth; no adversarial-paraphrase or long-context stress yet.

---

## 10. Code structure — why this layout (R14)

R14 wants *"clean modular structure, clear separation of concern, meaningful
docstrings, readable configuration management."* The package mirrors the **data flow**
(`ingestion → retrieval → agents → cli`) so a reader can follow a query end-to-end;
cross-cutting concerns get their own packages (`llm/` provider seam, `memory/`,
`config/`, `eval/`). *Why the seams:* the `LLMClient` seam makes the Ollama→vLLM/OpenAI
swap a config change (R9 + production); the `store` seam makes FAISS→Qdrant a config
change (scaling). Config is one typed `Settings` object (pydantic) + a readable
`config.yaml` — single source of truth for models, k's, and temperatures.
*Benefit:* every requirement maps to an obvious module, and the two things most likely
to change (LLM backend, vector store) are isolated behind interfaces.

---

## 11. v1 implementation scope — what we build first

Now the second ask: **the tasks in hand.** We build a thin but *complete* vertical
slice first — every requirement represented end-to-end on the 4-doc corpus — then
deepen. Order is chosen so each milestone is runnable and demoable.

**Milestone 1 — Ingestion (offline).** `parser.py` (digital-text extraction +
section/clause detection) → `chunker.py` (parent-child + metadata + SAC cue) →
`indexer.py` + `store.py` (BGE-M3 embeddings into FAISS/Chroma + BM25). *Exit:*
`python main.py --ingest` builds an index over the 4 contracts. *(Satisfies R7/R8 in
running code.)*

**Milestone 2 — Retrieval core.** `bm25.py`, `dense.py`, `fusion.py` (RRF),
`rerank.py`, parent expansion. *Exit:* a query returns ranked, cited clauses on the
CLI. *(R10, R11.)*

**Milestone 3 — Answer path.** `llm/client.py` (Ollama) → `synthesizer.py` (grounded
answer + citations) → minimal `cli/console.py`. *Exit:* Q1–Q3 answered with citations.
*(R1, R3, R4, R9 prompting/determinism.)*

**Milestone 4 — Agent graph + control.** `planner.py` (intent + scope + filters +
decomposition), `orchestrator.py`, `verifier.py` (pre-gen sufficiency + corrective
loop, post-gen faithfulness). *Exit:* cross-doc (Q8/Q9), abstention, and the corrective
loop work. *(R6, grounding/faithfulness.)*

**Milestone 5 — Risk + guardrail + memory.** `risk.py` (taxonomy → flags),
out-of-scope refusal (Q16/Q17), `memory/session.py` (multi-turn rewrite). *Exit:*
Q7/Q10–Q15 produce flags; Q16/Q17 refuse safely; follow-ups resolve. *(R2, R5, R16.)*

**Milestone 6 — Evaluation.** `eval/gold_set.py` (17 queries + no-answer cases) +
`eval/runner.py` (RAGAS + checks). *Exit:* `python main.py --eval` prints the metric
report. *(R13.)*

**Deferred (documented, not built):** structured-extraction/knowledge layer for
corpus-wide analytics at scale, vLLM serving, ANN store swap, caching, RBAC,
observability, OCR. These live in the README "Scaling" section and `DESIGN.md §9` as
the production evolution — building them now would be over-engineering for a 4-doc
corpus.

**Suggested first task to code:** Milestone 1 (ingestion), because every other
milestone depends on a populated index, and it's where the two most consequential RAG
decisions (chunking, embeddings) become real and testable.
