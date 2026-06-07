# LegalRAG — Evaluation Findings

A running log of evaluation results and what they tell us. Newest run at the top.

---

## Run 2 — ablation (bge-m3 ± cross-encoder) + system eval after fixes

### Ablation: retrieval metrics by config

| config | recall@1 | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| fake / lexical (baseline) | 0.580 | **0.849** | 0.856 | 0.797 |
| ollama (bge-m3) / lexical | 0.513 | 0.816 | 0.833 | 0.770 |
| **ollama (bge-m3) / cross_encoder** | **0.624** | 0.804 | **0.867** | 0.793 |

**Reading it.** The thesis held, with a precise shape:
- **`bge-m3` + cross-encoder is the only config that beats the baseline**, and it wins
  exactly where it matters: **recall@1 0.624 (+0.044 vs baseline, +0.111 vs bge-m3/lexical)
  and the best MRR (0.867)**. Getting the right clause to **rank 1** is what counts when
  the synthesizer reads the top passages.
- **recall@5 is flat/slightly lower** (0.804) — expected. A reranker *reorders* the
  retrieved set; it cannot add a clause that dense+BM25 never surfaced. So its gain shows
  up at the top of the ranking (recall@1, MRR), not in deeper recall.
- **bge-m3 alone (lexical) ≈ baseline** — confirms Run 1's finding that the lexical
  reranker was the bottleneck, not the embedder. The embedder only pays off once the
  reranker can exploit it.

→ **Recommended production config: `bge-m3` embeddings + `bge-reranker-v2-m3` cross-encoder.**

### System eval after the two fixes (bge-m3 / lexical)

| metric | Run 1 (bge-m3/lexical) | **Run 2 (after fixes)** | Δ |
|---|---|---|---|
| routing | 0.947 | **1.000** | +0.05 |
| abstention | 0.824 | **0.882** | +0.06 |
| risk_recall | 0.667 | **0.833** | +0.17 |
| doc_hit | 0.900 | 0.900 | 0 |
| clause_hit | 1.000 | 1.000 | 0 |
| refusal | 1.000 | 1.000 | 0 |

**Q7 fully recovered** (routing ✓, answered ✓, risk ✓) from the planner fix → it now
gathers broad risk evidence and raises the uncapped-liability flag. Remaining misses:
- **Q13** still fails on the *lexical* reranker (subcontractors→subprocessor). The
  ablation's recall@1 jump suggests the cross-encoder recovers it — **confirm with
  `python main.py --retrieval-eval --backend ollama --reranker cross_encoder`** (and ideally
  run the system eval with `--reranker cross_encoder` too).
- **Q14** correctly abstains (no consequences clause in the real DPA).

### Open items
- [ ] Confirm Q13 end-to-end under cross_encoder; if so, **set `reranker_backend: cross_encoder` as the default**.
- [ ] Lock a CI threshold gate on these floors (e.g. routing ≥ 0.95, refusal = 1.0, recall@1 ≥ 0.55, MRR ≥ 0.80).

---

## Fixes applied between Run 1 and Run 2

1. **Planner routing (Q7).** "Is X's liability capped for **data breaches**?" now routes
   to `risk_analysis` (was `interpretation`); the confidentiality variant (Q5) stays
   `interpretation`. Offline routing accuracy: **0.947 → 1.0**. Side effect: Q7 now goes
   through broad risk-evidence gathering, which should also recover its answer + flag.
2. **Reranker-agnostic retrieval grading.** `Verifier.grade_retrieval` no longer abstains
   on zero *lexical* overlap alone — a confident reranker score (cross-encoder) now counts
   as relevant. This is **inert for the lexical reranker** (offline numbers unchanged,
   adversarial queries still abstain) and specifically un-blocks dense / paraphrase matches
   under the cross-encoder (targets Q13-style misses). Threshold `min_rerank_relevance`
   is configurable (default 0.0) and may need per-reranker tuning.
3. Re-diagnosis of **Q14**: not a bug. The real DPA has **no "consequences of late
   notification" clause**, so abstaining on "what happens if delayed >72h" is *correct*
   behavior — the gold expectation was too strong for this corpus.

37 tests pass. Re-run with the cross-encoder to produce Run 2:
`pip install sentence-transformers torch && python main.py --ablation`

---

## Run 1 — bge-m3 (Ollama) + lexical reranker, real corpus

**Setup**
- Corpus: 4 real contracts (NDA, Vendor Services Agreement, SLA, DPA) → 21 parent / 41 child chunks.
- Embedder: `ollama:bge-m3` (1024-dim). Reranker: `lexical`. LLM (synthesis): `qwen2.5:14b-instruct`. Planner/Verifier: heuristic.
- Baseline for comparison: `fake-hash` embedder + `lexical` reranker (deterministic, no model).

### Retrieval metrics (vs qrels, 15 answerable queries)

| metric | fake / lexical (baseline) | **bge-m3 / lexical** | Δ |
|---|---|---|---|
| MRR | 0.856 | **0.833** | −0.02 |
| recall@1 | 0.580 | **0.513** | −0.07 |
| recall@3 | 0.813 | **0.816** | ≈0 |
| recall@5 | 0.849 | **0.816** | −0.03 |
| nDCG@5 | 0.797 | **0.770** | −0.03 |
| hit@5 | 0.933 | **0.933** | 0 |
| hit@10 | 0.933 | **0.933** | 0 |

### System metrics (19 queries: 17 sample + 2 adversarial)

| metric | fake / lexical | **bge-m3 / lexical** | Δ |
|---|---|---|---|
| routing | 0.947 | 0.947 | 0 |
| doc_hit | 0.900 | 0.900 | 0 |
| clause_hit | 1.000 | 1.000 | 0 |
| refusal | 1.000 | 1.000 | 0 |
| abstention | 1.000 | **0.824** | −0.18 |
| risk_recall | 0.833 | **0.667** | −0.17 |

### Per-query misses (bge-m3 / lexical)

| Q | query | symptom | likely cause |
|---|---|---|---|
| 2 | uptime commitment in the SLA | r@1=0 (recovers by r@3) | "uptime" appears in both SLA §1 and §2; lexical reranker tie |
| 7 | Vendor liability capped for data breaches | routing=F, answered=F (abstained) | "Is…" → interpretation not risk_analysis; answer needs DPA§5 **+** Vendor§4 stitched, LLM abstained |
| 12 | any unlimited liability | r@1=0 (recovers by r@3) | "No explicit limitation" phrasing ranks below a closer lexical match |
| 13 | share data with **subcontractors** | **recall=0 at all k**, answered=F, doc_hit=F | dense finds DPA§4 "subprocessor", but the **lexical reranker demotes it** (no token overlap with "subcontractor") |
| 14 | delay breach notification > 72h | answered=F (abstained) | retrieval r@1=1.0 (found it), but LLM/verifier abstained |

### Findings

1. **The reranker, not the embedder, is the current bottleneck.** Under the lexical
   reranker, `bge-m3` is statistically tied with (marginally below) the fake baseline.
   The lexical reranker re-sorts the fused candidates by query-token overlap, which
   *undoes* dense retrieval's semantic match on paraphrase queries — most visibly
   **Q13** (`subcontractors` ≠ `subprocessor`), which stays at recall 0 even though
   `bge-m3` can match them. **→ The bge-m3 + cross-encoder ablation row is the real
   test; expect Q13 and Q2/Q12 ties to resolve there.**

2. **False abstentions appeared with dense retrieval (Q7, Q14).** Retrieval actually
   found the relevant clause (Q14 r@1=1.0), but the turn still abstained. Two
   contributing factors: (a) the heuristic verifier grades retrieval by *lexical*
   coverage, which mismatches dense results; (b) `qwen` is conservative and emits the
   abstain phrase when the context doesn't fully/explicitly answer (Q7 needs two
   clauses stitched across DPA + Vendor). This is *safe* behavior but costs recall.

3. **risk_recall dropped (0.83 → 0.67)** for the same reason as Q13: the subprocessor
   and data-breach clauses that risk detection keys on were demoted by the lexical
   reranker / not surfaced, so flags weren't raised.

4. **What's robust regardless of embedder:** routing 0.95, clause_hit 1.0, refusal
   1.0, doc_hit 0.90, and hit@3/@5/@10 = 0.93. The structure (planner, filters,
   parent-child, citations, guardrails) carries most of the quality; the embedder
   choice matters specifically on paraphrase-heavy retrieval.

### Action items

- [ ] **Run `python main.py --ablation`** to get the `bge-m3 / cross_encoder` row
      (needs `pip install sentence-transformers torch`). Primary question: does the
      cross-encoder recover Q13 / lift recall@1 + MRR above baseline?
- [ ] **Grade retrieval on reranker score, not lexical coverage**, in the Verifier —
      so dense retrieval doesn't trigger false "incorrect" → abstain (Q14).
- [ ] **Q7 routing**: "Is X's liability capped…" should route to risk_analysis, not
      interpretation (tweak planner intent rules).
- [ ] Consider widening `final_k` for risk/summary gathering so demoted clauses still
      reach the risk agent.

---

### Raw output — retrieval-eval (bge-m3 / lexical)

```
MRR 0.833 | recall@1 0.513 | recall@3 0.816 | recall@5 0.816 | recall@10 0.816
nDCG@1 0.733 | nDCG@3 0.783 | nDCG@5 0.77 | nDCG@10 0.77 | hit@{1,3,5,10}=0.733/0.933/0.933/0.933
Per-query r@1: Q1 1.0 · Q2 0.0 · Q3 1.0 · Q4 1.0 · Q5 0.5 · Q6 1.0 · Q7 0.0 · Q8 1.0
              Q9 0.333 · Q10 0.333 · Q11 0.333 · Q12 0.0 · Q13 0.0 · Q14 1.0 · Q15 0.2
```

### Raw output — system eval (bge-m3 / lexical)

```
routing 0.947 | doc_hit 0.9 | clause_hit 1.0 | refusal 1.0 | abstention 0.824 | risk_recall 0.667
Misses: Q7 (routing,answered,risk) · Q13 (answered,doc_hit,risk) · Q14 (answered)
```
