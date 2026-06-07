#!/usr/bin/env bash
# Commit + push Milestones 5 & 6 (run on your machine, AFTER the M3+M4 PR is merged
# to main so this branches cleanly off it).
set -e
cd "$(dirname "$0")/.."

rm -f .git/HEAD.lock .git/index.lock .git/objects/maintenance.lock 2>/dev/null || true

git rev-parse --verify feature/m5-m6-risk-eval >/dev/null 2>&1 \
  && git checkout feature/m5-m6-risk-eval \
  || git checkout -b feature/m5-m6-risk-eval

git add legal_rag/agents/risk.py legal_rag/agents/orchestrator.py legal_rag/app.py \
        legal_rag/agents/planner.py legal_rag/agents/verifier.py \
        legal_rag/ingestion/clause_tags.py \
        legal_rag/eval/gold_set.py legal_rag/eval/runner.py \
        main.py tests/test_eval.py README.md scripts/

git commit -m "feat(risk+eval): Milestones 5 & 6 — risk assessor + evaluation harness

M5 (risk):
- RiskAssessor: per-clause detectors (uncapped liability, cap-excludes-confidentiality,
  breach-notification window, subprocessor sharing, one-sided indemnification,
  auto-renewal) + corpus-level governing-law-conflict detection
- orchestrator gathers risk-bearing clauses across the corpus for risk/summary intents;
  risk flags rendered in the console
- planner refinements (clause filter only for fact/cross-doc; broader risk triggering;
  governing-law keyword)

M6 (evaluation):
- gold set: 17 sample queries + adversarial no-answer cases
- runner: routing, doc-hit, clause-hit, refusal, abstention, risk-recall metrics
- python main.py --eval
- corrective-loop abstention when retrieval stays unreliable
- 11 new offline tests (30 total passing)

Offline eval: routing 0.95, clause-hit 1.0, refusal 1.0, abstention 0.94, risk-recall 1.0"

git push -u origin feature/m5-m6-risk-eval
echo
echo "Open the PR: base=main  compare=feature/m5-m6-risk-eval"
