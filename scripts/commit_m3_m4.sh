#!/usr/bin/env bash
# Commit + push Milestones 3 & 4 (run on your machine). They were built together
# (the console wires the orchestrator), so they go in one branch / PR.
set -e
cd "$(dirname "$0")/.."

rm -f .git/index.lock

git rev-parse --verify feature/m3-m4-agents >/dev/null 2>&1 \
  && git checkout feature/m3-m4-agents \
  || git checkout -b feature/m3-m4-agents

git add legal_rag/llm/client.py \
        legal_rag/agents/synthesizer.py legal_rag/agents/planner.py \
        legal_rag/agents/verifier.py legal_rag/agents/orchestrator.py \
        legal_rag/agents/prompts/templates.py \
        legal_rag/memory/session.py legal_rag/app.py legal_rag/cli/console.py \
        legal_rag/models.py legal_rag/config/settings.py main.py \
        tests/test_agents.py README.md scripts/

git commit -m "feat(agents): Milestones 3 & 4 — answer path + multi-agent orchestration

M3 (answer path):
- provider-agnostic LLM client (Ollama chat + offline FakeLLM), determinism controls
- Synthesizer: grounded answer, programmatic (verifiable) citations, abstention
- prompt templates (synthesizer/planner/verifier/refusal)

M4 (agent graph + control):
- Planner: intent + scope guardrail (Q16/Q17) + filters + decomposition (heuristic/LLM)
- Verifier: CRAG retrieval grading + Self-RAG faithfulness check
- SessionMemory: history-aware coreference rewrite for multi-turn
- Orchestrator: control flow + corrective re-retrieval loop + safe refusal + abstention
- interactive console (python main.py); app factory wiring
- 9 new offline tests (25 total passing)"

git push -u origin feature/m3-m4-agents
echo
echo "Open the PR: base=main  compare=feature/m3-m4-agents"
