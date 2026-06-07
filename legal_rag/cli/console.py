"""Interactive multi-turn console (REPL).

Renders the three things the brief asks for each turn: the final answer, the
referenced clauses (citations), and — once Milestone 5 lands — risk flags.
"""

from __future__ import annotations

from legal_rag.app import build_system
from legal_rag.config.settings import load_settings


def repl(settings=None) -> None:
    settings = settings or load_settings()
    try:
        orchestrator = build_system(settings)
    except FileNotFoundError:
        print("No index found. Run `python main.py --ingest` first.")
        return

    print("LegalRAG console — ask about the contracts. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        result = orchestrator.handle_turn(user_input)
        _render(result)


def _render(result) -> None:
    ans = result.answer
    print(f"\n{ans.text}\n")
    if ans.citations:
        print("Referenced clauses:")
        for c in ans.citations:
            print(f"  - {c}")
    if result.risk_flags:
        print("Risk flags:")
        for f in result.risk_flags:
            print(f"  - [{f.get('severity', '?')}] {f.get('risk_type')}: {f.get('citation')}")
    print(f"\n(intent: {result.plan.get('intent')}"
          f"{' | refused' if result.refused else ''}"
          f"{' | abstained' if ans.abstained else ''})\n")
    print("-" * 60)
