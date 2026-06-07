"""Evaluation harness — run the gold set through the system and report metrics.

What we measure and why (legal-RAG specific):
- routing_accuracy   : did the planner pick the right intent / scope?
- doc_hit_rate       : is the top evidence from the RIGHT contract? (the #1 legal
                       failure is the right clause from the wrong agreement)
- clause_hit_rate    : does the evidence contain the expected clause type?
- refusal_accuracy   : are out-of-scope requests (Q16/Q17) declined?
- abstention_accuracy: does the system say "not found" when the answer is absent,
                       and answer when it is present? (a confident wrong answer is
                       the worst outcome in legal QA)
- risk_recall        : fraction of expected risk flags actually raised.

Limitations: tiny corpus / 19 items -> directional, not significant; the offline
FakeEmbedder understates retrieval; no licensed-attorney ground truth; faithfulness
!= legal correctness. (See DESIGN.md §5.)
"""

from __future__ import annotations

from legal_rag.eval.gold_set import GOLD
from legal_rag.memory.session import SessionMemory


def _reset_memory(orchestrator) -> None:
    orchestrator.memory = SessionMemory()


def run_eval(orchestrator, gold: list[dict] | None = None) -> dict:
    gold = gold or GOLD
    rows: list[dict] = []

    agg = {
        "routing": [0, 0], "doc_hit": [0, 0], "clause_hit": [0, 0],
        "refusal": [0, 0], "abstention": [0, 0], "risk_recall": [0, 0],
    }

    for item in gold:
        _reset_memory(orchestrator)
        result = orchestrator.handle_turn(item["query"])
        ev = result.answer.evidence
        flags = {f["risk_type"] for f in result.risk_flags}
        row = {"id": item["id"], "query": item["query"][:48],
               "intent_pred": result.plan.get("intent"), "checks": {}}

        # routing
        agg["routing"][1] += 1
        ok = result.plan.get("intent") == item.get("intent")
        agg["routing"][0] += ok
        row["checks"]["routing"] = ok

        # refusal
        if item.get("expect_refuse"):
            agg["refusal"][1] += 1
            ok = result.refused
            agg["refusal"][0] += ok
            row["checks"]["refusal"] = ok

        # abstention (both directions)
        if item.get("expect_abstain"):
            agg["abstention"][1] += 1
            ok = result.answer.abstained
            agg["abstention"][0] += ok
            row["checks"]["abstain"] = ok
        elif not item.get("expect_refuse"):
            agg["abstention"][1] += 1
            ok = not result.answer.abstained
            agg["abstention"][0] += ok
            row["checks"]["answered"] = ok

        # doc-hit (top evidence from an expected document)
        if item.get("expected_doc_types") and ev:
            agg["doc_hit"][1] += 1
            ok = ev[0].doc_type in item["expected_doc_types"]
            agg["doc_hit"][0] += ok
            row["checks"]["doc_hit"] = ok

        # clause-hit (expected clause type present in evidence)
        if item.get("expected_clause") and ev:
            agg["clause_hit"][1] += 1
            ok = any(e.clause_type == item["expected_clause"] for e in ev)
            agg["clause_hit"][0] += ok
            row["checks"]["clause_hit"] = ok

        # risk recall
        if item.get("expected_risks"):
            agg["risk_recall"][1] += 1
            ok = item["expected_risks"].issubset(flags)
            agg["risk_recall"][0] += ok
            row["checks"]["risk"] = ok

        rows.append(row)

    metrics = {k: (round(c / t, 3) if t else None) for k, (c, t) in agg.items()}
    return {"metrics": metrics, "rows": rows, "n": len(gold)}


def format_report(report: dict) -> str:
    lines = ["", "=== LegalRAG evaluation ===", f"items: {report['n']}", ""]
    for row in report["rows"]:
        checks = " ".join(f"{k}={'P' if v else 'F'}" for k, v in row["checks"].items())
        lines.append(f"Q{row['id']:>2} [{row['intent_pred']:>20}] {row['query']:<50} {checks}")
    lines.append("")
    lines.append("--- aggregate metrics ---")
    for k, v in report["metrics"].items():
        lines.append(f"  {k:18s}: {v}")
    lines.append("")
    return "\n".join(lines)
