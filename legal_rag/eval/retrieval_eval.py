"""Retrieval evaluation: run queries through the retriever and score against qrels.

Uses the planner to derive the same metadata filters the live system would apply
(so the numbers reflect real end-to-end retrieval), then measures recall@k, MRR, and
nDCG@k against the labeled gold passages. Retrieval is graded at the *section* level —
the Retriever de-duplicates by parent section, and qrels are keyed on (doc_id,
section_no), so the two align.
"""

from __future__ import annotations

from legal_rag.eval.gold_set import GOLD
from legal_rag.eval.qrels import QRELS
from legal_rag.eval.retrieval_metrics import (
    mrr, ndcg_at_k, recall_at_k, hit_at_k,
)

DEFAULT_KS = (1, 3, 5, 10)


def _query_text(qid: int) -> str:
    for item in GOLD:
        if item["id"] == qid:
            return item["query"]
    raise KeyError(qid)


def evaluate_retrieval(retriever, planner, qrels=None, ks=DEFAULT_KS,
                       use_planner_filters: bool = True) -> dict:
    qrels = qrels or QRELS
    max_k = max(ks)
    rows: list[dict] = []
    agg_recall = {k: [] for k in ks}
    agg_ndcg = {k: [] for k in ks}
    agg_hit = {k: [] for k in ks}
    agg_mrr: list[float] = []

    for qid, relevant in qrels.items():
        query = _query_text(qid)
        filters = {}
        if use_planner_filters and planner is not None:
            plan = planner.plan(query)
            filters = plan.get("filters") or {}
        evidence = retriever.retrieve(query, filters, final_k=max_k)
        ranked_rel = [(e.doc_id, e.section_no) in relevant for e in evidence]
        n_rel = len(relevant)

        row = {"id": qid, "n_rel": n_rel, "mrr": round(mrr(ranked_rel), 3)}
        agg_mrr.append(mrr(ranked_rel))
        for k in ks:
            r = recall_at_k(ranked_rel, k, n_rel)
            n = ndcg_at_k(ranked_rel, k, n_rel)
            agg_recall[k].append(r)
            agg_ndcg[k].append(n)
            agg_hit[k].append(hit_at_k(ranked_rel, k))
            row[f"r@{k}"] = round(r, 3)
            row[f"ndcg@{k}"] = round(n, 3)
        rows.append(row)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    metrics = {"MRR": mean(agg_mrr)}
    for k in ks:
        metrics[f"recall@{k}"] = mean(agg_recall[k])
        metrics[f"nDCG@{k}"] = mean(agg_ndcg[k])
        metrics[f"hit@{k}"] = mean(agg_hit[k])
    return {"metrics": metrics, "rows": rows, "ks": list(ks), "n": len(qrels)}


def format_retrieval_report(report: dict) -> str:
    ks = report["ks"]
    lines = ["", "=== Retrieval evaluation (vs qrels) ===", f"queries: {report['n']}", ""]
    header = "Q   n_rel  MRR   " + "  ".join(f"r@{k:<2} nDCG@{k:<2}" for k in ks)
    lines.append(header)
    for row in report["rows"]:
        cells = "  ".join(f"{row[f'r@{k}']:<4} {row[f'ndcg@{k}']:<6}" for k in ks)
        lines.append(f"{row['id']:<3} {row['n_rel']:<5} {row['mrr']:<5} {cells}")
    lines.append("")
    lines.append("--- aggregate ---")
    for k, v in report["metrics"].items():
        lines.append(f"  {k:12s}: {v}")
    lines.append("")
    return "\n".join(lines)
