"""Relevance judgments (qrels) for retrieval evaluation.

Each query is mapped to the set of (doc_id, section_no) that actually answer it — the
gold passages. This is what recall@k / MRR / nDCG need: without it you can only check
"right doc-type", not "the correct clause ranked first".

Labels are tied to the FROZEN fixture corpus (tests/fixtures/contracts), so the
numbers are stable across runs and comparable across model ablations. Keep these in
sync with the gold set IDs in gold_set.py.

Section map (real corpus):
  nda_acme_vendor:            1 Definition · 2 Confidentiality · 3 Term&Termination(+survival)
                              4 Liability · 5 Governing Law
  vendor_services_agreement:  1 Scope · 2 Payment · 3 Indemnification · 4 Limitation of Liability
                              5 Termination · 6 Governing Law
  service_level_agreement:    1 Service Availability · 2 Service Credits · 3 Exclusions · 4 Liability
  data_processing_agreement:  1 Scope · 2 Security · 3 Data Breach Notification · 4 Subprocessors
                              5 Liability · 6 Governing Law
"""

from __future__ import annotations

NDA = "nda_acme_vendor"
VEN = "vendor_services_agreement"
SLA = "service_level_agreement"
DPA = "data_processing_agreement"

# query_id -> set of (doc_id, section_no) that are relevant
QRELS: dict[int, set[tuple[str, str]]] = {
    1:  {(NDA, "3")},                                   # NDA termination notice
    2:  {(SLA, "1")},                                   # uptime commitment
    3:  {(VEN, "6")},                                   # governing law of vendor agmt
    4:  {(NDA, "3")},                                   # confidentiality survival
    5:  {(NDA, "4"), (VEN, "4")},                       # liability cap vs confidentiality
    6:  {(SLA, "2")},                                   # SLA remedies / service credits
    7:  {(VEN, "4"), (DPA, "5")},                       # vendor liability for data breach
    8:  {(DPA, "3")},                                   # breach notification timelines
    9:  {(NDA, "5"), (VEN, "6"), (DPA, "6")},           # conflicting governing laws
    10: {(NDA, "4"), (VEN, "4"), (DPA, "5")},           # liability exposure risk
    11: {(VEN, "3"), (VEN, "4"), (NDA, "4")},           # financial risk to Acme
    12: {(NDA, "4")},                                   # unlimited liability
    13: {(DPA, "4")},                                   # subprocessor data sharing
    14: {(DPA, "3")},                                   # 72h breach notification delay
    15: {(NDA, "4"), (VEN, "3"), (VEN, "4"),
         (DPA, "3"), (DPA, "4")},                       # summarize all risks (broad)
}
