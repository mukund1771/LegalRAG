"""Run the gold set and report metrics.

Retrieval:  context precision/recall, MRR@k, document-level retrieval accuracy.
Generation: faithfulness, answer relevancy, citation correctness (RAGAS + checks).
Risk:       flag precision/recall + severity agreement.
Safety:     abstention accuracy, out-of-scope refusal accuracy.
"""
def run_eval(system, gold) -> dict: raise NotImplementedError
