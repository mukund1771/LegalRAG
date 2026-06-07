"""Prompt templates. See DESIGN.md section 4 for the full, annotated versions."""

PLANNER = """..."""        # intent + scope + decompose (JSON, temp 0)
SYNTHESIZER = """..."""    # grounded answer + [doc_id §section] citations, IRAC
RISK = """..."""           # risk taxonomy -> JSON flags
VERIFIER_POSTGEN = """..."""  # claim-by-claim faithfulness check
REFUSAL = """..."""        # scoped, useful out-of-scope refusal + disclaimer
