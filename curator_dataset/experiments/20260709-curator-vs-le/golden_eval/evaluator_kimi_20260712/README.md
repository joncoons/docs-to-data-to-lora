# Optional NeMo Evaluator RAGAS Smoke

Date: 2026-07-12

This directory captures a one-row NeMo Evaluator live/RAGAS smoke attempt with Kimi as the judge and MLflow artifact export enabled.

Status: provenance only. Do not use this output for the formal LE-vs-Curator winner evaluation.

Reason: the active winner evaluation is no-RAG and compares saved question-only model answers against immutable golden reference answers. RAGAS is context/retrieval-oriented and can require embedding-backed judge metrics such as `response_relevancy`, which changes the semantics of the experiment.

Use `scripts/eval/run_nemo_evaluator_saved_responses.py` only for a later optional RAGAS diagnostic pass.
