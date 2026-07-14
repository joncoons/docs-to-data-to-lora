# Golden Evaluation Graphics

Generated documentation graphics for the `golden-v1` evaluation.

## Files

- `rag_vs_norag_composite.svg` - grouped composite score chart for the reduced no-RAG and RAG population.
- `rag_vs_norag_axis_heatmap.svg` - axis-level score heatmap for accuracy, completeness, faithfulness, and clarity.
- `golden_eval_score_table.svg` - numeric SVG table for the same reduced population.
- `ragas_coverage_status.svg` - RAGAS coverage/status graphic.
- `golden_eval_graphics_data.json` - source data used to render the SVGs.

## Scope

The no-RAG and RAG graphics use the same reduced comparison set: 1B, 3B, and 8B LE r32 adapters plus the dense Llama 3.3 70B reference for both NIM and NeMo Microservices corpora. Scores are Claude Sonnet 4.6 single-axis judge means on a 1-5 scale.

The RAGAS graphic is intentionally a status artifact. A full reduced-population RAGAS run has not been completed yet; the existing one-row smoke reached NeMo Evaluator but failed before scoring because `params.judge_embeddings.model` was not configured.

## Reusing For Another Corpus

For another corpus, keep the same artifact contract: write compatible single-axis summary files, preserve corpus slugs and source provenance, and rerun `scripts/eval/render_golden_eval_graphics.py` after updating the target descriptors. The SVGs are documentation artifacts derived from summaries, not hand-maintained screenshots.

Generated at: `2026-07-14T13:05:39.317309+00:00`
