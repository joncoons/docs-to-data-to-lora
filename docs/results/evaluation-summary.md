# Evaluation Results

This page keeps the public repo focused on the final, reusable workflow while still showing the case-study outcome that motivated the implementation. Raw logs, JSONL corpora, Customizer run directories, evaluator traces, and local manifests are intentionally excluded from the public branch.

## What Was Compared

The case study used two representative public technical-documentation corpora to compare two dataset-generation strategies for LoRA SFT. In the retained charts, Corpus A is the NIM documentation slice and Corpus B is the NeMo Microservices documentation slice. They are representative corpora, not the conceptual target of the repo:

- Logical Entailment (LE): premise/conclusion extraction followed by grounded QA/KVP generation.
- Curator DiverseQA: NeMo Curator synthetic QA generation over the same corpus scope.

Adapters were trained over dense Llama 1B, 3B, and 8B bases with LoRA ranks r16 and r32. A dense Llama 3.3 70B Instruct target was used as a reference comparison target, not as the judge. The formal no-RAG scoring judged saved model answers against an immutable golden QA set so that retrieval quality did not mask model/adapter behavior.

## Validation-Loss Signal

The retained validation-loss evidence favored the LE dataset path over Curator DiverseQA across the matched 8B comparison slices. Both dataset paths used matched five-epoch Customizer LoRA SFT runs for this comparison. Lower is better. The corpus labels identify representative public examples only.

<table width="100%">
<thead>
<tr><th>Corpus</th><th align="right">Rank</th><th align="right">LE best val loss</th><th align="right">Curator best val loss</th><th align="right">Curator minus LE</th></tr>
</thead>
<tbody>
<tr><td>Corpus A</td><td align="right">r16</td><td align="right">1.311</td><td align="right">1.424</td><td align="right">+8.6%</td></tr>
<tr><td>Corpus A</td><td align="right">r32</td><td align="right">1.269</td><td align="right">1.422</td><td align="right">+12.1%</td></tr>
<tr><td>Corpus B</td><td align="right">r16</td><td align="right">0.991</td><td align="right">1.457</td><td align="right">+47.0%</td></tr>
<tr><td>Corpus B</td><td align="right">r32</td><td align="right">0.947</td><td align="right">1.416</td><td align="right">+49.6%</td></tr>
</tbody>
</table>

<img src="le-vs-curator-validation-loss.svg" alt="LE vs Curator validation-loss comparison" width="100%">

Both columns report the retained best validation-loss signal from matched five-epoch Customizer LoRA SFT runs.

## Golden Evaluation Methodology

The golden test set was built from prior held-out QA rows rather than current validation splits. The construction process removed exact normalized prompt overlaps with active train/validation files, retained source provenance, and emitted deterministic row identifiers plus checksums. For the primary model-quality test, answers were collected without RAG or injected context.

The no-RAG single-axis evaluation used four 1-5 axes: accuracy, completeness, reference-grounded faithfulness, and clarity. A separate independent judge scored saved responses only against the question, immutable reference answer, and model answer. The reduced RAG follow-up used the same winner population and judged saved RAG answers against the same immutable reference answers.

## Reduced Winner Population

The completed single-axis pass selected the LE r32 adapters for the reduced follow-up population across the 1B, 3B, and 8B dense bases. The table below shows composite means on the 1-5 scale for the reduced no-RAG and RAG evaluations.

<table width="100%">
<thead>
<tr><th>Corpus</th><th>Target</th><th align="right">No-RAG composite</th><th align="right">RAG composite</th><th align="right">Rows</th></tr>
</thead>
<tbody>
<tr><td>Corpus A</td><td>1B LE r32</td><td align="right">2.461</td><td align="right">3.155</td><td align="right">424</td></tr>
<tr><td>Corpus A</td><td>3B LE r32</td><td align="right">2.768</td><td align="right">3.791</td><td align="right">424</td></tr>
<tr><td>Corpus A</td><td>8B LE r32</td><td align="right">2.948</td><td align="right">3.950</td><td align="right">424</td></tr>
<tr><td>Corpus A</td><td>Llama 3.3 70B base</td><td align="right">2.150</td><td align="right">3.866</td><td align="right">424</td></tr>
<tr><td>Corpus B</td><td>1B LE r32</td><td align="right">3.024</td><td align="right">3.295</td><td align="right">431</td></tr>
<tr><td>Corpus B</td><td>3B LE r32</td><td align="right">3.197</td><td align="right">3.795</td><td align="right">431</td></tr>
<tr><td>Corpus B</td><td>8B LE r32</td><td align="right">3.382</td><td align="right">3.931</td><td align="right">431</td></tr>
<tr><td>Corpus B</td><td>Llama 3.3 70B base</td><td align="right">2.077</td><td align="right">3.854</td><td align="right">431</td></tr>
</tbody>
</table>

<img src="rag-vs-norag-composite.svg" alt="Reduced no-RAG versus RAG composite scores" width="100%">

<img src="rag-vs-norag-axis-heatmap.svg" alt="Reduced no-RAG versus RAG axis heatmap" width="100%">

<img src="golden-eval-score-table.svg" alt="Golden evaluation score table" width="100%">

## Interpretation

The result supports the repo thesis: for small, domain-specific corpora, a provenance-preserving LE extraction path can produce high-utility SFT data for smaller dense LoRA adapters. The point is the repeatable NVAIE-centered representation, not the identity of the example documentation. The 8B LE r32 adapters were the strongest reduced-population LoRA targets in both corpora, and the 3B LE r32 adapters were close enough to be operationally relevant when serving cost or GPU footprint matters.

RAG improved all reduced targets in this retained case study, which is expected: retrieval supplies source-grounded context at answer time. This should be read as a workload-specific result, not as a claim that either LoRA-only or RAG-assisted serving is universally preferable. The no-RAG result is the cleaner measure of what the LoRA adapter itself learned; the RAG result shows the operational upside of pairing that domain-adapted model with retrieved context when freshness, citations, or auditability matter.

## RAGAS Status

RAGAS-style retrieval diagnostics were scored for the reduced RAG population shown above. The retained public artifact records the status of the RAG answer capture and scored RAG evaluation: 3,420 RAG answers across eight reduced result sets, with no unresolved answer-capture or scoring failures.

RAGAS remains a secondary retrieval diagnostic, not the primary LoRA winner criterion. The primary winner is still determined from the no-RAG model-quality evaluation so retrieval does not mask what the adapter learned.

<img src="ragas-coverage-status.svg" alt="RAGAS evaluation status" width="100%">
