# Stage 2: Dataset Creation

> **Status**: production methodology. Pipeline implemented in
> `scripts/build_v2_dataset.py`. Stage 1 (curated crawl) provides the input;
> Stage 3 (PEFT training) is the downstream consumer.

## What this stage produces

Two independent runs of the same pipeline — one per Stage 1 ES collection — each
producing a `training.jsonl` + `validation.jsonl` pair in NeMo Customizer SFT
format, split 90/10. Every record has the shape `{prompt, completion, system}`.
The two datasets are entirely independent: no chunks cross between them, and the
resulting LoRA adapters are specialized for their respective domains. The
`nim_curated` collection (2,086 chunks, 50 NIM products) produces a NIM-specific
adapter; the `nemo_usvcs_curated` collection (1,289 chunks, NeMo Microservices
platform) produces a NeMo Microservices-specific adapter.

Every generated pair is grounded in retrieved corpus text. No parametric-only
generation is used anywhere in the pipeline — the LLM synthesizes or transforms
chunks already in the corpus, it does not draw on its own training knowledge.

## Pipeline overview

The pipeline has eight stages, run in order for each collection:

1. **Stage 0 — Corpus prep**: retrieve chunks from ES, group HTML chunks by
   source URL, apply noise filters. Output: `passages.jsonl`.
2. **Stage 1A — Logical Entailment → KVP**: for each passage, extract logical
   entailments and convert premise/conclusion pairs into Q+A. Output:
   `stage1a_le.jsonl`.
3. **Stage 1B — Semantic Neighborhood Synthesis**: for each passage, fetch the
   k-nearest neighbor chunks and generate BRIDGING and CONTRASTIVE questions
   whose answers require synthesizing across passages. Output:
   `stage1b_synthesis.jsonl`.
4. **Stage 1C — Instruction Diversity Pass**: for selected passages, generate
   SUMMARY / LISTICLE / PROCEDURAL instruction-following examples. The default
   selection mode is stratified across density bands; top-density and all-passage
   modes are available. Output: `stage1c_instruction.jsonl`.
5. **Stage 1.5 — Optional Bias analysis + Data Designer gap-fill**: measure
   per-product KVP density and, only when synthetic augmentation is desired,
   hand under-represented products to NeMo Data Designer for grounded synthetic
   generation. Output: `bias_report.json` + optional `stage1_5_gapfill.jsonl`.
6. **Stage 2 — QA Admission + Refinement**: Curator-backed LLM quality gating
   with a frontier-grade model — refine or drop each pair based on source
   grounding and answer fidelity. Output: `stage2_eval.jsonl`.
7. **Stage 3 — NeMo Curator**: exact dedup, MinHash fuzzy dedup, length filter,
   non-LLM quality filters, train/val split. Outputs: `training.jsonl` +
   `validation.jsonl`.
8. **Stage 4 — Validation Gate**: an independent external judge spot-checks 100
   pairs per collection on three binary criteria; pipeline passes if grounding
   rate ≥ 90%.

### Operational levers

The main runner is designed to be runnable as-is, while still exposing the knobs
that materially change dataset coverage, cost, and recovery behavior.

- `--stage` runs one stage or the full pipeline; `--resume` reuses durable stage
  artifacts and skips completed work where the stage supports per-passage
  status. `--max-passages` is the smoke-test lever.
- `--stage1a-mode legacy|batched` chooses the original one-KVP-call-per-premise
  path or the batched KVP expansion path. Both paths extract all entailments and
  all premises; there is no artificial entailment or premise cap.
- `--stage1a-nim-endpoints`, `--stage1a-model`, `--stage1a-temperature`,
  `--stage1a-le-max-tokens`, `--stage1a-batched-kvp-max-tokens`,
  `--stage1a-max-premises-per-batch`, and `--stage1a-batch-parse-attempts`
  control Stage 1A endpoint placement, model selection, recall temperature,
  response budget, batch size, and fallback behavior. The project default for
  Stage 1A is Nemotron 3 Super 120B; Ultra 550B is reserved for downstream audit
  or augmentation unless explicitly requested for an experiment.
- `--stage1c-selection-mode stratified|top_density|all` controls how much
  instruction diversity is added. `stratified` is the default because it keeps
  high-density passages represented without making Stage 1C a narrow
  high-density-only sample.
- Source-kind filtering is optional. The canonical pipeline can use all Stage 0
  passages; experiment runners may expose `--source-doc-kind html` when a run
  intentionally excludes parsed PDFs for publication or comparison.
- Stage 1B and Stage 1C append rows as each passage finishes and write
  `stage1b_passage_results.jsonl` / `stage1c_passage_results.jsonl`. Interrupted
  runs can resume without replaying completed passages or losing already written
  rows.
- Stage 1B and Stage 1C should use a frontier-level synthesis model. The
  current experiments use Nemotron 3 Ultra 550B through NVIDIA-hosted inference,
  but the downstream runner accepts any OpenAI-compatible chat-completions
  endpoint/model pair via repeated `--target ENDPOINT=MODEL[@MAX_CONTEXT]`
  arguments.
- Stage 1.5 is optional. For grounded-only dataset creation, proceed from Stage
  1C directly to Stage 2. Enable Stage 1.5 when the source-grounded set is small,
  product coverage is visibly imbalanced, or a controlled synthetic augmentation
  experiment is explicitly part of the plan. Synthetic generation should use a
  frontier-grade model, not a small local fallback, because it can otherwise
  introduce style drift or shallow paraphrases into the training mix.
- Stage 2 is a required logical quality/admission stage. The preferred execution
  surface is Curator-backed LLM filtering/refinement using a frontier-grade model
  such as Nemotron 3 Ultra 550B. The older direct `qa_eval` prompt path remains
  useful as a local fallback and ablation path, but should not be the canonical
  production posture when Curator LLM quality tooling is available.

---

## Stage 0: Corpus prep

### Chunk grouping rule

For each unique `content_url` in the source collection:

- **HTML/markdown chunks** (detected by URL extension NOT in `{.pdf, .docx,
  .pptx}` AND/OR `document_type == "md"`): sort by `chunk_index`, concatenate
  all `text` fields with a single space separator. The resulting passage has no
  token cap. Carry the first chunk's `vector` as the seed vector for Stage 1B.
- **PDF/binary chunks** (URL extension in `{.pdf, .docx, .pptx}` OR
  `document_type` indicates binary): emit each chunk as its own passage. Carry
  per-chunk `vector` as seed vector. PDF chunks from nemoretriever-parse are
  already semantically split — further grouping reduces signal.

`document_type` and URL extension are cross-checked; if they disagree, log a
warning and use the URL extension as authoritative.

This grouping rule replaces the 900-token passage reconstruction used in the
April 2026 pipeline. Grouping by URL produces passages that exactly match a
source page, which reduces split-context hallucination risk.

### Noise filter

A passage is dropped if any of the following conditions hold:

- Token count (tiktoken `cl100k_base`) < 60.
- First line starts with any of: `apache license`, `mit license`, `copyright
  (c)`, `bsd license`, `gnu general public`, `mozilla public license`, `terms
  and conditions` (case-insensitive).
- URL path contains any of: `acknowledgement`, `third-party`, `open-source`,
  `legal`, `license-notice`.
- Body matches any of (regex, case-insensitive, DOTALL): `apache\.org/licenses/LICENSE-2\.0`,
  `WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND`, `THIS SOFTWARE IS PROVIDED.*?AS IS`,
  `Redistribution and use in source and binary forms`,
  `Permission is hereby granted, free of charge`,
  `under the terms of the GNU`, `Lesser General Public License`,
  `Mozilla Public License`, `creativecommons\.org/licenses`.
- More than 60% of lines are markdown link-list items (regex `^- \[`).
- More than 85% of body chars are inside triple-backtick code fences.

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/passages.jsonl`

```json
{
  "passage_id": "<content_url>#p<n>",
  "url": "<content_url>",
  "text": "<concatenated or chunk text>",
  "token_count": 284,
  "chunk_ids": ["<es_id>", "..."],
  "product_family": "<from metadata>",
  "product_name": "<from metadata>",
  "doc_kind": "html|pdf"
}
```

The seed vector is NOT written to disk (large, only needed at runtime); the script
holds it in memory as `{passage_id: vector}`.

### Expected yield

- `nim_curated` (2,086 chunks across ~485 URLs + 4 PDFs × ~20 chunks each ≈ 80
  PDF chunks): ~485 HTML passages + ~80 PDF passages = ~565 passages before noise
  filter, ~500 after.
- `nemo_usvcs_curated` (1,289 chunks across 648 URLs, no PDFs): ~648 HTML
  passages before filter, ~620 after.

---

## Stage 1A: Logical Entailment → KVP

This stage is a direct port of `prompt_zoo.logical_entailment()` and
`prompt_zoo.kvp_generation()` from the Jan 2025 LE pipeline (see References),
with the Pydantic schemas `LogEntailment` and `QAKeyValuePair`.

### Per-passage flow

1. Call super-120b with the LE prompt; parse `LogEntailment` (conclusion,
   premises[], context, entities, recommendations).
2. If `conclusion` is empty OR `premises` is empty/missing → skip this passage.
3. For each premise: call super-120b with the KVP prompt (premise +
   conclusion + source text); parse `QAKeyValuePair`.
4. Emit one row per successful (premise, conclusion) → (question, answer) pair.

The default production path preserves this one-KVP-call-per-premise behavior. An
optional conservative efficiency path is available with `--stage1a-mode batched`.
It keeps the LE call unchanged, batches only the KVP expansion over parsed
premises, and falls back to the original per-premise KVP prompt for missing or
invalid batched rows. The batched path writes the same canonical outputs:
`stage1a_le.jsonl` and `provenance/entailments.jsonl`.

Example:

```bash
python scripts/build_v2_dataset.py \
  --collection nim_curated \
  --output /mnt/nvme2/peft/datasets/v2/nim_curated \
  --stage 1a \
  --stage1a-mode batched
```

### LLM details

- Default endpoint: the configured Nemotron 3 Super endpoint list from
  `PIPELINE_NIM_ENDPOINTS` / `Config.nim_endpoints`, usually the local
  `nim-llm-super-120b-bw` service. Multiple endpoints are round-robined by the
  shared LLM client.
- Default model: `nvidia/nemotron-3-super-120b-a12b`. The hosted alias
  `nvidia/nvidia/nemotron-3-super-v3` is the same model family for this work and
  can be supplied with `--stage1a-model` when using NVIDIA-hosted inference.
- Legacy temperature: 0.2. Batched mode defaults to 0.95 for higher-recall
  extraction, unless `--stage1a-temperature` overrides it.
- Endpoint/model overrides: use `--stage1a-nim-endpoints`, `--stage1a-model`,
  and `--stage1a-api-key` when an experiment needs hosted inference or a
  local-plus-hosted endpoint mix. Ultra 550B is not the default Stage 1A
  extraction model; use it here only as an explicit experiment.
- Batched KVP controls: `--stage1a-max-premises-per-batch`,
  `--stage1a-batch-parse-attempts`, `--stage1a-le-max-tokens`, and
  `--stage1a-batched-kvp-max-tokens`. The default completion budget is 16,384
  tokens for LE and batched KVP calls; this is an operational budget, not a
  schema cap.
- `MAX_WORKERS=5`, `MIN_INTERVAL=0.5s`, 3 retries with linear backoff (5s × attempt).

### Multi-premise expansion

The LE step extracts all premises needed for each entailment. Each premise
independently supports its entailment's conclusion and generates its own Q+A
pair. This multi-premise expansion is the mechanism that allows a 500-passage
corpus to produce ~750-930 pairs at Stage 1A — ratio of ~1.5 pairs/passage on
the smaller by-URL passages (compared to ~2.5 on the April 2026 900-token
passages, which often spanned multiple logical claims).

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/stage1a_le.jsonl`

```json
{
  "passage_id": "...",
  "source_url": "...",
  "product_family": "...",
  "stage": "1a",
  "premise_index": 1,
  "question": "<derived from premise>",
  "answer": "<derived from conclusion>",
  "context": "<source passage text>",
  "refined": false
}
```

### Expected yield

~500 passages × 1.5 = **~750 NIM pairs**; ~620 × 1.5 = **~930 NeMo USvcs pairs**.

---

## Stage 1B: Semantic Neighborhood Synthesis

### Per-passage flow

1. Take the passage's seed vector (held in memory from Stage 0).
2. ES kNN query: `k=6`, `num_candidates=50`, with a `must_not` filter excluding
   chunks sharing the same `content_url`.
3. Trim neighbors to top-3 by score. Combine seed passage + top-3 neighbor chunks
   into a context capped at ~1,200 tokens.
4. Call the configured frontier-level synthesis model with the synthesis prompt
   (BRIDGING + CONTRASTIVE in a single request, structured JSON output).
5. Emit one row per question (typically 2 per neighborhood).

### Prompts

System:
```
You are a technical documentation analyst. Generate QA pairs as JSON. Return
ONLY valid JSON — no explanation, no markdown fences.
```

User:
```
Given the following related passages from NVIDIA <NIM or NeMo Microservices>
documentation, generate two high-value questions that require synthesizing
information across the passages.

- BRIDGING: a question whose complete answer requires combining a fact from
  passage A with a fact from passage B (e.g. "Given that X requires Y, and Y
  depends on Z, what must be true when configuring...?")
- CONTRASTIVE: a question that asks how two related concepts, models, or
  configurations differ (e.g. "How does deploying NIM on Kubernetes differ
  from bare-metal deployment?")

Rules:
- Questions must be answerable ONLY from the provided passages — no outside
  knowledge
- Answers must be complete sentences, 1-3 sentences max
- Do not fabricate facts not present in the passages

Passages:
{neighborhood_context}

Output JSON:
{
  "pairs": [
    {"type": "bridging",    "question": "...", "answer": "..."},
    {"type": "contrastive", "question": "...", "answer": "..."}
  ]
}
```

### Model and endpoint guidance

Stage 1B is a synthesis step, not raw entailment extraction. Use a
frontier-level instruction/reasoning model for this stage so cross-passage
bridging and contrastive questions are not bottlenecked by a smaller local model.
The current experiment uses `nvidia/nvidia/nemotron-3-ultra`, but the downstream
runner is model-agnostic as long as the endpoint implements OpenAI-compatible
chat completions. Use repeated `--target ENDPOINT=MODEL[@MAX_CONTEXT]` arguments
to choose one or more endpoints; non-NVIDIA secured endpoints can use
`--api-key`, and NVIDIA-hosted inference reads the configured Kubernetes secret.

### Coverage

Stage 1B runs on **all** Stage 0 passages selected for the run - full
coverage, no subsampling. The kNN retrieval with `num_candidates=50` is scoped
to the active `--collection`, so NIM passages retrieve NIM neighbors and NeMo
Microservices passages retrieve NeMo Microservices neighbors.

Rows are appended to `stage1b_synthesis.jsonl` as each passage finishes.
`stage1b_passage_results.jsonl` records per-passage status, row count, source
URL, finish time, and any exception text. With `--resume`, passages that already
have persisted rows, no seed vector, or no neighbors are skipped; transient
exceptions remain retryable. This avoids holding the whole stage in memory and
prevents an interrupted run from discarding completed work.

The kNN retrieval with `num_candidates=50` is the only meaningful cost; on the
two corpora (500-620 passages each), the full pass completes in ~30-45 min per
collection.

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/stage1b_synthesis.jsonl`

Same schema as Stage 1A, with:
- `stage: "1b"`
- `qa_type: "bridging"|"contrastive"`
- `neighbor_urls: [...]`

### Expected yield

- NIM: ~500 passages × 2 pairs = **~1,000 pairs**.
- NeMo USvcs: ~620 × 2 = **~1,240 pairs**.

---

## Stage 1C: Instruction Diversity Pass

### Model and endpoint guidance

Stage 1C should use the same frontier-level model posture as Stage 1B. Its job
is to rewrite grounded passage content into diverse instruction formats, so the
model should be strong enough to preserve factual boundaries while changing task
shape. The active downstream runner uses the same `--target
ENDPOINT=MODEL[@MAX_CONTEXT]` mechanism described for Stage 1B; examples may
name Nemotron 3 Ultra, but any OpenAI-compatible endpoint/model pair can be
used when it meets the quality bar.

### Selection

Stage 1C uses the same density score as the earlier high-density-only design:

```
density_score = (max_chunk_index - min_chunk_index + 1)
                × (unique_product_terms / token_count)
```

where `unique_product_terms` is the count of distinct values from
`metadata.product_family` + `metadata.product_name` + anchor-tagged section
headings that appear in the body. Raw token count is intentionally NOT used as
the primary heuristic because it breaks down on the smaller by-URL passages
produced by Stage 0.

The default selection mode is `stratified`: compute a target count using
`max(stage1c_min_passages, int(stage1c_top_percent * passage_count))`, cap it at
the corpus size, then select across density bands. This keeps dense technical
pages in the sample while allowing lower-density but still useful documentation
pages to contribute summary/list/procedure behavior.

Selection is controlled with `--stage1c-selection-mode` or
`PIPELINE_STAGE1C_SELECTION_MODE`:

- `stratified` - default; span density bands up to the configured target count.
- `top_density` - legacy behavior; choose the highest-density passages only.
- `all` - run instruction generation for every selected Stage 0 passage.

The floor prevents small corpora from producing too few instruction-format pairs.
`stage1c_top_percent` remains the cost-control knob, and `stage1c_min_passages`
remains the minimum-coverage knob for non-`all` modes.

### Prompts

System:
```
You are a technical writer. Generate instruction-following training examples as
JSON. Return ONLY valid JSON.
```

User:
```
Given the following <NIM | NeMo Microservices> documentation passage, generate
three instruction-following training examples in different formats:

1. SUMMARY:    "Summarize the key points of [topic] from the following..."
2. LISTICLE:   "List all [configuration options / prerequisites / steps] for..."
3. PROCEDURAL: "Provide step-by-step instructions for..."
   (Omit type 3 if the passage is not procedural — no numbered steps, no
   command sequence.)

Rules:
- Answer only from the provided text — no outside knowledge.
- Each answer 50-200 tokens.

Passage:
{passage_text}

Output JSON:
{
  "pairs": [
    {"type": "summary",    "question": "...", "answer": "..."},
    {"type": "listicle",   "question": "...", "answer": "..."},
    {"type": "procedural", "question": "...", "answer": "..."}    // optional
  ]
}
```

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/stage1c_instruction.jsonl`

Same schema as Stage 1A, with:
- `stage: "1c"`
- `instr_type: "summary"|"listicle"|"procedural"`

Rows are appended as each passage finishes. `stage1c_passage_results.jsonl`
records per-passage status and row count. With `--resume`, passages with
existing rows or terminal no-work statuses (`no_pairs`, `no_valid_pairs`) are
skipped; passages with transient exceptions remain retryable.

### Expected yield

Yield depends on `--stage1c-selection-mode`:

- `stratified` default: target count is max(25% of passages, 100), capped by
  corpus size, times ~2.5 instruction rows per passage.
- `top_density`: same target count, but concentrated in the highest-density
  passages.
- `all`: every selected Stage 0 passage, times ~2.5 instruction rows per
  passage.

For the default `stratified` mode, approximate yields are **~310 NIM pairs** and
**~390 NeMo USvcs pairs**.

---

## Stage 1.5: Optional Bias Analysis + Data Designer Handoff

Stage 1.5 is an optional synthetic-augmentation stage, not a required step for
every dataset. The default grounded path for the current LE rerun is to skip it
and continue from Stage 1C to Stage 2. Use it when the grounded dataset is too
small for the target adapter, when product-family coverage is materially
imbalanced, or when the experiment is explicitly testing synthetic-data lift.

When Stage 1.5 performs synthetic generation, use a frontier-grade model through
Data Designer or an equivalent OpenAI-compatible endpoint. The generator should
be strong enough to preserve source constraints while creating genuinely useful
new questions, not just superficial paraphrases. Smaller models can be useful
for smoke testing the handoff mechanics, but they should not be the production
synthetic generator.

### Bias analysis

After Stages 1A/1B/1C complete:

1. Merge all generated pairs (`stage1a_le.jsonl` + `stage1b_synthesis.jsonl` +
   `stage1c_instruction.jsonl`) into a working pool.
2. Aggregate counts per `product_family`:
   - `chunk_count[p]` — number of source chunks in the collection with
     `product_family == p`
   - `kvp_count[p]` — number of generated pairs derived from passages with
     `product_family == p`
3. Compute density: `density[p] = kvp_count[p] / chunk_count[p]`
4. Compute the median density across all products present in the collection.
5. Flag any product where `density[p] < median × 0.5` as under-represented.

The bias signal comes from the `product_family` / `product_name` metadata that
`CRAWLER_PRODUCT_URL_MAP` already attaches to every chunk during ingest — there
are no hand-curated keyword lists involved.

Output files:

- `/mnt/nvme2/peft/datasets/v2/<collection>/bias_report.json`
- `/mnt/nvme2/peft/datasets/v2/<collection>/provenance/gap_manifest.json`
- `/mnt/nvme2/peft/datasets/v2/<collection>/data_designer/gapfill_requests.jsonl`
- `/mnt/nvme2/peft/datasets/v2/<collection>/data_designer/request_manifest.json`

```json
{
  "collection": "<name>",
  "median_density": 1.42,
  "threshold": 0.71,
  "products": [
    {
      "product_family": "...",
      "chunk_count": 45,
      "kvp_count": 12,
      "density": 0.27,
      "underrepresented": true
    }
  ]
}
```

### Gap-fill via NeMo Data Designer

Stage 1.5 now stops at a native handoff boundary by default. For each
under-represented product `T` it writes a gap record and a Data Designer seed
record instead of directly calling an LLM. If generation is enabled, configure
Data Designer with a frontier-grade model and preserve all synthetic lineage:

1. **Gap manifest**: `provenance/gap_manifest.json` records `gap_id`, coverage
   dimension, observed count, target count, severity, seed entailment IDs, seed
   chunk IDs, and a generation brief.
2. **ES retrieval (same collection only)**: retrieve top documentation chunks
   for `product_family == T` and embed the retrieved text plus source URLs into
   `data_designer/gapfill_requests.jsonl`.
3. **Data Designer recipe**: one recipe per collection consumes the seed record
   fields: `gap_id`, `retrieved_chunks`, `product_family`, `pairs_count`,
   `seed_styles`, and `generation_brief`.
4. **Target**: bring each under-represented product up to ≥ `median × 0.8`.
   Stage 1.5 computes pairs-needed and number of requested seed records; the
   native Data Designer submission/result-collection Job owns generation.

The K8s-native handoff job is `deploy/data-designer-gapfill/`. Its `prepare`
mode creates `data_designer/seed_dataset.csv` and `submission_plan.json`; its
`collect` mode converts Data Designer result records into `stage1_5_gapfill.jsonl`,
`data_designer/generated_samples.jsonl`, and `provenance/data_designer_samples.jsonl`.
After collection, the normal Stage 2 QA admission/finalization path reads
`stage1_5_gapfill.jsonl` automatically and admits matching
`provenance/data_designer_samples.jsonl` sidecar records by `sample_id`, so the
final `provenance/dataset_samples.jsonl` retains Data Designer job IDs, gap IDs,
seed references, and recipe metadata.

Use `python scripts/build_v2_dataset.py ... --stage1-5-mode legacy-direct` only
when you intentionally want the older direct LLM fallback to emit synthetic rows
locally. This mode is for controlled experiments or fallback diagnostics; the
preferred production path is Data Designer handoff with a frontier-grade
generator.

### No-think mode and the May 2026 Jinja2 failure mode

Many frontier-grade models are reasoning models. If `reasoning_effort:
"minimal"` (or `chat_template_kwargs={"enable_thinking": false}` in vLLM 0.7+)
is not honored, the model may wrap its response in `<think>...</think>` blocks.
When these land inside a Jinja2 template, the `{{ }}` inside the `<think>` block
causes a Jinja2 parse error that silently produces empty output or raises a
TemplateError — the failure mode observed in May 2026 with NeMo Customizer jobs.

**Verification**: before running Stage 1.5 generation, probe the configured
frontier endpoint, for example:

```bash
curl -s -X POST http://nim-llm-super-120b-bw:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "reasoning_effort": "minimal"
  }' | jq '.choices[0].message.content'
```

If the response contains `<think>...</think>`, no-think mode is not working.

### Fallback path

If the chosen frontier-grade generator cannot be coerced into no-think mode,
prefer switching to another frontier-grade model or provider that can produce
clean structured output. Smaller non-reasoning models may be acceptable for
smoke tests of the Data Designer handoff, but they should not be used as the
production synthetic generator unless an experiment is explicitly measuring that
tradeoff. Recipe templates are identical; only the `model` and provider fields
change.

### Output schema

Default handoff files:

- `provenance/gap_manifest.json` follows `schemas/provenance/gap_manifest.schema.json`.
- `data_designer/gapfill_requests.jsonl` contains one seed record per gap, with
  retrieved chunks, source URLs, seed IDs, and requested pair counts.
- `data_designer/seed_dataset.csv` and `data_designer/submission_plan.json` are
  written by the Data Designer gap-fill Job before native submission.
- `stage1_5_gapfill.jsonl` is written empty by Stage 1.5 to make resume behavior
  explicit; the Data Designer collect path overwrites it with normalized
  synthetic rows after generation.
- `provenance/data_designer_samples.jsonl` carries the exact Data Designer job,
  gap, seed, and recipe lineage for those rows.
- `provenance/dataset_samples.jsonl` is written after Stage 2 QA admission and
  overlays matching Data Designer sidecar records by `sample_id`, preserving
  native service lineage even when Stage 2 refines the prompt or completion
  text.

Legacy direct mode still writes `/mnt/nvme2/peft/datasets/v2/<collection>/stage1_5_gapfill.jsonl`
with the Stage 1A-compatible row schema:

- `stage: "1.5"`
- `target_product_family: "..."`
- `retrieved_urls: [...]`

### Expected yield

Optional and highly variable. For grounded-only runs, Stage 1.5 yield is **0**
by design. When synthetic augmentation is enabled, the NIM corpus is known to be
skewed (LLM-NIM ≈ 88% in the April 2026 analysis); expect **~200-400 gap-fill
pairs** rebalancing the under-represented 30-40 NIM product families. NeMo USvcs
is more uniform (single-prefix crawl); expect **~50-150 pairs**. For very small
grounded datasets, a larger controlled synthetic ratio may be useful, but keep
validation/test splits grounded and unchanged.

---

## Stage 2: QA Admission + Refinement

This stage is the semantic quality gate for generated Q+A pairs. It is not
optional, but the preferred implementation should move through Curator-backed
LLM quality filtering/refinement rather than a standalone self-eval script. The
direct `prompt_zoo.qa_eval()` + `QAEvaluation` path remains a compatibility
fallback and an ablation tool.

Operationally, Stage 2 earns its place before Stage 3 because it prevents raw
LLM-generated rows from becoming curation inputs before they have been judged
against their source context. Stage 2 handles semantic admission: repair a weak
but source-grounded pair, drop an irreparable or ungrounded pair, and write an
auditable quality decision. Stage 3 then performs structural curation over the
admitted set: deduplication, heuristic quality filters, and train/validation
splitting. Keeping this boundary avoids three failure modes:

- Curator deduplication can preserve a polished but ungrounded answer if no
  earlier semantic gate rejects it.
- Train/validation splits become polluted if bad rows are only identified after
  splitting, because removing them later changes split composition and lineage.
- Expensive native Curator work is wasted on rows that a frontier QA gate could
  have repaired or rejected first.

The net effect is a cleaner Stage 3 input contract: every row handed to Curator
has already been admitted by a source-grounded QA gate, and every rejection has a
local reason in `stage2_dropped.jsonl` plus `provenance/stage2_quality.jsonl`.

### Per-pair flow

For every pair from Stages 1A + 1B + 1C plus optional Stage 1.5 synthetic rows:

1. Call the configured frontier-grade QA model with the QA-eval prompt (current
   Q, current A, source `context`). For the current NIM/NeMo experiments this
   should be Nemotron 3 Ultra 550B through an OpenAI-compatible endpoint.
2. Parse `QAEvaluation`:
   - If the model rewrites Q or A → flag `refined: true`, write the new pair.
   - If the model indicates the pair is ungrounded and cannot be repaired from
     context → drop the pair (log to `stage2_dropped.jsonl` for inspection).
   - If unchanged → flag `refined: false`, pass through.
3. Preserve Curator or direct-run quality metadata with each admitted row so
   later dedup, train/val splitting, and audit reports can attribute why a pair
   was retained or rejected. The durable runner appends each admitted row and
   each quality decision as work finishes so interrupted frontier-model runs can
   resume without replaying completed rows.

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/stage2_eval.jsonl`

Same schema as Stage 1A, `refined` may now be `true`.

Also:

- `/mnt/nvme2/peft/datasets/v2/<collection>/stage2_dropped.jsonl` — rejected or
  retryable failed pairs with QA status and reason.
- `/mnt/nvme2/peft/datasets/v2/<collection>/provenance/stage2_quality.jsonl` —
  one append-only decision record per attempted row, including judge model,
  endpoint labels, admission status, grounding flags, repairability, and reason.

### Curator migration posture

Generic Curator filters are not a substitute for source-grounded QA checks by
themselves. The migrated Stage 2 gate should explicitly score grounding, answer
fidelity, hallucination risk, repairability, and schema validity. Curator should
be the execution surface where possible because it centralizes quality metadata,
rejection lineage, and downstream curation handoff. The direct Python QA runner
should remain available for smoke tests, fallback execution, and A/B comparison
against native Curator quality results.

From an operator's perspective, Stage 2 is the point where model choice matters
most for dataset integrity. It should use a frontier-grade model such as
Nemotron 3 Ultra 550B, low temperature, no-think/structured-output controls when
needed, and durable resume settings appropriate for hosted endpoint rate limits.
Stage 3 should not be asked to infer source grounding from generic text-quality
signals; it should consume the Stage 2-admitted rows and the Stage 2 quality
sidecar.

### Agreement-bias caveat

Even with a frontier-grade model, Stage 2 can still inherit agreement bias if the
same model family generated a substantial share of the rows being judged. Stage
4 compensates by independently sampling post-Curator output with a separate
judge model. If Stage 2 uses Nemotron 3 Ultra, Stage 4 should use a different
frontier-level judge such as Claude Sonnet 4.6 through the NVIDIA-hosted
OpenAI-compatible endpoint.

---

## Stage 3: NeMo Curator

After Stage 2 QA admission, Curator's Stage 3 role is **dedup + structural
quality filtering + train/val splitting**. Synthetic augmentation is no longer a
Curator responsibility; when enabled, it is handled as optional Stage 1.5
RAG-grounded Data Designer handoff before Stage 2. LLM-based QA/refinement is
treated as the Stage 2 logical gate even when Curator is the native execution
surface.

### Pipeline steps

The showcase path uses `scripts/pipeline/curator_handoff.py` and
`deploy/curator/` to hand native NeMo Curator a normalized
`curator/input/dataset_samples.jsonl` file. Curator should run exact/fuzzy
deduplication and non-LLM quality filters in the official Curator container or
Curator-backed cluster, then the collect step maps retained/removed records back
to dataset sample lineage.

1. **Prepare Curator input** from `provenance/dataset_samples.jsonl`.
2. **Native Curator non-LLM quality filters** using `configs/curator/sft-dedup-quality.yaml`.
3. **Native Curator deduplication** for exact and fuzzy duplicates; semantic dedup
   remains disabled until the embedding model/GPU budget is selected.
4. **Collect Curator outputs** into `curator/accepted_samples.jsonl`,
   `curator/rejected_samples.jsonl`, and `curator/curation_manifest.json`.
5. **Train/val split**: 90/10 random stratified by sample origin and task type.

The older pure-Python `scripts/pipeline/stage3_curator.py` path remains a local
offline fallback for exact dedup, MinHash, token length filters, substring
checks, and split writing.

The 10% val split (vs. the April-era 5%) is intentional: the smaller
post-Curator counts (~2,000-2,400 per collection) would yield only ~100-120 val
pairs at 5% — too few for stable val_loss during LoRA training.

### Output format

Files:
- `/mnt/nvme2/peft/datasets/v2/<collection>/curator/input/dataset_samples.jsonl`
- `/mnt/nvme2/peft/datasets/v2/<collection>/curator/accepted_samples.jsonl`
- `/mnt/nvme2/peft/datasets/v2/<collection>/curator/rejected_samples.jsonl`
- `/mnt/nvme2/peft/datasets/v2/<collection>/curator/curation_manifest.json`
- `/mnt/nvme2/peft/datasets/v2/<collection>/training.jsonl`
- `/mnt/nvme2/peft/datasets/v2/<collection>/validation.jsonl`

Format (NeMo Customizer SFT convention):

```json
{
  "prompt": "<question>",
  "completion": "<answer>",
  "system": "You are a precise NVIDIA <NIM | NeMo Microservices> technical assistant. Answer based on official documentation."
}
```

### Expected final yield

| Collection | Pre-Curator grounded-only | Post-Curator | Train (90%) | Val (10%) |
|---|---:|---:|---:|---:|
| `nim_curated` | ~2,060 | ~1,850 | ~1,665 | ~185 |
| `nemo_usvcs_curated` | ~2,560 | ~2,300 | ~2,070 | ~230 |

Pre-Curator grounded-only totals: Stage 1A (750/930) + 1B at 100%
(1,000/1,240) + 1C default stratified target (310/390) ≈ 2,060/2,560. Optional
Stage 1.5 synthetic gap-fill can add roughly ~300/~100 rows when enabled.
Post-Curator assumes ~10% loss to exact/MinHash dedup + length filter.

These totals are smaller than the April 2026 7,049-sample dataset because the
source corpora are roughly half the size (2,086 and 1,289 chunks vs. 7,189 in the
original NIM index) and the per-passage multi-premise yield is lower. The trade
is quality (smaller, RAG-grounded, less repetitive) over volume.

---

## Stage 4: External-Judge Validation Gate

### Purpose

Break the closed-loop agreement bias of Stage 2 by validating a sample with an
independent model that had no role in generating, refining, or admitting the
pairs.

### Method

1. Random stratified sample of 100 pairs per collection from Stage 3
   `training.jsonl` (stratified by `stage` so all generation strategies are
   represented).
2. For each sampled pair, call the configured independent judge endpoint with
   the validation prompt. API key material is read from a Kubernetes Secret.
3. Score each pair on three binary criteria:
   - **Grounded**: every factual claim in the answer is supported by the source
     `context`.
   - **Answer-fidelity**: the answer directly addresses the question asked.
   - **No-hallucination**: the answer contains no entities, version numbers, or
     commands not present in the source `context`.
4. Aggregate: per-collection grounding rate = (# pairs passing all three) / 100.

### Threshold

**Pass gate if grounding rate ≥ 90%.** Fail otherwise — investigate the failing
pairs, tighten the Stage 2 QA/Curator quality prompt or filters, and re-run
Stage 2 forward.

The 90% threshold is a deliberate underrun of perfect: some pairs may be
factually grounded but ambiguously phrased, and the external judge may conservatively
flag them. A floor of 90% leaves room for that noise while still rejecting
systematically hallucinated batches.

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/validation_report.json`

```json
{
  "collection": "nim_curated",
  "sample_size": 100,
  "grounded_count": 94,
  "answer_fidelity_count": 96,
  "no_hallucination_count": 95,
  "all_three_count": 93,
  "pass_rate": 0.93,
  "threshold": 0.9,
  "passed": true,
  "failures": [
    {"pair_idx": 12, "reason": "answer references version number not in context"},
    ...
  ]
}
```

---

## Reproduction guide

How to point the pipeline at a new ES collection:

1. Add the collection name to the CLI's `--collection` choices in
   `scripts/build_v2_dataset.py`.
2. Update the domain label mapping (NIM → NeMo MS → your-domain) — this
   controls the `system` prompt in the final JSONL.
3. Confirm chunk metadata fields match the required schema: `content_url`,
   `chunk_index`, `document_type`, `product_family`, `vector` (1024-dim
   dense_vector).
4. Run:
   ```bash
   python scripts/build_v2_dataset.py \
     --collection <your-collection> \
     --output /mnt/nvme2/peft/datasets/v2/<name>
   ```

Supported flags:

```
--stage [0|1a|1b|1c|1.5|2|3|4|all]   run a single stage or all in sequence
                                       use explicit stages to skip optional 1.5
--resume                               reuse durable outputs and stage progress
--dry-run                              print stage plan + estimated yield, no LLM calls
--max-passages N                       smoke test with N passages
--stage1a-mode legacy|batched          choose per-premise or batched KVP expansion
--stage1c-selection-mode MODE          stratified, top_density, or all
--stage2-qa-endpoints URLS             comma-separated OpenAI-compatible QA endpoints
--stage2-qa-model MODEL                frontier QA admission model
--stage2-qa-max-tokens N               QA admission completion budget
--stage2-execution-surface LABEL       audit label, e.g. curator_llm_quality
```

Each stage writes its output file and a checkpoint to `<output>/progress.json`.
Stages with per-passage status files re-read those files on resume and skip
already-completed work units, so interrupted runs pick up where they left off
without replaying completed passages.

### Two parallel runs

The two collections are entirely independent — no shared state. They can be run
sequentially or in parallel; the cluster has the GPU budget either way.

```bash
# Sequential
python scripts/build_v2_dataset.py \
  --collection nim_curated \
  --output /mnt/nvme2/peft/datasets/v2/nim/

python scripts/build_v2_dataset.py \
  --collection nemo_usvcs_curated \
  --output /mnt/nvme2/peft/datasets/v2/nemo/

# Or in parallel (background)
python scripts/build_v2_dataset.py \
  --collection nim_curated \
  --output /mnt/nvme2/peft/datasets/v2/nim/ &

python scripts/build_v2_dataset.py \
  --collection nemo_usvcs_curated \
  --output /mnt/nvme2/peft/datasets/v2/nemo/ &
wait
```

---

## File layout

```
/mnt/nvme2/peft/datasets/v2/
├── nim/
│   ├── passages.jsonl               ← Stage 0 output
│   ├── stage1a_le.jsonl             ← Stage 1A: LE → KVP
│   ├── stage1b_synthesis.jsonl      ← Stage 1B: kNN synthesis
│   ├── stage1b_passage_results.jsonl  ← Stage 1B: per-passage durable status
│   ├── stage1c_instruction.jsonl    ← Stage 1C: instruction diversity
│   ├── stage1c_passage_results.jsonl  ← Stage 1C: per-passage durable status
│   ├── bias_report.json             ← Optional Stage 1.5: per-product density
│   ├── stage1_5_gapfill.jsonl       ← Optional Stage 1.5: RAG-grounded gap-fill
│   ├── stage2_eval.jsonl            ← Stage 2: refined pairs
│   ├── stage2_dropped.jsonl         ← Stage 2: dropped pairs (inspection log)
│   ├── training.jsonl               ← Stage 3 output — registered with Customizer
│   ├── validation.jsonl             ← Stage 3 output
│   ├── validation_report.json       ← Stage 4: external-judge report
│   └── progress.json                ← checkpoint for --resume
└── nemo/
    └── (same structure)
```

The terminal artifact of this pipeline is the `training.jsonl` + `validation.jsonl`
pair per collection. Stage 3 (PEFT training) handles Customizer dataset
registration, training-pod paths, and adapter naming from that point forward.

---

## Key design decisions

### Per-collection scoping

Each ES collection gets its own independent pipeline run and produces its own
adapter. No mixing of NIM chunks into the NeMo Microservices dataset, and vice
versa. This keeps each adapter specialized and prevents cross-product
hallucination where the model blends product details from two domains.

### All generation is RAG-grounded

The April 2026 pipeline had a gap-fill step (NeMo Data Designer or Curator's
synthetic-gen) that called the LLM without grounding it in retrieved corpus
chunks. Super-120b has weak parametric knowledge of NIM/NeMo product minutiae —
exactly the topics that most need accurate training signal. This pipeline
eliminates parametric-only generation: every pair, including optional Stage
1.5 gap-fill when enabled, uses retrieved chunks as the LLM's sole factual
source.

### Chunk grouping by URL, not token count

The April pipeline used 900-token passages produced by a sliding window over
sorted chunks. This had two failure modes: (1) passages occasionally spanned
page boundaries, mixing context from two different topics; (2) token-count
ranking in Stage 1C broke down because all passages had similar token counts.
Grouping by `content_url` aligns passages with source pages, which is what the
model will see at inference time in a RAG setting.

### External judge for closed-loop avoidance

Stage 2 uses a frontier-grade model to evaluate and refine rows generated by the
LE/synthesis stages. Agreement bias is still real when the QA model resembles or
matches the generation model: models tend to approve output that resembles their
own generation style. Stage 4 breaks this loop by using an independent judge
model that had no role in generating or admitting the data as the final gate.
This pattern follows the `[[feedback_external_frontier_judge]]` principle: for
LLM-as-judge tasks, independence over self-contained is the priority.

### 90/10 train/val split

The April pipeline used 95/5. Post-Curator pair counts for these smaller corpora
are ~2,000-2,400, which yields only ~100-120 val pairs at 5% — too few for
stable val_loss curves during LoRA training. 10% gives ~200-240 val pairs, which
is sufficient.

### Durable recovery as a first-class control

The generation-heavy stages are expected to run against local or hosted NIM
endpoints where transient failures, rate limits, and user interruptions are
normal operational events. Stage 1B and Stage 1C therefore append completed rows
immediately and record per-passage status. The net effect is that retry policy,
endpoint fanout, and `--resume` can be used as operational controls instead of
requiring a full rerun after every interruption.

### Data Designer role: gap-fill only, not primary generation

Earlier specs and the April pipeline experimented with NeMo Data Designer as a
primary generation engine for the whole dataset. That approach was abandoned for
two reasons: (1) parametric hallucination risk at scale, and (2) the May 2026
Jinja2/`<think>` failure mode (see Stage 1.5 above). Data Designer is retained
as an optional targeted gap-fill tool for under-represented products or small
grounded datasets, where its controlled recipe format helps guarantee consistent
output structure. When used, it should be backed by a frontier-grade generator.
The Stages 1A/1B/1C LLM calls use direct API calls, not Data Designer.

---

## References

### Historical Inputs

- Earlier local pipeline and methodology notes informed the first version of
  this stage. The durable implementation for this repository is
  `scripts/build_v2_dataset.py`.
- Archive: `prompt_zoo.py` + `pydantic_models.py` in the `archive/` directory
  of this repository (provenance for the Jan 2025 LE/KVP/QA-eval prompts; not
  redistributed here, original path:
  `/mnt/nvme3/code_repo/LLM_Demos/NIM_CUDA_X_QA_public_website/archive/le_artifacts/`).
- `scripts/build_v2_dataset.py` — this pipeline's implementation.
- `data-designer-recipes/nim-gapfill.yaml` — Stage 1.5 Data Designer recipe for
  NIM gap-fill.
- `data-designer-recipes/nemo-usvcs-gapfill.yaml` — Stage 1.5 recipe for NeMo
  Microservices gap-fill.

### NeMo Microservices documentation

- NeMo Data Designer:
  `https://docs.nvidia.com/nemo/microservices/latest/set-up/data-designer.html`
- NeMo Curator:
  `https://docs.nvidia.com/nemo/microservices/latest/fine-tune/curator.html`
- NeMo Customizer (SFT format reference):
  `https://docs.nvidia.com/nemo/microservices/latest/fine-tune/customizer.html`

### Related docs in this repo

- Stage 1 input: [`docs/stage-1-curated-crawl.md`](stage-1-curated-crawl.md) —
  the curated crawl methodology that produces the ES collections this pipeline
  reads.
- Stage 3 consumer: [`docs/stage-3-peft-training.md`](stage-3-peft-training.md)
  — LoRA training pipeline that registers and trains on the `training.jsonl` /
  `validation.jsonl` pairs this pipeline produces.
- NIM worked example: [`examples/nim.md`](../examples/nim.md)
- NeMo Microservices worked example:
  [`examples/nemo-microservices.md`](../examples/nemo-microservices.md)
