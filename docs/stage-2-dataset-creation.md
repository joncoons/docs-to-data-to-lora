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
4. **Stage 1C — Instruction Diversity Pass**: for the highest-density 25% of
   passages, generate SUMMARY / LISTICLE / PROCEDURAL instruction-following
   examples. Output: `stage1c_instruction.jsonl`.
5. **Stage 1.5 — Bias analysis + Data Designer gap-fill**: measure per-product
   KVP density; for under-represented products, run RAG-grounded NeMo Data
   Designer recipes to synthesize additional pairs. Output: `bias_report.json` +
   `stage1_5_gapfill.jsonl`.
6. **Stage 2 — QA Eval Refinement**: super-120b self-eval — refine or drop each
   pair. Output: `stage2_eval.jsonl`.
7. **Stage 3 — NeMo Curator**: exact dedup, MinHash fuzzy dedup, length filter,
   quality filter, train/val split. Outputs: `training.jsonl` + `validation.jsonl`.
8. **Stage 4 — Validation Gate**: an independent external judge spot-checks 100
   pairs per collection on three binary criteria; pipeline passes if grounding
   rate ≥ 90%.

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

- Legacy endpoint: round-robin over `nim-llm-super-120b-bw` pod IPs. Falls back to
  the ClusterIP service if pod discovery fails.
- Legacy model: `nvidia/nemotron-3-super-120b-a12b`
- Legacy temperature: 0.2
- Optional batched defaults: `https://inference-api.nvidia.com/v1`,
  `nvidia/nvidia/nemotron-3-ultra`, temperature 0.95. Override with
  `--stage1a-nim-endpoints`, `--stage1a-model`, `--stage1a-api-key`, and
  `--stage1a-temperature`.
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
4. Call super-120b with the synthesis prompt (BRIDGING + CONTRASTIVE in a single
   request, structured JSON output).
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

### Coverage

Stage 1B runs on **all** Stage 0 passages — full coverage, no subsampling. The
kNN retrieval with `num_candidates=50` is the only meaningful cost; on the two
corpora (500-620 passages each), the full pass completes in ~30-45 min per
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

### Selection

Rank passages by **chunk-index span × unique product-term hit density**:

```
density_score = (max_chunk_index - min_chunk_index + 1)
                × (unique_product_terms / token_count)
```

where `unique_product_terms` is the count of distinct values from
`metadata.product_family` + `metadata.product_name` + anchor-tagged section
headings that appear in the body. The top **25% of passages per collection, with
an absolute floor of 100 passages**, advance to Stage 1C.

The floor prevents small corpora from producing too few instruction-format pairs.
Raw token count is intentionally NOT used here — that heuristic breaks down on
the smaller by-URL passages produced by Stage 0.

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

### Expected yield

- NIM: top 25% of ~500 passages = 125 passages × 2.5 avg types = **~310 pairs**.
- NeMo USvcs: top 25% of ~620 = 155 × 2.5 = **~390 pairs**.

---

## Stage 1.5: Bias Analysis + Data Designer Handoff

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
record instead of directly calling an LLM:

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
After collection, the normal Stage 2 QA/finalization path reads
`stage1_5_gapfill.jsonl` automatically and admits matching
`provenance/data_designer_samples.jsonl` sidecar records by `sample_id`, so the
final `provenance/dataset_samples.jsonl` retains Data Designer job IDs, gap IDs,
seed references, and recipe metadata.

Use `python scripts/build_v2_dataset.py ... --stage1-5-mode legacy-direct` only
when you intentionally want the older direct LLM fallback to emit synthetic rows
locally.

### No-think mode and the May 2026 Jinja2 failure mode

Super-120b is a reasoning model. If `reasoning_effort: "minimal"` (or
`chat_template_kwargs={"enable_thinking": false}` in vLLM 0.7+) is not honored,
the model wraps its response in `<think>...</think>` blocks. When these land
inside a Jinja2 template, the `{{ }}` inside the `<think>` block causes a Jinja2
parse error that silently produces empty output or raises a TemplateError — the
failure mode observed in May 2026 with NeMo Customizer jobs.

**Verification**: before running Stage 1.5, probe the super-120b endpoint:

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

If super-120b cannot be coerced into no-think mode, fall back to:

- **Llama-3.1-8B-Instruct** via `nim-llm-8b-blackwell:8000/v1` (deployed on
  demand from the PEFT cluster), or
- **Llama-3.3-Nemotron-Super-49B-v1.5** NVFP4 TP1 in non-reasoning mode.

Recipe templates are identical; only the `model` field changes. The non-reasoning
fallback loses some synthesis nuance but keeps the pipeline deterministic.

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
- `provenance/dataset_samples.jsonl` is written after Stage 2 QA and overlays
  matching Data Designer sidecar records by `sample_id`, preserving native
  service lineage even when Stage 2 refines the prompt or completion text.

Legacy direct mode still writes `/mnt/nvme2/peft/datasets/v2/<collection>/stage1_5_gapfill.jsonl`
with the Stage 1A-compatible row schema:

- `stage: "1.5"`
- `target_product_family: "..."`
- `retrieved_urls: [...]`

### Expected yield

Highly variable. The NIM corpus is known to be skewed (LLM-NIM ≈ 88% in the
April 2026 analysis); expect **~200-400 gap-fill pairs** rebalancing the
under-represented 30-40 NIM product families. NeMo USvcs is more uniform
(single-prefix crawl); expect **~50-150 pairs**.

---

## Stage 2: QA Eval Refinement

A direct port of `prompt_zoo.qa_eval()` + `QAEvaluation` Pydantic model.

### Per-pair flow

For every pair from Stages 1A + 1B + 1C + 1.5:

1. Call super-120b with the QA-eval prompt (current Q, current A, source
   `context`).
2. Parse `QAEvaluation`:
   - If the model rewrites Q or A → flag `refined: true`, write the new pair.
   - If the model indicates the pair is ungrounded and cannot be repaired from
     context → drop the pair (log to `stage2_dropped.jsonl` for inspection).
   - If unchanged → flag `refined: false`, pass through.

### Output schema

File: `/mnt/nvme2/peft/datasets/v2/<collection>/stage2_eval.jsonl`

Same schema as Stage 1A, `refined` may now be `true`.

Also: `/mnt/nvme2/peft/datasets/v2/<collection>/stage2_dropped.jsonl`
(dropped pairs with drop reason, for post-run inspection).

### Self-eval caveat

Super-120b is judging its own output here — a known agreement bias: the model
tends to approve pairs that share its own generation style, even when they
contain subtle errors. Stage 4 (external judge) compensates by independently
sampling Stage 2 output. Stage 2 stays in the pipeline because it is cheap,
removes obviously broken pairs, and reduces the volume the external judge has to
spot-check.

---

## Stage 3: NeMo Curator

Curator's role in this pipeline is **dedup + quality filtering only**. The
synthetic-augmentation step that originally lived inside Curator has been moved
upstream to Stage 1.5, where it is RAG-grounded. Curator here is a dedup and
quality gate, not a generation step.

### Pipeline steps

The showcase path uses `scripts/pipeline/curator_handoff.py` and
`deploy/curator/` to hand native NeMo Curator a normalized
`curator/input/dataset_samples.jsonl` file. Curator should run exact/fuzzy
deduplication and quality filters in the official Curator container or
Curator-backed cluster, then the collect step maps retained/removed records back
to dataset sample lineage.

1. **Prepare Curator input** from `provenance/dataset_samples.jsonl`.
2. **Native Curator quality filters** using `configs/curator/sft-dedup-quality.yaml`.
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

| Collection | Pre-Curator | Post-Curator | Train (90%) | Val (10%) |
|---|---:|---:|---:|---:|
| `nim_curated` | ~2,360 | ~2,090 | ~1,880 | ~210 |
| `nemo_usvcs_curated` | ~2,660 | ~2,390 | ~2,150 | ~240 |

Pre-Curator totals: Stage 1A (750/930) + 1B at 100% (1,000/1,240) + 1C at 25%
(310/390) + 1.5 gap-fill (~300/~100) ≈ 2,360/2,660. Post-Curator assumes ~10%
loss to exact/MinHash dedup + length filter.

These totals are smaller than the April 2026 7,049-sample dataset because the
source corpora are roughly half the size (2,086 and 1,289 chunks vs. 7,189 in the
original NIM index) and the per-passage multi-premise yield is lower. The trade
is quality (smaller, RAG-grounded, less repetitive) over volume.

---

## Stage 4: External-Judge Validation Gate

### Purpose

Break the closed-loop agreement bias of Stage 2 (super-120b refining
super-120b output) by validating a sample with an independent model that had no
role in generating the pairs.

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
pairs, tighten the Stage 2 QA-eval prompt, and re-run Stage 2 forward.

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
--resume                               skip stages whose output file already exists
--dry-run                              print stage plan + estimated yield, no LLM calls
--max-passages N                       smoke test with N passages
```

Each stage writes its output file and a checkpoint to `<output>/progress.json`.
Resume re-reads `progress.json` and skips already-completed work units (by
`passage_id` or `pair_id`), so interrupted runs pick up where they left off.

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
│   ├── stage1c_instruction.jsonl    ← Stage 1C: instruction diversity
│   ├── bias_report.json             ← Stage 1.5: per-product density
│   ├── stage1_5_gapfill.jsonl       ← Stage 1.5: RAG-grounded gap-fill
│   ├── stage2_eval.jsonl            ← Stage 2: refined pairs
│   ├── stage2_dropped.jsonl         ← Stage 2: dropped pairs (inspection log)
│   ├── training.jsonl               ← Stage 3 output — registered with Customizer
│   ├── validation.jsonl             ← Stage 3 output
│   ├── validation_report.json       ← Stage 4: external-judge report
│   └── progress.json               ← checkpoint for --resume
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
eliminates parametric-only generation: every pair, including Stage 1.5 gap-fill,
uses retrieved chunks as the LLM's sole factual source.

### Chunk grouping by URL, not token count

The April pipeline used 900-token passages produced by a sliding window over
sorted chunks. This had two failure modes: (1) passages occasionally spanned
page boundaries, mixing context from two different topics; (2) token-count
ranking in Stage 1C broke down because all passages had similar token counts.
Grouping by `content_url` aligns passages with source pages, which is what the
model will see at inference time in a RAG setting.

### External judge for closed-loop avoidance

Stage 2 uses super-120b to evaluate pairs that super-120b generated. The
agreement bias is real: models tend to approve output that resembles their own
generation style. Stage 4 breaks this loop by using an independent judge model
that had no role in generating the data as the final gate.
This pattern follows the `[[feedback_external_frontier_judge]]` principle: for
LLM-as-judge tasks, independence over self-contained is the priority.

### 90/10 train/val split

The April pipeline used 95/5. Post-Curator pair counts for these smaller corpora
are ~2,000-2,400, which yields only ~100-120 val pairs at 5% — too few for
stable val_loss curves during LoRA training. 10% gives ~200-240 val pairs, which
is sufficient.

### Data Designer role: gap-fill only, not primary generation

Earlier specs and the April pipeline experimented with NeMo Data Designer as a
primary generation engine for the whole dataset. That approach was abandoned for
two reasons: (1) parametric hallucination risk at scale, and (2) the May 2026
Jinja2/`<think>` failure mode (see Stage 1.5 above). Data Designer is retained
as a targeted gap-fill tool for under-represented products, where its controlled
recipe format helps guarantee consistent output structure. The Stages 1A/1B/1C
LLM calls use direct API calls, not Data Designer.

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
