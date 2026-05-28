# Provenance and Delta Update Contract

This contract defines the durable metadata model for the docs-to-data-to-LoRA
pipeline. The crawler and entailment extractor are intentional custom
components, but their outputs must be versioned and auditable enough to support
incremental recrawls, dataset rebuilds, and NVIDIA-native downstream services.

The guiding rule is that source-grounded data and synthetic gap-fill data are
different lineages. They can be blended for training, but the blend must be a
derived dataset with explicit provenance.

## Goals

- Preserve row-level evidence for every source-grounded sample.
- Support cheap delta updates when a URL changes, disappears, or is added.
- Keep previous source revisions and entailments immutable.
- Track which prompt, model, chunker, validator, and curation versions produced
  each artifact.
- Let NeMo Data Designer generate targeted synthetic data from a gap manifest
  without overwriting the source-grounded reference dataset.
- Register dataset versions in NeMo Entity Store/Data Store with enough
  high-level metadata to find the detailed ledger artifacts.

## Non-Goals

- Do not treat NeMo Entity Store custom fields as the row-level provenance
  store. Entity custom fields are useful pointers and summary metadata, not the
  full ledger.
- Do not silently mutate existing dataset rows after a recrawl. New source
  evidence creates new revisions and new derived dataset versions.
- Do not mix source-grounded and synthetic rows without an explicit blend
  manifest.

## Artifact Graph

```text
crawl_run
  -> source_revision
    -> source_chunk
      -> entailment
        -> dataset_sample
          -> dataset_version_manifest

gap_manifest
  -> data_designer_job
    -> synthetic dataset_sample
      -> dataset_version_manifest
```

## Artifact Types

### crawl_run

A `crawl_run` records one execution of the crawler over a curated URL scope.
It is the root audit object for source-grounded data.

Required fields are defined in
[`crawl_run.schema.json`](../../schemas/provenance/crawl_run.schema.json).

Key requirements:

- Record crawler code version and runtime configuration hash.
- Record the curated seed URL set hash, not just individual URLs.
- Record the reason for the run: initial crawl, scheduled recrawl, manual
  recrawl, hotfix, or backfill.
- Record the previous crawl run when the run is intended to produce a delta.

### source_revision

A `source_revision` is an immutable observation of a canonical URL at a point
in time.

Required fields are defined in
[`source_revision.schema.json`](../../schemas/provenance/source_revision.schema.json).

Key requirements:

- Keep both raw and normalized hashes.
- Preserve HTTP freshness metadata when available: ETag, Last-Modified, cache
  control, and final URL after redirects.
- Link to the previous revision for the same canonical URL.
- Track whether the revision is active, unchanged, changed, deleted, redirected,
  failed, or excluded.
- Store a pointer to raw content and normalized content when retained.

### source_chunk

A `source_chunk` is a stable extraction unit derived from a source revision.
Chunks are the evidence substrate for entailments.

Required fields are defined in
[`source_chunk.schema.json`](../../schemas/provenance/source_chunk.schema.json).

Key requirements:

- Include chunker version and extraction method.
- Preserve structural anchors such as heading path, DOM path, markdown heading,
  page number, or binary parser element ID when available.
- Preserve character or byte spans when available.
- Include a content hash for exact matching across recrawls.
- Keep enough text for local validation, unless storage policy requires a
  content pointer instead.

### entailment

An `entailment` is a source-grounded claim extracted from one or more source
chunks.

Required fields are defined in
[`entailment.schema.json`](../../schemas/provenance/entailment.schema.json).

Key requirements:

- Evidence must point to source chunk IDs and spans.
- Store extractor model, extractor prompt hash, validator model, validator
  prompt hash, and scores.
- Use lifecycle status instead of mutation: active, superseded, retracted,
  rejected, or needs_review.
- Record supersession links when a changed source revision replaces an older
  claim.

### dataset_sample

A `dataset_sample` is the training or evaluation row derived from source
entailments or synthetic generation.

Required fields are defined in
[`dataset_sample.schema.json`](../../schemas/provenance/dataset_sample.schema.json).

Key requirements:

- Set `origin` to `source_entailed`, `synthetic_gapfill`, `human_reviewed`, or
  `imported_baseline`.
- Source-grounded samples must reference entailment IDs and source revisions.
- Synthetic samples must reference a gap ID and Data Designer job ID.
- The prompt/completion format can evolve, but the source lineage cannot be
  optional.

### delta_manifest

A `delta_manifest` summarizes what changed between two crawl runs.

Required fields are defined in
[`delta_manifest.schema.json`](../../schemas/provenance/delta_manifest.schema.json).

Key requirements:

- List added, changed, unchanged, deleted, redirected, failed, and excluded
  URLs.
- List entailments that require revalidation.
- List entailments that were superseded or retracted.
- Include rebuild recommendations for affected dataset versions.

### gap_manifest

A `gap_manifest` describes coverage gaps discovered after source-grounded
dataset analysis. It is the contract between custom coverage analysis and
NeMo Data Designer.

Required fields are defined in
[`gap_manifest.schema.json`](../../schemas/provenance/gap_manifest.schema.json).

Key requirements:

- Gaps must be tied to explicit dimensions such as product, microservice, task
  type, API surface, configuration pattern, failure mode, or version.
- Each gap must include seed entailments or source chunks when available.
- Each gap must state whether synthetic generation is recommended, blocked, or
  needs human review.

### dataset_version_manifest

A `dataset_version_manifest` describes an immutable dataset build and its
inputs.

Required fields are defined in
[`dataset_version_manifest.schema.json`](../../schemas/provenance/dataset_version_manifest.schema.json).

Key requirements:

- Identify source-grounded, synthetic, and blended dataset versions.
- Record exact input artifact IDs and hashes.
- Record NeMo Curator job/config IDs when curation is used.
- Record NeMo Entity Store/Data Store registration pointers.
- Record train/validation/test split method and seed.

## Delta Recrawl Algorithm

1. Canonicalize each URL using the same canonicalization version used for the
   prior crawl.
2. Fetch the URL and create a new `source_revision` when the fetch result is
   new, changed, deleted, redirected, failed, or explicitly excluded.
3. Compare raw and normalized hashes. If the normalized hash is unchanged, the
   URL does not need re-chunking or entailment extraction.
4. For changed content, chunk the new revision with the current chunker.
5. Match new chunks to prior chunks in this order: exact text hash, structural
   anchor, then semantic similarity when enabled.
6. Revalidate only entailments connected to changed, deleted, or unmatched
   chunks.
7. Mark old entailments as active, superseded, retracted, rejected, or
   needs_review. Do not overwrite the old records.
8. Extract new entailments only from added or changed chunks.
9. Emit a `delta_manifest`.
10. Rebuild affected dataset versions from active entailments.
11. Run coverage analysis to create or update the `gap_manifest`.
12. Run NeMo Data Designer only for approved gap-fill work.

## NeMo Integration Points

### NeMo Entity Store and Data Store

Register each dataset version as a dataset entity. Use high-level entity fields
for discoverability and store the full provenance ledger as dataset files.

Recommended dataset `custom_fields`:

```json
{
  "pipeline": "docs-to-data-to-lora",
  "dataset_role": "source_entailed",
  "lineage_manifest": "manifests/dataset_version_manifest.json",
  "crawl_run_id": "crawlrun_...",
  "source_revision_count": 123,
  "entailment_count": 456,
  "synthetic_count": 0,
  "curator_config_hash": "sha256:...",
  "schema_version": "provenance.v1"
}
```

### NeMo Curator

Use NeMo Curator for exact, fuzzy, and semantic deduplication where practical.
Record the curator config hash and output artifact IDs in the
`dataset_version_manifest`.

The custom pipeline may still compute domain-specific coverage and policy
filters, but generic deduplication should be represented as Curator-backed in
the showcase path.

### NeMo Data Designer

Use the `gap_manifest` and selected source-grounded rows as seed inputs.
Synthetic rows must be written with `origin: synthetic_gapfill` and must include
the Data Designer job ID, gap ID, seed sample IDs, and generation config hash.

### NeMo Evaluator and NIM Proxy

Evaluation targets should prefer native NeMo Evaluator model or RAG targets
pointing at NIM Proxy endpoints. Custom proxy code should only remain when a
specific transformation cannot be represented with Evaluator target
configuration or Evaluator interceptors.

## Recommended File Layout

```text
datasets/
  <dataset_name>/
    manifests/
      crawl_run.json
      delta_manifest.json
      gap_manifest.json
      dataset_version_manifest.json
    provenance/
      source_revisions.jsonl
      source_chunks.jsonl
      entailments.jsonl
      dataset_samples.jsonl
    splits/
      training.jsonl
      validation.jsonl
      test.jsonl
```

The exact storage root can be local, NeMo Data Store, Hugging Face, or S3. The
manifest structure should remain stable.

## Open Design Decisions

- Whether row-level ledgers should be JSONL only, Parquet only, or both.
- Whether source text should be duplicated in `source_chunk` or stored by
  pointer for large binary-derived corpora.
- Whether entailment validation should require a second model or allow a
  deterministic source-span verifier for some claim types.
- Whether deleted URLs should remain in future evaluation sets as regression
  checks or be removed from active train/eval splits.
