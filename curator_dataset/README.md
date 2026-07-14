# Curator corpus-to-QA experiment

This directory is an isolated exploration of NVIDIA NeMo Curator as an
alternative to the repository's custom logical-entailment (LE) extraction.
Nothing in the existing pipeline is modified by this experiment.

The proposed comparison starts from the same frozen `passages.jsonl` corpora:

- **Control:** the existing LE -> premise -> QA extraction represented by the
  canonical `stage1a_le.jsonl` artifacts.
- **Treatment:** NeMo Curator's native Nemotron-CC `DiverseQAStage`, including
  its published preprocessing and postprocessing pattern.

The repository's existing Curator handoff is not the treatment in this
experiment. That handoff filters already-generated `dataset_samples.jsonl`;
it does not generate QA pairs from corpus passages. It also currently invokes
only heuristic document filters, despite the adjacent recipe describing exact
and fuzzy deduplication as planned stages.

## Isolation contract

Future implementation work must follow these rules:

1. All experiment code, configs, tests, deployment templates, and reports live
   under `curator_dataset/`.
2. Canonical datasets under `/mnt/nvme2/peft/datasets/v2/` are read-only.
3. Runtime outputs go to a new root such as
   `/mnt/nvme2/peft/experiments/curator_dataset/<run-id>/`; no command may use a
   canonical collection directory as an output.
4. No existing source file, config, manifest, dataset, test set, adapter, or
   cluster job is changed in place.
5. A preflight step must reject an output path that resolves inside
   `/mnt/nvme2/peft/datasets/v2/`.
6. Container tags, model identifiers, prompts, configs, input hashes, random
   seeds, and image digests are recorded before execution.

## Documents

- [Execution plan](EXECUTION_PLAN.md)
- [License and evidence plan](LEGAL_AND_PROVENANCE.md)

No experiment has been executed and no cluster resource has been changed as
part of this initial scope.
