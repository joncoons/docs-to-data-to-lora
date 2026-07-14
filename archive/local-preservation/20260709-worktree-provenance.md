# Local Worktree Preservation - 2026-07-09

This checkpoint is intended to preserve the current local worktree before the next experiment edge. It is local-only unless a later explicit publishing review decides otherwise.

## Preservation policy

- Track substantive project artifacts currently present in the worktree.
- Do not delete tracked files as part of this preservation pass.
- Preserve original paths even when archive copies exist.
- Keep machine-local runtime state out of git: `.venv/`, `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` remain ignored by the normal repo `.gitignore`.
- Use `archive/local-preservation/REMOTE_PUBLISH_EXCLUDE.gitignore` as the advisory checklist before preparing any remote/public branch.

## Questionable but preserved locally

The following classes are intentionally retained in the local checkpoint even though they should be reviewed before remote publication:

- Generated Curator corpora, JSONL datasets, run outputs, watcher logs, and tokenizer/cache material under `curator_dataset/`.
- Historical Kubernetes YAML snapshots under `archive/cluster-backups/`.
- One-off evaluation jobs under `deploy/evaluation-matrix/`.
- Generated evaluation reports under `docs/experiments/` and `evals/`.
- Archived stage3 orchestrator scripts and runbooks under `archive/stage3-*`.
- Tutorial and model-serving scaffolding that may still be useful as implementation provenance.

## Non-deletion handling

Before this checkpoint, several files had been staged as renames into `archive/`. To avoid deleting anything, the original paths were restored and the archive paths are kept as additional copies. The deleted `scripts/pipeline/claude_client.py` file was also restored from `HEAD` so the local checkpoint does not remove it while the newer `external_judge_client.py` path remains available.

## Secret scan note

A broad scan found expected placeholders and test-only strings such as `api_key`, `secret-key`, `secret-token`, and documented examples like `nvapi-xxxxxxxxxxxxxxxxxxxxxx`. No real API key value is intentionally being preserved. The archive still needs a manual review before any remote publication because it includes cluster configuration snapshots and generated corpora copied from local systems.

## Related local commits

- `9c84b90` captured the Curator-vs-LE comparison, SVG, and Ultra 550B smoke-test design.
- This preservation pass should be committed separately to keep the experiment result commit clean.
