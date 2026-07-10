# Publish Review Notes - 2026-07-10

This repository is being kept as a local experiment ledger during the LE/Curator/LoRA work. Before publishing to a shared remote, revisit the archive policy and decide what belongs in the remote branch versus local-only preservation.

## Already moved local-only

- `archive/generated/` was moved to `.local-archive/20260710-generated-archive-snapshot/generated/`. It contained generated runtime/cache/build artifacts and is intentionally excluded by `.gitignore`.

## Review before remote publish

- `archive/cluster-backups/`: Kubernetes snapshots may contain environment-specific configuration and should be reviewed for sensitivity and relevance.
- `archive/stage3-orchestrators/` and `archive/stage3-runbooks/`: useful implementation provenance, but verify they still represent the intended public workflow.
- `archive/iteration-notes/` and `archive/local-preservation/`: local history may be valuable, but should be trimmed or summarized if the remote is meant to be turnkey.
- `curator_dataset/experiments/`: retain committed experiment summaries and scripts, but review generated run outputs before publishing.

Use `archive/local-preservation/REMOTE_PUBLISH_EXCLUDE.gitignore` as the checklist for building a publishable branch.
