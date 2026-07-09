#!/usr/bin/env python3
"""Resubmit parent-matched NeMo jobs after enforcing NIM-first admission."""
from __future__ import annotations

import submit_customizer_training_parent_matched as submission


for _spec in submission.JOBS.values():
    if _spec["phase"] == "nemo_usvcs":
        _spec["output_model"] = _spec["output_model"].replace(
            "-20260629", "-q1-20260629"
        )


if __name__ == "__main__":
    raise SystemExit(submission.main())
