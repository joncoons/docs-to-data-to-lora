#!/usr/bin/env python3
"""Submit the corrected five-epoch Curator LoRA matrix.

This is deliberately a thin wrapper around the audited e5 submission logic.  It
changes only output entity names, ensuring the repaired run cannot collide with
entities partially created by the failed handler-placement attempt.
"""
from __future__ import annotations

import submit_customizer_training_e5 as submission


for _spec in submission.JOBS.values():
    _spec["output_model"] = _spec["output_model"].replace(
        "-e5-20260629", "-e5-fix1-20260629"
    )


if __name__ == "__main__":
    raise SystemExit(submission.main())
