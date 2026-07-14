#!/usr/bin/env python3
"""Submit the split-placement corrected five-epoch Curator LoRA matrix."""
from __future__ import annotations

import submit_customizer_training_e5 as submission


for _spec in submission.JOBS.values():
    _spec["output_model"] = _spec["output_model"].replace(
        "-e5-20260629", "-e5-fix2-20260629"
    )


if __name__ == "__main__":
    raise SystemExit(submission.main())
