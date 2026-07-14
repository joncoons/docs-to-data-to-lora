#!/usr/bin/env python3
"""Submit fix2 NeMo jobs with Customizer-compatible entity name lengths."""
from __future__ import annotations

import submit_customizer_training_e5_fix2 as fix2


for _spec in fix2.submission.JOBS.values():
    _spec["output_model"] = _spec["output_model"].replace(
        "lora-nemo-usvcs-", "lora-nemo-ms-"
    )


if __name__ == "__main__":
    raise SystemExit(fix2.submission.main())
