"""External judge: Claude Sonnet 4.6 via NVIDIA Inference API."""
from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI

log = logging.getLogger(__name__)


class ClaudeJudge:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_retries: int = 3, retry_base_s: float = 5.0) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s

    def grade(self, system: str, user: str, max_tokens: int = 512) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                log.warning("Claude judge attempt %d failed: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_s * (attempt + 1))
        return None
