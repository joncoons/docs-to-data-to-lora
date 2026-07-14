"""Super-120b NIM client with round-robin pool, rate limiting, and retries."""
from __future__ import annotations

import itertools
import logging
import re
import time
from threading import Semaphore
from typing import Optional

from openai import OpenAI

log = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_PRELUDE_RE = re.compile(r"^.*?</think>", re.DOTALL)


def _strip_think_blocks(content: str) -> str:
    """Remove reasoning-model think blocks from LLM output.

    Handles two patterns:
      1. Explicit pairs: <think>...</think>
      2. Prelude that starts in think mode (no opening tag): "blah blah </think>\\n\\nactual answer"
    """
    if not content:
        return content
    # First remove any well-formed pairs
    content = _THINK_BLOCK_RE.sub("", content)
    # Then strip any prelude up to and including </think>
    content = _THINK_PRELUDE_RE.sub("", content)
    return content.strip()


class LLMClient:
    """Round-robin client pool for one or more OpenAI-compatible endpoints."""

    def __init__(
        self,
        endpoints: list[str],
        model: str,
        api_key: str = "local",
        max_workers: int = 5,
        min_interval_s: float = 0.5,
        retry_attempts: int = 3,
        retry_base_delay_s: float = 5.0,
        no_think: bool = False,
        temperature: float = 0.2,
    ) -> None:
        self.endpoints = endpoints
        self.model = model
        self.no_think = no_think
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s
        self._semaphore = Semaphore(max_workers)
        self._last_call = [0.0]
        self._min_interval = min_interval_s
        self._counter = itertools.count()
        self._clients = [OpenAI(base_url=ep, api_key=api_key) for ep in endpoints]

    def _next_client(self) -> tuple[OpenAI, str]:
        index = next(self._counter) % len(self._clients)
        return self._clients[index], self.endpoints[index]

    @staticmethod
    def _no_think_extra_body(endpoint: str) -> dict:
        # Generic OpenAI-compatible hint; providers that do not support it should ignore it.
        return {"reasoning_effort": "none"}

    def call(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        with self._semaphore:
            for attempt in range(self.retry_attempts):
                elapsed = time.time() - self._last_call[0]
                if elapsed < self._min_interval:
                    time.sleep(self._min_interval - elapsed)
                client, endpoint = self._next_client()
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                    }
                    if self.no_think:
                        kwargs["extra_body"] = self._no_think_extra_body(endpoint)
                    resp = client.chat.completions.create(**kwargs)
                    self._last_call[0] = time.time()
                    return _strip_think_blocks(resp.choices[0].message.content)
                except Exception as e:
                    log.warning("LLM call attempt %d failed: %s", attempt + 1, e)
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_base_delay_s * (attempt + 1))
            return None
