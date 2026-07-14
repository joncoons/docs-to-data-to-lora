"""
embed_client.py — async client for the NVIDIA NIM embedding endpoint.

Batches texts, calls POST /v1/embeddings with input_type="passage",
and returns vectors aligned with the input list.  None is returned in
the position of any text that fails to embed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from . import config

logger = logging.getLogger(__name__)


async def embed_texts(
    texts: list[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> list[list[float] | None]:
    """
    Embed *texts* via the NIM embedding endpoint.

    Returns a list of float vectors (or None on failure) in the same order as
    *texts*.  Requests are batched by EMBED_BATCH_SIZE and run with up to
    EMBED_CONCURRENCY concurrent POST calls.
    """
    if not texts:
        return []

    batch_size = config.EMBED_BATCH_SIZE
    results: list[list[float] | None] = [None] * len(texts)

    async def _embed_batch(start: int, batch: list[str]) -> None:
        payload: dict[str, Any] = {
            "input": batch,
            "model": config.EMBED_MODEL,
            "input_type": "passage",
        }
        async with semaphore:
            try:
                async with session.post(
                    config.EMBED_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    embeddings = data.get("data", [])
                    for item in embeddings:
                        idx = item.get("index", 0)
                        results[start + idx] = item.get("embedding")
            except Exception as exc:
                logger.warning(
                    "embed_client: batch at offset %d failed: %r", start, exc
                )

    tasks = [
        _embed_batch(i, texts[i : i + batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    await asyncio.gather(*tasks)
    return results
