"""
task_handler.py — Redis-backed task state for long-running crawls.

Task lifecycle: PENDING → FINISHED | FAILURE | CANCELLED

State is persisted to Redis so it survives pod restarts.  Falls back to an
in-memory dict when Redis is unavailable or ENABLE_REDIS_BACKEND=false.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# In-memory fallback (used when Redis is disabled or unreachable)
_MEM_STORE: dict[str, dict] = {}

_redis_client: Any = None


def _get_redis() -> Any:
    global _redis_client
    if not config.ENABLE_REDIS_BACKEND:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # noqa: PLC0415
        _redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        _redis_client.ping()
        logger.info(
            "task_handler: Redis connected at %s:%d",
            config.REDIS_HOST, config.REDIS_PORT,
        )
    except Exception as exc:
        logger.warning("task_handler: Redis unavailable (%s) — using memory store", exc)
        _redis_client = None
    return _redis_client


def new_task_id() -> str:
    return str(uuid.uuid4())


def set_task(task_id: str, state: str, result: dict | None = None) -> None:
    payload = {"state": state, "result": result or {}}
    r = _get_redis()
    if r:
        try:
            r.setex(
                f"crawler:task:{task_id}",
                config.REDIS_TTL_SECONDS,
                json.dumps(payload),
            )
            return
        except Exception as exc:
            logger.warning("task_handler: Redis set failed: %r", exc)
    _MEM_STORE[task_id] = payload


def get_task(task_id: str) -> dict:
    r = _get_redis()
    if r:
        try:
            raw = r.get(f"crawler:task:{task_id}")
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("task_handler: Redis get failed: %r", exc)
    return _MEM_STORE.get(task_id, {"state": "UNKNOWN", "result": {}})


def set_progress(task_id: str, progress: dict) -> None:
    """Update in-flight progress (short TTL — 10 min is sufficient)."""
    r = _get_redis()
    payload = json.dumps(progress)
    if r:
        try:
            r.setex(f"crawler:progress:{task_id}", 600, payload)
            return
        except Exception:
            pass
    _MEM_STORE[f"progress:{task_id}"] = progress


def get_progress(task_id: str) -> dict | None:
    r = _get_redis()
    if r:
        try:
            raw = r.get(f"crawler:progress:{task_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _MEM_STORE.get(f"progress:{task_id}")


# Cancel signals — stored in Redis so any pod can trigger them
_CANCEL_KEY = "crawler:cancel:{}"


def request_cancel(task_id: str) -> None:
    r = _get_redis()
    if r:
        try:
            r.setex(_CANCEL_KEY.format(task_id), 3600, "1")
            return
        except Exception:
            pass
    _MEM_STORE[f"cancel:{task_id}"] = True


def is_cancel_requested(task_id: str) -> bool:
    r = _get_redis()
    if r:
        try:
            return bool(r.exists(_CANCEL_KEY.format(task_id)))
        except Exception:
            pass
    return bool(_MEM_STORE.get(f"cancel:{task_id}"))


def clear_cancel(task_id: str) -> None:
    r = _get_redis()
    if r:
        try:
            r.delete(_CANCEL_KEY.format(task_id))
        except Exception:
            pass
    _MEM_STORE.pop(f"cancel:{task_id}", None)
