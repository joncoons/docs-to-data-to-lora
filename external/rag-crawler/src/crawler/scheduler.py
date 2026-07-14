"""
scheduler.py — APScheduler-backed crawl schedule manager.

Schedules are persisted as JSON to CONFIGS_DIR/schedules.json (NFS), so
they survive pod restarts.  Each schedule fires an internal crawl task
identical to what POST /crawl would produce.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, Field

from . import config, task_handler

logger = logging.getLogger(__name__)

# Module-level singleton — initialised in start()
_scheduler: AsyncIOScheduler | None = None


# ── Data model ────────────────────────────────────────────────────────────────

class ScheduleConfig(BaseModel):
    id: str
    label: str
    start_url: str
    collection_name: str | None = None
    max_pages: int | None = None
    max_depth: int | None = None
    batch_ingest_size: int = 50
    extract_linked_files: bool = False
    cron_expression: str = Field(..., description="Standard 5-field cron (minute hour dom month dow)")
    enabled: bool = True
    last_run_at: str | None = None
    next_run_at: str | None = None
    created_at: str


# ── Persistence helpers ───────────────────────────────────────────────────────

def _schedules_path() -> Path:
    # Fall back to REGISTRY_DIR if CONFIGS_DIR doesn't exist yet
    d = Path(config.CONFIGS_DIR)
    if not d.exists():
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            d = Path(config.REGISTRY_DIR)
    return d / "schedules.json"


def _load() -> list[ScheduleConfig]:
    p = _schedules_path()
    if not p.exists():
        return []
    try:
        return [ScheduleConfig(**item) for item in json.loads(p.read_text())]
    except Exception as exc:
        logger.warning("scheduler: could not load schedules.json: %s", exc)
        return []


def _save(schedules: list[ScheduleConfig]) -> None:
    p = _schedules_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([s.model_dump() for s in schedules], indent=2))
    except Exception as exc:
        logger.error("scheduler: could not save schedules.json: %s", exc)


def _next_run(cron_expr: str) -> str | None:
    try:
        trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")
        nf = trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        return nf.isoformat() if nf else None
    except Exception:
        return None


# ── Internal crawl runner (avoids circular import with server.py) ─────────────

async def _fire_crawl(task_id: str, sched: ScheduleConfig) -> None:
    from .crawl import SimpleWebCrawler
    try:
        crawler = SimpleWebCrawler(
            start_url=sched.start_url,
            max_pages=sched.max_pages,
            max_depth=sched.max_depth,
            batch_size=sched.batch_ingest_size,
            collection_name=sched.collection_name or "",
            extract_linked_files=sched.extract_linked_files,
            task_id=task_id,
        )
        result = await crawler.crawl(collection_name=sched.collection_name or "")
        task_handler.set_task(task_id, "FINISHED", result or {})
    except Exception as exc:
        logger.exception("scheduler: crawl task %s failed", task_id)
        task_handler.set_task(task_id, "FAILURE", {"error": str(exc)})


async def _run_scheduled_crawl(schedule_id: str) -> None:
    """APScheduler job entry point."""
    schedules = _load()
    sched = next((s for s in schedules if s.id == schedule_id), None)
    if not sched or not sched.enabled:
        logger.info("scheduler: schedule %s not found or disabled — skipping", schedule_id)
        return

    logger.info("scheduler: firing %s → %s", schedule_id, sched.start_url)

    task_id = task_handler.new_task_id()
    task_handler.set_task(task_id, "PENDING")
    asyncio.create_task(_fire_crawl(task_id, sched))

    # Update timestamps
    now = datetime.now(timezone.utc).isoformat()
    for s in schedules:
        if s.id == schedule_id:
            s.last_run_at = now
            s.next_run_at = _next_run(sched.cron_expression)
    _save(schedules)


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

def _get() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
    return _scheduler


def _register_job(sched: ScheduleConfig) -> None:
    try:
        trigger = CronTrigger.from_crontab(sched.cron_expression, timezone="UTC")
        _get().add_job(
            _run_scheduled_crawl,
            trigger=trigger,
            id=sched.id,
            args=[sched.id],
            replace_existing=True,
        )
    except Exception as exc:
        logger.warning("scheduler: could not register job %s: %s", sched.id, exc)


def _unregister_job(schedule_id: str) -> None:
    try:
        _get().remove_job(schedule_id)
    except Exception:
        pass


def start() -> None:
    """Start the scheduler and reload all persisted enabled schedules."""
    sch = _get()
    loaded = _load()
    for s in loaded:
        if s.enabled:
            _register_job(s)
    sch.start()
    logger.info("scheduler: started — %d schedule(s) loaded", sum(1 for s in loaded if s.enabled))


def stop() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None


# ── CRUD ─────────────────────────────────────────────────────────────────────

def list_schedules() -> list[ScheduleConfig]:
    return _load()


def get_schedule(schedule_id: str) -> ScheduleConfig | None:
    return next((s for s in _load() if s.id == schedule_id), None)


def create_schedule(data: dict[str, Any]) -> ScheduleConfig:
    schedules = _load()
    sched = ScheduleConfig(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        next_run_at=_next_run(data["cron_expression"]),
        last_run_at=None,
        **{k: v for k, v in data.items() if k not in ("id", "created_at", "next_run_at", "last_run_at")},
    )
    schedules.append(sched)
    _save(schedules)
    if sched.enabled:
        _register_job(sched)
    return sched


def update_schedule(schedule_id: str, updates: dict[str, Any]) -> ScheduleConfig | None:
    schedules = _load()
    for i, s in enumerate(schedules):
        if s.id != schedule_id:
            continue
        updated = s.model_copy(update=updates)
        if "cron_expression" in updates:
            updated.next_run_at = _next_run(updated.cron_expression)
        schedules[i] = updated
        _save(schedules)
        # Re-register (handles enable → disable and cron changes)
        _unregister_job(schedule_id)
        if updated.enabled:
            _register_job(updated)
        return updated
    return None


def delete_schedule(schedule_id: str) -> bool:
    schedules = _load()
    filtered = [s for s in schedules if s.id != schedule_id]
    if len(filtered) == len(schedules):
        return False
    _save(filtered)
    _unregister_job(schedule_id)
    return True
