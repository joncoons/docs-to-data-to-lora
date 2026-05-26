"""Thin HTTP wrapper around NeMo Customizer's REST API.

Endpoint and exact request shape may differ slightly between Customizer versions
— verify against the running 25.12 cluster service at implementation time. If
the routes here don't match, adjust route names but keep the public Client
interface (submit_job, get_status, get_output_path, wait_until_done) unchanged.

Cluster service: nemo-customizer.nemo-peft:8000 (NodePort 30910).
"""
from __future__ import annotations

import enum
import logging
import time
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Status strings the Customizer may return → JobStatus. Verify against actual service.
_STATUS_MAP = {
    "queued": JobStatus.QUEUED,
    "pending": JobStatus.QUEUED,
    "running": JobStatus.RUNNING,
    "in_progress": JobStatus.RUNNING,
    "completed": JobStatus.COMPLETED,
    "succeeded": JobStatus.COMPLETED,
    "success": JobStatus.COMPLETED,
    "failed": JobStatus.FAILED,
    "error": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
}


class CustomizerClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers,
                                   timeout=timeout)

    def submit_job(self, config: dict) -> str:
        """POST a Customizer job. Returns the job id."""
        resp = self._http.post("/v1/customizations/jobs", json=config)
        resp.raise_for_status()
        body = resp.json()
        return body["id"]

    def get_status(self, job_id: str) -> JobStatus:
        resp = self._http.get(f"/v1/customizations/jobs/{job_id}")
        resp.raise_for_status()
        raw_status = resp.json()["status"].lower()
        if raw_status not in _STATUS_MAP:
            raise ValueError(f"Unknown status from Customizer: {raw_status!r}")
        return _STATUS_MAP[raw_status]

    def get_output_path(self, job_id: str) -> Optional[str]:
        resp = self._http.get(f"/v1/customizations/jobs/{job_id}")
        resp.raise_for_status()
        body = resp.json()
        return body.get("output", {}).get("path") or body.get("output_path")

    def wait_until_done(self, job_id: str, poll_interval_s: float = 30.0,
                        timeout_s: float = 4 * 3600) -> JobStatus:
        """Poll until COMPLETED / FAILED / CANCELLED. Returns terminal status."""
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        start = time.time()
        while time.time() - start < timeout_s:
            st = self.get_status(job_id)
            log.info("Customizer job %s status: %s", job_id, st.value)
            if st in terminal:
                return st
            time.sleep(poll_interval_s)
        raise TimeoutError(f"Customizer job {job_id} did not terminate within {timeout_s}s")

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
