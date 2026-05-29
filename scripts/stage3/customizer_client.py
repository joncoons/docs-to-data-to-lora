"""Thin HTTP wrapper around NeMo Customizer's REST API.

Endpoint and exact request shape may differ between legacy standalone
Customizer and the NeMo Platform API. This client preserves the existing
submit_job, get_status, get_output_path, and wait_until_done interface and adds
Platform SDK status helpers for spec/FileSet based jobs.

Default endpoint selection lives in scripts.nemo_platform.
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
    "active": JobStatus.RUNNING,
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


def _status_from_raw(raw_status: str) -> JobStatus:
    normalized = raw_status.lower()
    if normalized not in _STATUS_MAP:
        raise ValueError(f"Unknown status from Customizer: {raw_status!r}")
    return _STATUS_MAP[normalized]


def _extract_job_id(job: Any) -> str:
    for attr in ("id", "job_id", "name"):
        value = getattr(job, attr, None)
        if value:
            return str(value)
    if isinstance(job, dict):
        for key in ("id", "job_id", "name"):
            if job.get(key):
                return str(job[key])
    return str(job)


def _to_plain_dict(value: Any) -> dict:
    """Best-effort conversion for SDK models so callers can log JSON artifacts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": str(value)}


class CustomizerClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def submit_job(self, config: dict) -> str:
        """POST a Customizer job. Returns the job id."""
        resp = self._http.post("/v1/customization/jobs", json=config)
        resp.raise_for_status()
        body = resp.json()
        return body["id"]

    def submit_platform_job(self, name: str, workspace: str, spec: dict) -> str:
        """Create a NeMo Platform Customizer job through the Platform SDK."""
        try:
            from nemo_platform import NeMoPlatform  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "NeMo Platform SDK is not installed. Install nemo-platform or "
                "run with --payload-format legacy against standalone Customizer."
            ) from exc

        client = NeMoPlatform(base_url=self.base_url, workspace=workspace)
        job = client.customization.jobs.create(
            name=name,
            workspace=workspace,
            spec=spec,
        )
        return _extract_job_id(job)

    def get_platform_status_detail(self, name: str, workspace: str) -> dict:
        """Return detailed NeMo Platform Customizer status as a plain dict."""
        try:
            from nemo_platform import NeMoPlatform  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "NeMo Platform SDK is not installed. Install nemo-platform to "
                "poll Platform Customizer jobs."
            ) from exc

        client = NeMoPlatform(base_url=self.base_url, workspace=workspace)
        status = client.customization.jobs.get_status(
            name=name,
            workspace=workspace,
        )
        return _to_plain_dict(status)

    def get_platform_status(self, name: str, workspace: str) -> JobStatus:
        """Return normalized status for a NeMo Platform Customizer job."""
        detail = self.get_platform_status_detail(name=name, workspace=workspace)
        return _status_from_raw(str(detail["status"]))

    def get_status(self, job_id: str) -> JobStatus:
        resp = self._http.get(f"/v1/customization/jobs/{job_id}")
        resp.raise_for_status()
        raw_status = resp.json()["status"]
        return _status_from_raw(str(raw_status))

    def get_output_path(self, job_id: str) -> Optional[str]:
        resp = self._http.get(f"/v1/customization/jobs/{job_id}")
        resp.raise_for_status()
        body = resp.json()
        return body.get("output", {}).get("path") or body.get("output_path")

    def wait_until_done(
        self, job_id: str, poll_interval_s: float = 30.0, timeout_s: float = 4 * 3600
    ) -> JobStatus:
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

    def wait_platform_until_done(
        self,
        name: str,
        workspace: str,
        poll_interval_s: float = 30.0,
        timeout_s: float = 4 * 3600,
    ) -> JobStatus:
        """Poll a Platform Customizer job by name/workspace until terminal."""
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        start = time.time()
        while time.time() - start < timeout_s:
            st = self.get_platform_status(name=name, workspace=workspace)
            log.info("Platform Customizer job %s/%s status: %s", workspace, name, st.value)
            if st in terminal:
                return st
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Platform Customizer job {workspace}/{name} did not terminate within {timeout_s}s"
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
