"""Thin REST wrapper over NeMo Evaluator's /v1/evaluation/* endpoints."""
from __future__ import annotations

import logging
import time
from enum import Enum

import httpx

log = logging.getLogger(__name__)


class EvalJobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_STATUS_MAP = {s.value: s for s in EvalJobStatus}
_STATUS_MAP.update({"queued": EvalJobStatus.PENDING, "ready": EvalJobStatus.COMPLETED})
_TERMINAL = {EvalJobStatus.COMPLETED, EvalJobStatus.FAILED, EvalJobStatus.CANCELLED}


class EvaluatorClient:
    def __init__(self, base_url: str = "http://nemo-evaluator:7331",
                 api_key: str | None = None, timeout: float = 30.0) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.base_url = base_url
        self._http = httpx.Client(base_url=base_url, headers=headers,
                                   timeout=timeout)

    def __enter__(self): return self
    def __exit__(self, *exc): self._http.close()

    def create_target(self, payload: dict) -> str:
        resp = self._http.post("/v1/evaluation/targets", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def create_config(self, payload: dict) -> str:
        resp = self._http.post("/v1/evaluation/configs", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def update_target(self, namespace: str, name: str, payload: dict) -> str:
        resp = self._http.patch(
            f"/v1/evaluation/targets/{namespace}/{name}", json=payload
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def update_config(self, namespace: str, name: str, payload: dict) -> str:
        resp = self._http.patch(
            f"/v1/evaluation/configs/{namespace}/{name}", json=payload
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def delete_target(self, namespace: str, name: str) -> None:
        resp = self._http.delete(f"/v1/evaluation/targets/{namespace}/{name}")
        resp.raise_for_status()

    def delete_config(self, namespace: str, name: str) -> None:
        resp = self._http.delete(f"/v1/evaluation/configs/{namespace}/{name}")
        resp.raise_for_status()

    def submit_job(self, payload: dict) -> str:
        resp = self._http.post("/v1/evaluation/jobs", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def submit_live(self, payload: dict) -> dict:
        resp = self._http.post("/v1/evaluation/live", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_status(self, job_id: str) -> EvalJobStatus:
        resp = self._http.get(f"/v1/evaluation/jobs/{job_id}")
        resp.raise_for_status()
        raw = resp.json()["status"].lower()
        if raw not in _STATUS_MAP:
            raise ValueError(f"Unknown status from Evaluator: {raw!r}")
        return _STATUS_MAP[raw]

    def get_results(self, job_id: str) -> dict:
        resp = self._http.get(f"/v1/evaluation/jobs/{job_id}/results")
        resp.raise_for_status()
        return resp.json()

    def wait_until_done(self, job_id: str, poll_interval: float = 30.0,
                        max_wait_s: float = 4 * 3600) -> EvalJobStatus:
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            s = self.get_status(job_id)
            log.info("evaluator job %s status: %s", job_id, s.value)
            if s in _TERMINAL:
                return s
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not terminate within {max_wait_s}s")
