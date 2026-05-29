"""Tests for Stage 3 Customizer client (mock HTTP)."""

from unittest.mock import MagicMock, patch

from scripts.stage3.customizer_client import CustomizerClient, JobStatus


def test_submit_job_returns_job_id():
    client = CustomizerClient(base_url="http://customizer.nemo-peft:8000", api_key="ignored")
    mock_resp = MagicMock(status_code=201)
    mock_resp.json.return_value = {"id": "cust-abc123", "status": "queued"}
    with patch.object(client._http, "post", return_value=mock_resp) as p:
        job_id = client.submit_job(config={"foo": "bar"})
    assert job_id == "cust-abc123"
    p.assert_called_once()


def test_get_status_returns_enum():
    client = CustomizerClient(base_url="http://x", api_key="x")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"id": "cust-abc123", "status": "running"}
    with patch.object(client._http, "get", return_value=mock_resp):
        st = client.get_status("cust-abc123")
    assert st == JobStatus.RUNNING


def test_get_status_completed():
    client = CustomizerClient(base_url="http://x", api_key="x")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"id": "cust-abc123", "status": "completed"}
    with patch.object(client._http, "get", return_value=mock_resp):
        st = client.get_status("cust-abc123")
    assert st == JobStatus.COMPLETED


def test_get_status_failed_returns_enum():
    client = CustomizerClient(base_url="http://x", api_key="x")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"id": "cust-abc123", "status": "failed", "error": "NCCL timeout"}
    with patch.object(client._http, "get", return_value=mock_resp):
        st = client.get_status("cust-abc123")
    assert st == JobStatus.FAILED


def test_get_status_unknown_raises():
    import pytest

    client = CustomizerClient(base_url="http://x", api_key="x")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"id": "cust-abc123", "status": "weird_state"}
    with patch.object(client._http, "get", return_value=mock_resp):
        with pytest.raises(ValueError, match="Unknown status"):
            client.get_status("cust-abc123")


def test_get_platform_status_uses_sdk_status_detail(monkeypatch):
    import sys
    import types

    class FakeJobs:
        def get_status(self, name, workspace):
            assert name == "lora-job"
            assert workspace == "default"
            return {"name": name, "status": "active"}

    class FakePlatform:
        def __init__(self, base_url, workspace):
            assert base_url == "http://platform"
            assert workspace == "default"
            self.customization = types.SimpleNamespace(jobs=FakeJobs())

    monkeypatch.setitem(
        sys.modules,
        "nemo_platform",
        types.SimpleNamespace(NeMoPlatform=FakePlatform),
    )

    client = CustomizerClient(base_url="http://platform")

    assert client.get_platform_status("lora-job", "default") == JobStatus.RUNNING


def test_wait_platform_until_done_polls_by_name_workspace(monkeypatch):
    client = CustomizerClient(base_url="http://platform")
    statuses = iter([JobStatus.RUNNING, JobStatus.COMPLETED])
    calls = []

    def fake_status(name, workspace):
        calls.append((name, workspace))
        return next(statuses)

    monkeypatch.setattr(client, "get_platform_status", fake_status)
    monkeypatch.setattr("scripts.stage3.customizer_client.time.sleep", lambda _seconds: None)

    status = client.wait_platform_until_done(
        name="lora-job",
        workspace="default",
        poll_interval_s=0.01,
    )

    assert status == JobStatus.COMPLETED
    assert calls == [("lora-job", "default"), ("lora-job", "default")]
