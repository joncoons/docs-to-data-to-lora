"""Tests for EvaluatorClient — REST wrapper over /v1/evaluation/*."""
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval.evaluator_client import EvaluatorClient, EvalJobStatus


def _mock_resp(json_body, status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.raise_for_status = MagicMock()
    return r


def test_create_target_returns_id():
    client = EvaluatorClient(base_url="http://test:7331")
    payload = {"name": "test-target", "namespace": "default", "type": "model"}
    with patch.object(client._http, "post",
                      return_value=_mock_resp({"id": "tgt-abc"})) as mock_post:
        tid = client.create_target(payload)
    assert tid == "tgt-abc"
    args, kwargs = mock_post.call_args
    assert args[0] == "/v1/evaluation/targets"
    assert kwargs["json"] == payload


def test_create_config_returns_id():
    client = EvaluatorClient(base_url="http://test:7331")
    payload = {"name": "test-cfg", "namespace": "default", "type": "custom"}
    with patch.object(client._http, "post",
                      return_value=_mock_resp({"id": "cfg-xyz"})) as mock_post:
        cid = client.create_config(payload)
    assert cid == "cfg-xyz"
    assert mock_post.call_args.args[0] == "/v1/evaluation/configs"


def test_submit_job_returns_job_id():
    client = EvaluatorClient(base_url="http://test:7331")
    payload = {"config": "default/cfg1", "target": "default/tgt1"}
    with patch.object(client._http, "post",
                      return_value=_mock_resp({"id": "ej-001"})) as mock_post:
        job_id = client.submit_job(payload)
    assert job_id == "ej-001"
    assert mock_post.call_args.args[0] == "/v1/evaluation/jobs"


def test_get_status_normalizes_to_enum():
    client = EvaluatorClient(base_url="http://test:7331")
    with patch.object(client._http, "get",
                      return_value=_mock_resp({"status": "completed"})):
        s = client.get_status("ej-001")
    assert s == EvalJobStatus.COMPLETED


@pytest.mark.parametrize("raw, expected", [
    ("pending",   EvalJobStatus.PENDING),
    ("running",   EvalJobStatus.RUNNING),
    ("completed", EvalJobStatus.COMPLETED),
    ("failed",    EvalJobStatus.FAILED),
    ("cancelled", EvalJobStatus.CANCELLED),
    ("queued",    EvalJobStatus.PENDING),     # alias
    ("ready",     EvalJobStatus.COMPLETED),   # alias
])
def test_status_enum_mapping(raw, expected):
    client = EvaluatorClient(base_url="http://test:7331")
    with patch.object(client._http, "get",
                      return_value=_mock_resp({"status": raw})):
        assert client.get_status("ej-x") == expected


def test_get_status_raises_on_unknown_status():
    client = EvaluatorClient(base_url="http://test:7331")
    with patch.object(client._http, "get",
                      return_value=_mock_resp({"status": "weird-state"})):
        with pytest.raises(ValueError, match="Unknown status"):
            client.get_status("ej-x")


def test_wait_until_done_polls_until_terminal():
    client = EvaluatorClient(base_url="http://test:7331")
    statuses = iter(["pending", "running", "running", "completed"])
    with patch.object(client._http, "get",
                      side_effect=lambda *a, **k: _mock_resp({"status": next(statuses)})):
        with patch("scripts.eval.evaluator_client.time.sleep"):  # no real sleep
            final = client.wait_until_done("ej-x", poll_interval=0.01,
                                            max_wait_s=10)
    assert final == EvalJobStatus.COMPLETED


def test_get_results_calls_results_endpoint():
    client = EvaluatorClient(base_url="http://test:7331")
    body = {"per_question": [{"score": 4.5}], "aggregate": {"mean": 4.5}}
    with patch.object(client._http, "get",
                      return_value=_mock_resp(body)) as mock_get:
        r = client.get_results("ej-001")
    assert r == body
    assert mock_get.call_args.args[0] == "/v1/evaluation/jobs/ej-001/results"


def test_wait_until_done_raises_timeout_when_never_terminal():
    """If the deadline expires before any terminal status, raise TimeoutError."""
    client = EvaluatorClient(base_url="http://test:7331")
    with patch.object(client._http, "get",
                      return_value=_mock_resp({"status": "running"})):
        with patch("scripts.eval.evaluator_client.time.sleep"):
            with pytest.raises(TimeoutError, match="did not terminate"):
                client.wait_until_done("ej-x", poll_interval=0.0, max_wait_s=0.0)
