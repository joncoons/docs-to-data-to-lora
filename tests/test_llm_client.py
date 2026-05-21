"""Tests for super-120b LLM client (round-robin + retries)."""
from unittest.mock import MagicMock, patch
import pytest

from scripts.pipeline.llm_client import LLMClient


def test_round_robin_distributes_calls():
    """Two endpoints → calls alternate between them."""
    client = LLMClient(endpoints=["http://a:8000/v1", "http://b:8000/v1"],
                       model="m", retry_attempts=1)
    mock_a = MagicMock()
    mock_b = MagicMock()
    mock_a.chat.completions.create.return_value.choices[0].message.content = "from-a"
    mock_b.chat.completions.create.return_value.choices[0].message.content = "from-b"
    client._clients = [mock_a, mock_b]

    r1 = client.call("sys", "user1")
    r2 = client.call("sys", "user2")
    r3 = client.call("sys", "user3")

    assert r1 == "from-a"
    assert r2 == "from-b"
    assert r3 == "from-a"


def test_retries_on_exception():
    client = LLMClient(endpoints=["http://a:8000/v1"], model="m",
                       retry_attempts=3, retry_base_delay_s=0.0)
    mock_c = MagicMock()
    mock_c.chat.completions.create.side_effect = [
        RuntimeError("boom"),
        RuntimeError("boom"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))]),
    ]
    client._clients = [mock_c]
    assert client.call("sys", "user") == "ok"
    assert mock_c.chat.completions.create.call_count == 3


def test_returns_none_after_max_retries():
    client = LLMClient(endpoints=["http://a:8000/v1"], model="m",
                       retry_attempts=2, retry_base_delay_s=0.0)
    mock_c = MagicMock()
    mock_c.chat.completions.create.side_effect = RuntimeError("boom")
    client._clients = [mock_c]
    assert client.call("sys", "user") is None


def test_no_think_mode_sets_reasoning_effort():
    client = LLMClient(endpoints=["http://a:8000/v1"], model="m",
                       retry_attempts=1, no_think=True)
    mock_c = MagicMock()
    mock_c.chat.completions.create.return_value.choices[0].message.content = "ok"
    client._clients = [mock_c]
    client.call("sys", "user")
    kwargs = mock_c.chat.completions.create.call_args.kwargs
    assert kwargs.get("extra_body", {}).get("reasoning_effort") == "minimal"
