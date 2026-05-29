"""Tests for shared NeMo Platform endpoint defaults."""

from scripts import nemo_platform


def test_default_inference_gateway_derives_platform_openai_route(monkeypatch):
    monkeypatch.delenv("NIM_PROXY_URL", raising=False)
    monkeypatch.delenv("NMP_INFERENCE_GATEWAY_URL", raising=False)
    monkeypatch.setenv("NMP_BASE_URL", "http://nemo-platform-api:8080")
    monkeypatch.setenv("NMP_WORKSPACE", "default")

    assert nemo_platform.default_inference_gateway_url() == (
        "http://nemo-platform-api:8080/v2/workspaces/default/inference/gateway/openai/-"
    )


def test_default_inference_gateway_respects_explicit_proxy(monkeypatch):
    monkeypatch.setenv("NMP_BASE_URL", "http://nemo-platform-api:8080")
    monkeypatch.setenv("NMP_WORKSPACE", "default")
    monkeypatch.setenv("NMP_INFERENCE_GATEWAY_URL", "http://nemo-nim-proxy:8000")

    assert nemo_platform.default_inference_gateway_url() == "http://nemo-nim-proxy:8000"


def test_platform_openai_gateway_url_uses_workspace():
    assert (
        nemo_platform.platform_openai_gateway_url(
            "http://platform:8080/",
            "team-a",
        )
        == "http://platform:8080/v2/workspaces/team-a/inference/gateway/openai/-"
    )
