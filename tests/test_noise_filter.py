"""Tests for the Stage 0 noise filter rules."""
from scripts.pipeline.noise_filter import is_noise


def test_short_passage_dropped():
    assert is_noise("Short.", url="") is True


def test_long_normal_passage_kept():
    text = "The NIM_MODEL_PROFILE env var selects the engine variant. " * 20
    assert is_noise(text, url="https://docs.nvidia.com/nim/foo.html") is False


def test_apache_license_prefix_dropped():
    text = "Apache License Version 2.0, January 2004 " + "x " * 200
    assert is_noise(text, url="https://x.com/license") is True


def test_license_url_segment_dropped():
    text = "x " * 200
    assert is_noise(text, url="https://x.com/legal/third-party-notices.html") is True


def test_license_boilerplate_body_dropped():
    text = "Some preamble. " + ("WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND " * 5) + "more text. " * 50
    assert is_noise(text, url="https://x.com/foo") is True


def test_nav_link_heavy_passage_dropped():
    text = "\n".join(["- [link](https://x.com)"] * 20)
    assert is_noise(text, url="https://x.com/index") is True


def test_pure_code_block_dropped():
    body_code = "```\n" + ("x = 1\n" * 100) + "```"
    assert is_noise(body_code, url="https://x.com/code") is True


def test_mixed_text_and_code_kept():
    text = "Here's how to do it. " * 30 + "\n```\nx = 1\n```\n" + "And that's it. " * 30
    assert is_noise(text, url="https://x.com/howto") is False
