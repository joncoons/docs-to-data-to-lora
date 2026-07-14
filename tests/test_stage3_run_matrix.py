"""Tests for orchestrator — wave structure and pair enumeration."""
import json
import sys
from itertools import combinations
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval.register_evaluator_entities import AdapterRow
import scripts.eval.run_evaluation_matrix as rem
from scripts.eval.run_evaluation_matrix import (
    build_reference_pairwise_jobs,
    build_pairwise_jobs,
    build_singleaxis_jobs,
    submit_wave,
    wait_all,
    write_job_map,
)


def _adapter(name, base, coll):
    return AdapterRow(name=name, base_model=base, job_id="cust-x",
                       collection=coll)


def _all_12():
    out = []
    for coll, short_coll in [("nim_curated", "nim"),
                              ("nemo_usvcs_curated", "nemo-usvcs")]:
        for base_short, base in [
            ("llama-3.2-1b", "meta/llama-3.2-1b-instruct"),
            ("llama-3.2-3b", "meta/llama-3.2-3b-instruct"),
            ("llama-3.1-8b", "meta/llama-3.1-8b-instruct"),
        ]:
            for r in (16, 32):
                out.append(_adapter(f"lora-{short_coll}-{base_short}-r{r}",
                                     base, coll))
    return out


def test_singleaxis_jobs_count_is_18():
    """12 adapters (1 ds each) + 3 bases (2 ds each) = 18.

    The 70B reference is not a Wave A target; it appears in Wave C pairwise.
    """
    jobs = build_singleaxis_jobs(
        adapters=_all_12(),
        config_name="default/stage3-singleaxis-rubric",
    )
    assert len(jobs) == 18


def test_singleaxis_includes_each_adapter_with_matching_corpus():
    jobs = build_singleaxis_jobs(
        adapters=_all_12(),
        config_name="default/stage3-singleaxis-rubric",
    )
    for adapter in _all_12():
        matches = [j for j in jobs
                   if j["target"] == f"default/{adapter.name}"]
        assert len(matches) == 1, f"adapter {adapter.name} not 1-job"
        ds = matches[0]["dataset"]
        if adapter.collection == "nim_curated":
            assert "nim-curated-test" in ds
        else:
            assert "nemo-usvcs-curated-test" in ds


def test_singleaxis_includes_each_base_on_both_corpora():
    jobs = build_singleaxis_jobs(
        adapters=_all_12(),
        config_name="default/stage3-singleaxis-rubric",
    )
    # 3 base targets, each appears in 2 jobs
    base_jobs = [j for j in jobs if j["target"] in {
        "default/llama-3.2-1b-instruct",
        "default/llama-3.2-3b-instruct",
        "default/llama-3.1-8b-instruct",
    }]
    assert len(base_jobs) == 6
    bases_seen = {j["target"] for j in base_jobs}
    assert len(bases_seen) == 3


def test_singleaxis_excludes_reference_comparator():
    jobs = build_singleaxis_jobs(
        adapters=_all_12(),
        config_name="default/stage3-singleaxis-rubric",
    )
    assert all("llama-3.3-70b" not in j["target"] for j in jobs)


def test_reference_pairwise_jobs_compare_reference_against_each_adapter():
    jobs = build_reference_pairwise_jobs(
        adapters=_all_12(),
        config_name="default/stage3-pairwise-tournament",
    )
    assert len(jobs) == 12
    assert all(j["target"] == "default/llama-3.3-70b-instruct" for j in jobs)
    assert all(j["extra"]["target_a"] == "default/llama-3.3-70b-instruct" for j in jobs)
    assert all(j["extra"]["target_b"].startswith("default/lora-") for j in jobs)


def test_pairwise_jobs_count_is_42():
    """Per corpus: 6 base-vs-adapter jobs + C(6,2)=15 rank/variant jobs."""
    jobs = build_pairwise_jobs(
        adapters=_all_12(),
        config_name="default/stage3-pairwise-tournament",
    )
    assert len(jobs) == 42


def test_pairwise_adapter_pairs_stay_within_corpus():
    jobs = build_pairwise_jobs(
        adapters=_all_12(),
        config_name="default/stage3-pairwise-tournament",
    )
    adapter_pairs = [j for j in jobs if j["extra"]["target_a"].startswith("default/lora-")]
    assert adapter_pairs
    for j in adapter_pairs:
        a = j["extra"]["target_a"]
        b = j["extra"]["target_b"]
        assert b.startswith("default/lora-")
        a_kind = "nim" if "-nim-" in a else "nemo-usvcs"
        b_kind = "nim" if "-nim-" in b else "nemo-usvcs"
        assert a_kind == b_kind, f"cross-corpus pair: {a} vs {b}"


def test_pairwise_includes_base_vs_each_adapter():
    jobs = build_pairwise_jobs(
        adapters=_all_12(),
        config_name="default/stage3-pairwise-tournament",
    )
    base_pairs = [j for j in jobs if not j["extra"]["target_a"].startswith("default/lora-")]
    assert len(base_pairs) == 12

    expected = {
        ("default/llama-3.2-1b-instruct", "default/lora-nim-llama-3.2-1b-r16"),
        ("default/llama-3.2-1b-instruct", "default/lora-nim-llama-3.2-1b-r32"),
        ("default/llama-3.2-3b-instruct", "default/lora-nim-llama-3.2-3b-r16"),
        ("default/llama-3.2-3b-instruct", "default/lora-nim-llama-3.2-3b-r32"),
        ("default/llama-3.1-8b-instruct", "default/lora-nim-llama-3.1-8b-r16"),
        ("default/llama-3.1-8b-instruct", "default/lora-nim-llama-3.1-8b-r32"),
        ("default/llama-3.2-1b-instruct", "default/lora-nemo-usvcs-llama-3.2-1b-r16"),
        ("default/llama-3.2-1b-instruct", "default/lora-nemo-usvcs-llama-3.2-1b-r32"),
        ("default/llama-3.2-3b-instruct", "default/lora-nemo-usvcs-llama-3.2-3b-r16"),
        ("default/llama-3.2-3b-instruct", "default/lora-nemo-usvcs-llama-3.2-3b-r32"),
        ("default/llama-3.1-8b-instruct", "default/lora-nemo-usvcs-llama-3.1-8b-r16"),
        ("default/llama-3.1-8b-instruct", "default/lora-nemo-usvcs-llama-3.1-8b-r32"),
    }
    actual = {(j["extra"]["target_a"], j["extra"]["target_b"]) for j in base_pairs}
    assert actual == expected


def test_submit_wave_invokes_client_per_job():
    client = MagicMock()
    client.submit_job.side_effect = [f"ej-{i:03d}" for i in range(3)]
    jobs = [{"config": "c", "target": "t1", "dataset": "d"},
            {"config": "c", "target": "t2", "dataset": "d"},
            {"config": "c", "target": "t3", "dataset": "d"}]

    ids = submit_wave(client, jobs)

    assert ids == ["ej-000", "ej-001", "ej-002"]
    assert client.submit_job.call_count == 3


def test_wait_all_aborts_after_consecutive_failures():
    """If a job fails to poll N times in a row, raise RuntimeError."""
    client = MagicMock()
    client.get_status.side_effect = ConnectionError("network down")

    with patch("scripts.eval.run_evaluation_matrix.time.sleep"):
        with pytest.raises(RuntimeError, match="consecutive polls"):
            wait_all(client, job_ids=["ej-001"],
                     poll_interval=0.0, max_wait_s=60,
                     max_consecutive_errors=3)


def test_write_job_map_creates_parent_directory(tmp_path):
    out = tmp_path / "evals" / "evaluator_job_ids.json"
    payload = {"wave_a": [["job-1", {"target": "default/model"}]]}

    write_job_map(out, payload)

    assert json.loads(out.read_text()) == payload


def test_main_submit_only_writes_all_wave_ids_without_polling(tmp_path, monkeypatch):
    log_path = tmp_path / "training_session.log"
    log_path.write_text(
        "| lora-nim-llama-3.2-3b-r16 | cust-abc | 1.0 | 1.0 | ~5 min |\n"
    )
    out = tmp_path / "outputs" / "evaluator_job_ids.json"
    submitted = []

    class DummyClient:
        def __init__(self, base_url, api_key=None):
            assert base_url == "http://evaluator.test"
            assert api_key == "secret-token"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def submit_job(self, payload):
            submitted.append(payload)
            return f"job-{len(submitted)}"

    wait_mock = MagicMock()
    monkeypatch.setattr(rem, "EvaluatorClient", DummyClient)
    monkeypatch.setattr(rem, "wait_all", wait_mock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluation_matrix.py",
            "--evaluator-url",
            "http://evaluator.test",
            "--evaluator-api-key",
            "secret-token",
            "--log-path",
            str(log_path),
            "--out",
            str(out),
            "--wave",
            "all",
            "--submit-only",
        ],
    )

    assert rem.main() == 0

    assert wait_mock.call_count == 0
    assert len(submitted) == 5
    data = json.loads(out.read_text())
    assert set(data) == {"wave_a", "wave_b", "wave_c"}
    assert len(data["wave_a"]) == 3
    assert len(data["wave_b"]) == 1
    assert len(data["wave_c"]) == 1
