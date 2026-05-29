"""Tests for pure functions in build_moe_shards.py."""
import pytest

from scripts.stage3.build_moe_shards import (
    stratified_2way_split,
    fold_system_into_prompt,
    shard_dataset_name,
)


# ---------------------------------------------------------------------------
# shard_dataset_name
# ---------------------------------------------------------------------------

def test_shard_dataset_name_nim_a():
    assert shard_dataset_name("nim_curated", "a") == "stage3-nim-curated-shard-a"


def test_shard_dataset_name_nim_b():
    assert shard_dataset_name("nim_curated", "b") == "stage3-nim-curated-shard-b"


def test_shard_dataset_name_nemo_usvcs_a():
    assert shard_dataset_name("nemo_usvcs_curated", "a") == "stage3-nemo-usvcs-curated-shard-a"


def test_shard_dataset_name_nemo_usvcs_b():
    assert shard_dataset_name("nemo_usvcs_curated", "b") == "stage3-nemo-usvcs-curated-shard-b"


# ---------------------------------------------------------------------------
# fold_system_into_prompt
# ---------------------------------------------------------------------------

def test_fold_system_into_prompt_with_system():
    rows = [{"prompt": "hello", "completion": "world", "system": "You are helpful."}]
    out = fold_system_into_prompt(rows)
    assert out == [{"prompt": "You are helpful.\n\nhello", "completion": "world"}]


def test_fold_system_into_prompt_empty_system():
    """Empty/missing system field → prompt unchanged."""
    rows = [{"prompt": "hello", "completion": "world", "system": ""}]
    out = fold_system_into_prompt(rows)
    assert out == [{"prompt": "hello", "completion": "world"}]


def test_fold_system_into_prompt_missing_system_key():
    """Missing system key is treated as empty."""
    rows = [{"prompt": "q", "completion": "a"}]
    out = fold_system_into_prompt(rows)
    assert out == [{"prompt": "q", "completion": "a"}]


def test_fold_system_into_prompt_no_mutation():
    """Original rows are not mutated."""
    original = [{"prompt": "q", "completion": "a", "system": "sys"}]
    import copy
    original_copy = copy.deepcopy(original)
    fold_system_into_prompt(original)
    assert original == original_copy


# ---------------------------------------------------------------------------
# stratified_2way_split
# ---------------------------------------------------------------------------

def _make_rows(n: int, stages: list[str]) -> list[dict]:
    """Create n rows cycling through `stages`."""
    return [
        {"prompt": f"q{i}", "completion": f"a{i}", "stage": stages[i % len(stages)]}
        for i in range(n)
    ]


def test_stratified_split_total_rows_preserved():
    rows = _make_rows(100, ["A", "B", "C"])
    a, b = stratified_2way_split(rows, stratify_key="stage", seed=42)
    assert len(a) + len(b) == 100


def test_stratified_split_roughly_equal():
    rows = _make_rows(100, ["A", "B"])
    a, b = stratified_2way_split(rows, stratify_key="stage", seed=42)
    # Each shard should be ~50 ± 3
    assert abs(len(a) - 50) <= 3
    assert abs(len(b) - 50) <= 3


def test_stratified_split_stage_distribution_preserved():
    """Each shard should contain roughly equal counts of each stage."""
    rows = _make_rows(200, ["A", "B", "C", "D"])
    a, b = stratified_2way_split(rows, stratify_key="stage", seed=42)

    def stage_counts(shard):
        counts = {}
        for r in shard:
            counts[r["stage"]] = counts.get(r["stage"], 0) + 1
        return counts

    ca = stage_counts(a)
    cb = stage_counts(b)
    for stage in ["A", "B", "C", "D"]:
        # Each stage (50 rows total) should be ≈25 in each shard
        assert abs(ca.get(stage, 0) - 25) <= 3, f"shard-a stage {stage}: {ca.get(stage, 0)}"
        assert abs(cb.get(stage, 0) - 25) <= 3, f"shard-b stage {stage}: {cb.get(stage, 0)}"


def test_stratified_split_no_overlap():
    """A shard and B shard must have disjoint rows."""
    rows = _make_rows(60, ["X", "Y"])
    a, b = stratified_2way_split(rows, stratify_key="stage", seed=42)
    prompts_a = {r["prompt"] for r in a}
    prompts_b = {r["prompt"] for r in b}
    assert prompts_a.isdisjoint(prompts_b)


def test_stratified_split_deterministic():
    rows = _make_rows(100, ["A", "B"])
    a1, b1 = stratified_2way_split(rows, stratify_key="stage", seed=42)
    a2, b2 = stratified_2way_split(rows, stratify_key="stage", seed=42)
    assert [r["prompt"] for r in a1] == [r["prompt"] for r in a2]
    assert [r["prompt"] for r in b1] == [r["prompt"] for r in b2]


def test_stratified_split_fallback_when_no_stage_key():
    """Falls back to plain shuffle when stratify_key is absent from all rows."""
    rows = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(40)]
    a, b = stratified_2way_split(rows, stratify_key="stage", seed=42)
    assert len(a) + len(b) == 40


def test_stratified_split_different_seeds_produce_different_partitions():
    rows = _make_rows(100, ["A", "B"])
    a1, _ = stratified_2way_split(rows, stratify_key="stage", seed=1)
    a2, _ = stratified_2way_split(rows, stratify_key="stage", seed=99)
    # With 100 rows it would be astronomically unlikely for both to be identical
    assert [r["prompt"] for r in a1] != [r["prompt"] for r in a2]
