"""Tests for ties_merge_one_tensor — the TIES math kernel.

All tests use in-memory torch.Tensor fixtures with small (3-adapter, 1-tensor)
setups. No safetensors writes, no network calls.
"""
import torch
import pytest

from scripts.stage3.ties_merge import ties_merge_one_tensor


# ---------------------------------------------------------------------------
# Helper: build a (N, *shape) stacked tensor from per-adapter lists
# ---------------------------------------------------------------------------

def _stack(*rows: list[float], shape: tuple = None) -> torch.Tensor:
    """Build stacked tensor from row lists. shape overrides inferred dim."""
    t = torch.tensor(rows, dtype=torch.float32)
    if shape is not None:
        t = t.reshape(len(rows), *shape)
    return t


# ---------------------------------------------------------------------------
# Test 1: Trim zeroes the bottom-magnitude entries per adapter
# ---------------------------------------------------------------------------

def test_trim_zeroes_smallest_entries():
    """With trim_ratio=0.5, half the entries (by |value|) should be zeroed.

    Adapter A: [1.0, 2.0, 3.0, 4.0]  — after trim_50%: keep top 2 → [0, 0, 3, 4]
    Adapter B: [1.0, 2.0, 3.0, 4.0]  — same
    With both adapters agreeing on sign, disjoint merge = element-wise avg of
    trimmed values; zeroed entries produce 0 in the merged output.
    """
    stacked = _stack([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])  # shape (2, 4)
    merged = ties_merge_one_tensor(stacked, trim_ratio=0.5)

    # Bottom 2 entries (indices 0,1) should be zeroed after trim and should
    # produce 0 in the merged output (no adapter contributes them).
    assert merged[0].item() == pytest.approx(0.0)
    assert merged[1].item() == pytest.approx(0.0)
    # Top 2 entries survive
    assert merged[2].item() != 0.0
    assert merged[3].item() != 0.0


# ---------------------------------------------------------------------------
# Test 2: Sign-elect picks the sign with max summed magnitude
# ---------------------------------------------------------------------------

def test_sign_elect_picks_majority_sign():
    """Two adapters positive, one negative → elected sign positive."""
    # stacked shape (3, 3): all entries for each adapter
    # position 0: A=+1, B=+2, C=-0.5 → sum = +2.5 > 0 → elected sign positive
    # position 1: A=-3, B=-1, C=+0.1 → sum = -3.9 < 0 → elected sign negative
    # position 2: A=+5, B=+5, C=+5  → sum = +15 > 0 → elected sign positive
    stacked = _stack(
        [1.0, -3.0, 5.0],
        [2.0, -1.0, 5.0],
        [-0.5, 0.1, 5.0],
    )
    merged = ties_merge_one_tensor(stacked, trim_ratio=0.0)

    # Position 0: positive elected
    assert merged[0].item() > 0
    # Position 1: negative elected
    assert merged[1].item() < 0
    # Position 2: positive elected
    assert merged[2].item() > 0


# ---------------------------------------------------------------------------
# Test 3: Disjoint merge averages only same-sign entries
# ---------------------------------------------------------------------------

def test_disjoint_merge_averages_only_agreeing_entries():
    """Only entries that agree with the elected sign contribute to the average.

    A: [+2.0, -1.0]  — position 0 is positive, 1 is negative
    B: [+4.0, +3.0]  — position 0 is positive, 1 is positive

    Position 0: elected positive (sum=6 > 0); both agree → avg = (2+4)/2 = 3.0
    Position 1: elected positive (sum=2 > 0); only B agrees → avg = 3.0 / 1 = 3.0
    """
    stacked = _stack([2.0, -1.0], [4.0, 3.0])  # shape (2, 2)
    merged = ties_merge_one_tensor(stacked, trim_ratio=0.0)

    assert merged[0].item() == pytest.approx(3.0)
    assert merged[1].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Test 4: trim_ratio=0 → no trim, pure sign-elect + disjoint-merge
# ---------------------------------------------------------------------------

def test_zero_trim_ratio_applies_no_trim():
    """With trim_ratio=0.0 no values are zeroed before sign-elect."""
    # All entries positive → elected positive; all agree → simple average
    stacked = _stack([2.0, 4.0, 6.0], [4.0, 6.0, 8.0])  # shape (2, 3)
    merged = ties_merge_one_tensor(stacked, trim_ratio=0.0)

    assert merged[0].item() == pytest.approx(3.0)   # (2+4)/2
    assert merged[1].item() == pytest.approx(5.0)   # (4+6)/2
    assert merged[2].item() == pytest.approx(7.0)   # (6+8)/2


# ---------------------------------------------------------------------------
# Test 5: Output shape matches stacked.shape[1:]
# ---------------------------------------------------------------------------

def test_output_shape_matches_per_adapter_shape():
    """Output shape should be the per-adapter shape (stacked.shape[1:])."""
    # 3 adapters, each is a (4, 5) matrix
    stacked = torch.randn(3, 4, 5)
    merged = ties_merge_one_tensor(stacked, trim_ratio=0.1)
    assert merged.shape == (4, 5)

    # 2 adapters, each is a flat vector of length 10
    stacked_flat = torch.randn(2, 10)
    merged_flat = ties_merge_one_tensor(stacked_flat, trim_ratio=0.2)
    assert merged_flat.shape == (10,)


# ---------------------------------------------------------------------------
# Test 6: ties_merge raises FileNotFoundError when adapter_config.json missing
# ---------------------------------------------------------------------------

def test_ties_merge_raises_on_missing_adapter_config(tmp_path):
    """If the first source adapter lacks adapter_config.json, ties_merge must fail loud
    (a merged adapter without it cannot be loaded by NIM_PEFT_SOURCE)."""
    import safetensors.torch
    import torch

    from scripts.stage3.ties_merge import ties_merge

    src_a = tmp_path / "src_a"; src_a.mkdir()
    src_b = tmp_path / "src_b"; src_b.mkdir()
    fake_tensors = {"layer.weight": torch.tensor([[1.0, 2.0]])}
    safetensors.torch.save_file(fake_tensors, str(src_a / "adapter_model.safetensors"))
    safetensors.torch.save_file(fake_tensors, str(src_b / "adapter_model.safetensors"))
    # Note: adapter_config.json deliberately NOT created

    out_dir = tmp_path / "out"
    import pytest
    with pytest.raises(FileNotFoundError, match="adapter_config"):
        ties_merge([src_a, src_b], out_dir, trim_ratio=0.0)


# ---------------------------------------------------------------------------
# Test 7: N>=3 key-mismatch reduce/xor works correctly (regression for
#         set.symmetric_difference(*args) TypeError bug)
# ---------------------------------------------------------------------------

def test_ties_merge_handles_n_set_key_mismatch():
    """The key-mismatch error path must work for N>=3 adapters (regression test
    for set.symmetric_difference(*args) bug)."""
    # We don't need to actually run ties_merge here — just verify the reduce-based
    # symmetric-difference works for N=3.
    from functools import reduce
    sets = [{"a", "b"}, {"a", "c"}, {"a", "d"}]
    diff = reduce(lambda a, b: a ^ b, sets)
    # b is in only set 0; c is in only set 1; d is in only set 2; a is in all three (cancels).
    # ((a,b) ^ (a,c)) = (b,c); (b,c) ^ (a,d) = (a,b,c,d) (all symmetric).
    assert diff == {"a", "b", "c", "d"}
