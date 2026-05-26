"""Tests for Stage 3 — Curator dedup, quality filter, train/val split."""
from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage3_curator import (
    exact_dedup, length_filter, answer_subset_of_question_filter,
    train_val_split,
)


def _row(stage="1a",
         q="What are the key configuration parameters for NIM model profiles?",
         a="NIM model profiles control the engine variant by specifying a hash identifier "
           "that selects among FP8, BF16, NVFP4 and other quantization modes along with "
           "tensor parallelism settings.",
         product_family="nim"):
    return KVPRow(
        passage_id="p", source_url="u", product_family=product_family, stage=stage,
        question=q, answer=a, context="ctx", refined=False,
    )


def test_exact_dedup_on_question():
    rows = [_row(q="Same?"), _row(q="Same?"), _row(q="Different?")]
    deduped = exact_dedup(rows)
    assert len(deduped) == 2


def test_length_filter_drops_short():
    short_q = _row(q="Hi?")
    short_a = _row(a="Yes.")
    long_ok = _row()
    out = length_filter([short_q, short_a, long_ok], min_q_tokens=8, min_a_tokens=25)
    assert len(out) == 1


def test_answer_subset_filter():
    bad = _row(q="What is NVIDIA?", a="NVIDIA.")
    good = _row(q="What does NIM_MODEL_PROFILE control?",
                a="It controls the engine variant by hash.")
    out = answer_subset_of_question_filter([bad, good])
    assert len(out) == 1
    assert out[0].question == good.question


def test_train_val_split_stratifies_by_stage():
    rows = [_row(stage="1a") for _ in range(50)] + [_row(stage="1b") for _ in range(50)]
    train, val = train_val_split(rows, train_ratio=0.9, seed=42)
    train_stages = {r.stage for r in train}
    val_stages = {r.stage for r in val}
    assert "1a" in train_stages and "1b" in train_stages
    assert "1a" in val_stages and "1b" in val_stages
    assert 85 <= len(train) <= 95
    assert 5 <= len(val) <= 15
