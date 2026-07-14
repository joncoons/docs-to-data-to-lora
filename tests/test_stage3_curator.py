"""Tests for Stage 3 — Curator dedup, quality filter, train/val split."""
import json

from scripts.pipeline.models import KVPRow
from scripts.pipeline.stage3_curator import (
    exact_dedup, length_filter, answer_subset_of_question_filter,
    run_stage3, train_val_split,
)


class WhitespaceTokenizer:
    name_or_path = "test-whitespace-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return text.split()


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
    out = length_filter(
        [short_q, short_a, long_ok],
        min_q_tokens=8,
        min_a_tokens=25,
        tokenizer=WhitespaceTokenizer(),
    )
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


def test_run_stage3_writes_reduction_summary(tmp_path):
    good = _row(q="What configuration settings control NIM model deployment profiles?",
                a="NIM deployment profiles are controlled by model profile settings, "
                  "engine selection, quantization mode, tensor parallelism, and "
                  "runtime deployment parameters.")
    duplicate = _row(q=good.question,
                     a="This duplicate question should be removed before later filters.")
    short_question = _row(q="Hi?",
                          a="This answer is intentionally long enough to pass the "
                            "answer token threshold, but the question is too short.")
    answer_subset = _row(q="How does NVIDIA use NVIDIA in NVIDIA documentation examples?",
                         a="NVIDIA")

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text(
        '{"version":"1.0","truncation":null,"padding":null,'
        '"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},'
        '"post_processor":null,"decoder":null,"model":{"type":"WordLevel",'
        '"vocab":{"[UNK]":0,"What":1,"configuration":2,"settings":3,"control":4,'
        '"NIM":5,"model":6,"deployment":7,"profiles?":8,"profiles":9,"are":10,'
        '"controlled":11,"by":12,"profile":13,"settings,":14,"engine":15,'
        '"selection,":16,"quantization":17,"mode,":18,"tensor":19,"parallelism,":20,'
        '"and":21,"runtime":22,"parameters.":23,"Hi?":24,"This":25,"answer":26,'
        '"is":27,"intentionally":28,"long":29,"enough":30,"to":31,"pass":32,'
        '"the":33,"token":34,"threshold,":35,"but":36,"question":37,"too":38,'
        '"short.":39,"How":40,"does":41,"NVIDIA":42,"use":43,"in":44,'
        '"documentation":45,"examples?":46,"duplicate":47,"should":48,"be":49,'
        '"removed":50,"before":51,"later":52,"filters.":53},"unk_token":"[UNK]"}}'
    )
    (tokenizer_dir / "tokenizer_config.json").write_text('{"unk_token":"[UNK]"}')

    run_stage3(
        [good, duplicate, short_question, answer_subset],
        tmp_path,
        "system",
        train_ratio=1.0,
        minhash_threshold=0.85,
        min_q_tokens=8,
        min_a_tokens=1,
        tokenizer_name_or_path=str(tokenizer_dir),
    )

    summary = json.loads((tmp_path / "stage3_curator_summary.json").read_text())
    assert summary["counts"]["input_rows"] == 4
    assert summary["counts"]["after_exact_dedup"] == 3
    assert summary["counts"]["after_length_filter"] == 2
    assert summary["counts"]["after_subset_filter"] == 1
    assert summary["retention"]["removed_rows"] == 3
    assert summary["parameters"]["tokenizer_name_or_path"] == str(tokenizer_dir)
    assert (tmp_path / "stage3_curator_summary.md").exists()
