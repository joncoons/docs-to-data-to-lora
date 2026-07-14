"""Tests for direct Kimi pairwise judging over saved completions."""

import json

from scripts.eval.run_direct_kimi_pairwise import (
    build_judge_user,
    build_pair_rows,
    coerce_pairwise_payload,
    map_winner_to_side,
    pair_output_dir,
    parse_json_object,
    resolve_position_swap,
    summarize_pairwise,
    PairSpec,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_judge_user_splits_context_and_swaps_answers():
    left = {
        "prompt": "Context:\nFact one.\n\nQuestion: What is fact one?",
        "reference_completion": "Fact one is true.",
        "response": "left answer",
    }
    right = {"response": "right answer"}

    user = build_judge_user(left, right, swapped=True)

    assert "Question:\nWhat is fact one?" in user
    assert "Source context:\nContext:\nFact one." in user
    assert "Reference answer:\nFact one is true." in user
    assert "Answer A:\nright answer" in user
    assert "Answer B:\nleft answer" in user


def test_parse_json_object_and_coerce_winner_strip_think_tags():
    parsed = parse_json_object('<think>private</think>\n{"winner": "b", "reason": "better"}')
    payload = coerce_pairwise_payload(parsed)

    assert payload == {"winner": "B", "reason": "better"}


def test_map_and_resolve_position_swap():
    assert map_winner_to_side("A", swapped=False) == "left"
    assert map_winner_to_side("B", swapped=False) == "right"
    assert map_winner_to_side("A", swapped=True) == "right"
    assert map_winner_to_side("B", swapped=True) == "left"

    assert resolve_position_swap(["left", "left"]) == ("left", "agree")
    assert resolve_position_swap(["tie", "right"]) == ("right", "weak")
    assert resolve_position_swap(["left", "right"]) == ("tie", "conflict")


def test_build_pair_rows_aligns_by_source_row_index_and_skips_completed():
    left = {
        0: {"source_row_index": 0},
        1: {"source_row_index": 1},
        2: {"source_row_index": 2},
    }
    right = {
        1: {"source_row_index": 1},
        2: {"source_row_index": 2},
        3: {"source_row_index": 3},
    }

    rows = build_pair_rows(left, right, completed={1}, limit=None)

    assert rows == [(1, {"source_row_index": 2}, {"source_row_index": 2})]


def test_pair_output_dir_uses_dataset_and_labels(tmp_path):
    left_path = tmp_path / "responses-left.jsonl"
    right_path = tmp_path / "responses-right.jsonl"
    _write_jsonl(left_path, [{"source_row_index": 0, "dataset_slug": "nim_curated"}])
    _write_jsonl(right_path, [{"source_row_index": 0, "dataset_slug": "nim_curated"}])

    out_dir = pair_output_dir(
        PairSpec("1b/r16", left_path, "1b/r32", right_path),
        tmp_path / "pairwise-kimi",
        "run1",
    )

    assert out_dir == tmp_path / "pairwise-kimi" / "nim_curated" / "1b-r16__vs__1b-r32" / "run1"


def test_summarize_pairwise_counts_wins_and_tokens(tmp_path):
    pairwise = tmp_path / "pairwise.jsonl"
    errors = tmp_path / "errors.jsonl"
    _write_jsonl(pairwise, [
        {
            "winner": "left",
            "agreement": "agree",
            "target_token_counts": {
                "left": {"total_tokens_raw": 10},
                "right": {"total_tokens_raw": 12},
            },
            "outcomes": [
                {"judge": {"token_counts": {"prompt_tokens": 5, "completion_tokens_raw": 1, "total_tokens_raw": 6}}},
                {"judge": {"token_counts": {"prompt_tokens": 5, "completion_tokens_raw": 1, "total_tokens_raw": 6}}},
            ],
        },
        {
            "winner": "tie",
            "agreement": "conflict",
            "target_token_counts": {
                "left": {"total_tokens_raw": 20},
                "right": {"total_tokens_raw": 22},
            },
            "outcomes": [
                {"judge": {"token_counts": {"prompt_tokens": 7, "completion_tokens_raw": 1, "total_tokens_raw": 8}}},
            ],
        },
    ])
    _write_jsonl(errors, [{"error": "bad"}])

    summary = summarize_pairwise(pairwise, errors)

    assert summary["rows_compared"] == 2
    assert summary["rows_failed"] == 1
    assert summary["wins"] == {"left": 1, "right": 0, "tie": 1}
    assert summary["agreement_counts"] == {"agree": 1, "conflict": 1}
    assert summary["target_generation"]["combined_total_tokens_raw"] == 64
    assert summary["judge_scoring"]["total_tokens_raw"] == 20
    assert summary["combined_total_tokens_raw"] == 84
