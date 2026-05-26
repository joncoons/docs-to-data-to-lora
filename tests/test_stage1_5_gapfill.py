"""Tests for Stage 1.5: bias analysis + Data Designer gap-fill."""
from unittest.mock import MagicMock
from collections import Counter

from scripts.pipeline.models import Passage, KVPRow
from scripts.pipeline.stage1_5_gapfill import (
    compute_bias_report, build_seed_styles, render_recipe_prompt,
)


def _row(product_family, q="Q?", a="A."):
    return KVPRow(
        passage_id="x#0", source_url="https://x.com",
        product_family=product_family, stage="1a", premise_index=0,
        question=q, answer=a, context="ctx", refined=False,
    )


def _passage(product_family):
    return Passage(
        passage_id=f"p_{product_family}", url="https://x.com",
        text="text body " * 50, token_count=100,
        chunk_ids=["1"], product_family=product_family, product_name=product_family,
        doc_kind="html",
    )


def test_compute_bias_report_flags_underrepresented():
    # 100 chunks for product A → many KVPs
    # 100 chunks for product B → few KVPs (under-represented)
    passages = (
        [_passage("A") for _ in range(100)] +
        [_passage("B") for _ in range(100)]
    )
    kvps = (
        [_row("A") for _ in range(200)] +   # density 2.0
        [_row("B") for _ in range(20)]      # density 0.2
    )
    report = compute_bias_report(passages, kvps, threshold_factor=0.5)
    by_family = {p["product_family"]: p for p in report["products"]}
    assert by_family["A"]["underrepresented"] is False
    assert by_family["B"]["underrepresented"] is True
    assert report["median_density"] > 0


def test_build_seed_styles_picks_three():
    rows = [_row("A", q=f"Q{i}?", a=f"A{i}.") for i in range(10)]
    styles = build_seed_styles(rows, product_family="A", n=3)
    assert len(styles) == 3


def test_render_recipe_prompt_substitutes_jinja_vars():
    prompt = render_recipe_prompt(
        retrieved_chunks="CHUNK1\nCHUNK2",
        product_family="nim-deploy",
        seed_styles=["- Q1?", "- Q2?", "- Q3?"],
    )
    assert "CHUNK1" in prompt
    assert "nim-deploy" in prompt
    assert "Q1?" in prompt
