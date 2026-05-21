"""Tests for prompt templates."""
from scripts.pipeline.prompts import (
    LE_SYSTEM, LE_USER,
    KVP_SYSTEM, KVP_USER,
    SYNTHESIS_SYSTEM, SYNTHESIS_USER,
    INSTRUCTION_SYSTEM, INSTRUCTION_USER,
    QA_EVAL_SYSTEM, QA_EVAL_USER,
    JUDGE_SYSTEM, JUDGE_USER,
    GAPFILL_RECIPE_USER,
)


def test_le_user_substitutes_text():
    rendered = LE_USER.format(text="my passage")
    assert "my passage" in rendered
    assert "conclusion" in rendered.lower()
    assert "premises" in rendered.lower()


def test_kvp_user_substitutes_three_fields():
    rendered = KVP_USER.format(premise="P1", conclusion="C1", text="T1")
    assert "P1" in rendered and "C1" in rendered and "T1" in rendered


def test_synthesis_user_substitutes_domain_and_context():
    rendered = SYNTHESIS_USER.format(
        domain="NVIDIA NIM",
        neighborhood_context="Passage A:\n...\nPassage B:\n...",
    )
    assert "NVIDIA NIM" in rendered
    assert "bridging" in rendered.lower()
    assert "contrastive" in rendered.lower()


def test_instruction_user_substitutes_passage_and_domain():
    rendered = INSTRUCTION_USER.format(domain="NVIDIA NIM", passage="some passage")
    assert "some passage" in rendered
    assert "summary" in rendered.lower()
    assert "listicle" in rendered.lower()
    assert "procedural" in rendered.lower()


def test_qa_eval_user_substitutes_question_answer_context():
    rendered = QA_EVAL_USER.format(question="Q?", answer="A.", context="ctx")
    assert "Q?" in rendered and "A." in rendered and "ctx" in rendered


def test_judge_user_substitutes_qa_and_context():
    rendered = JUDGE_USER.format(question="Q?", answer="A.", context="ctx")
    assert "grounded" in rendered.lower()
    assert "hallucination" in rendered.lower()


def test_gapfill_recipe_user_substitutes_chunks_and_product():
    rendered = GAPFILL_RECIPE_USER.format(
        retrieved_chunks="CHUNK1\nCHUNK2",
        product_family="nim-deploy",
        seed_styles="- example seed",
    )
    assert "CHUNK1" in rendered
    assert "nim-deploy" in rendered
    assert "5 question-answer pairs" in rendered
