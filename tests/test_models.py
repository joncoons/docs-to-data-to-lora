"""Tests for Pydantic schemas — LogEntailment, QAKeyValuePair, QAEvaluation, etc."""
from scripts.pipeline.models import (
    LogEntailment,
    LogEntailmentList,
    QAKeyValuePair,
    QAEvaluation,
    SynthesisPairs,
    InstructionPairs,
    Passage,
    KVPRow,
)


def test_log_entailment_required_fields():
    le = LogEntailment(
        conclusion="The runtime supports FP8 quantization.",
        premises=["FP8 is enabled via NIM_MODEL_PROFILE.", "Profile hashes start with 'fp8-'."],
        context="Widget docs section 3.",
        entities="runtime, NIM_MODEL_PROFILE, FP8",
        recommendations="Set the profile hash before deploy.",
    )
    assert le.conclusion.startswith("The runtime")
    assert len(le.premises) == 2


def test_log_entailment_premises_min_one():
    """Premises must have at least one entry."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LogEntailment(
            conclusion="X.",
            premises=[],  # empty list should fail
            context="", entities="", recommendations="",
        )


def test_qa_key_value_pair_basic():
    kvp = QAKeyValuePair(question="What does NIM_MODEL_PROFILE control?",
                         answer="It selects the model engine variant.")
    assert "?" in kvp.question
    assert len(kvp.answer) > 10


def test_qa_evaluation_basic():
    ev = QAEvaluation(prompt="What is X?", completion="X is a thing.")
    assert ev.prompt == "What is X?"
    assert ev.admit is True


def test_qa_evaluation_can_reject_without_rewrite():
    ev = QAEvaluation(admit=False, grounded=False, reason="unsupported")
    assert ev.admit is False
    assert ev.prompt == ""
    assert ev.reason == "unsupported"


def test_synthesis_pairs_two_types():
    sp = SynthesisPairs(pairs=[
        {"type": "bridging", "question": "A given B?", "answer": "A."},
        {"type": "contrastive", "question": "How differ A vs B?", "answer": "A is X, B is Y."},
    ])
    assert len(sp.pairs) == 2
    assert {p["type"] for p in sp.pairs} == {"bridging", "contrastive"}


def test_instruction_pairs_optional_procedural():
    ip = InstructionPairs(pairs=[
        {"type": "summary", "question": "Summarize.", "answer": "It's X."},
        {"type": "listicle", "question": "List items.", "answer": "1. A. 2. B."},
    ])
    assert len(ip.pairs) == 2


def test_passage_required_fields():
    p = Passage(
        passage_id="https://x.com/y#p0",
        url="https://x.com/y",
        text="Some passage text that is at least 60 tokens long..." * 5,
        token_count=500,
        chunk_ids=["es_id_1", "es_id_2"],
        product_family="nim-deploy",
        product_name="nim-llm",
        doc_kind="html",
    )
    assert p.doc_kind == "html"


def test_kvp_row_serialization():
    row = KVPRow(
        passage_id="x#p0", source_url="https://x.com/y",
        product_family="nim-deploy", stage="1a", premise_index=0,
        question="Q?", answer="A.", context="ctx", refined=False,
    )
    j = row.model_dump_json()
    assert '"stage":"1a"' in j


def test_log_entailment_list_min_one():
    """LogEntailmentList rejects empty list."""
    import pytest
    from pydantic import ValidationError
    from scripts.pipeline.models import LogEntailmentList
    with pytest.raises(ValidationError):
        LogEntailmentList(entailments=[])


def test_log_entailment_list_carries_entries():
    from scripts.pipeline.models import LogEntailment, LogEntailmentList
    lel = LogEntailmentList(entailments=[
        LogEntailment(conclusion="A.", premises=["P1."]),
        LogEntailment(conclusion="B.", premises=["P2.", "P3."]),
    ])
    assert len(lel.entailments) == 2
