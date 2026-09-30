from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_character_engine.training import (
    AdapterAcceptanceGate,
    AdapterEvaluationMetrics,
    CharacterSFTDataset,
    ChatMessage,
    HuggingFaceSFTBackend,
    LoRAConfigSpec,
    QuantizationConfigSpec,
    SFTExample,
    SFTTrainingConfig,
    build_training_plan,
)


def example(i: int = 1, answer: str = "你好。") -> SFTExample:
    return SFTExample(
        example_id=f"e{i}",
        messages=(
            ChatMessage("system", "你是一個測試角色。"),
            ChatMessage("user", f"問題 {i}"),
            ChatMessage("assistant", answer),
        ),
        metadata={"source": "test"},
    )


def test_sft_example_requires_user_assistant_and_final_assistant():
    with pytest.raises(ValueError, match="user"):
        SFTExample("x", (ChatMessage("assistant", "a"),))
    with pytest.raises(ValueError, match="end with an assistant"):
        SFTExample("x", (ChatMessage("user", "q"), ChatMessage("assistant", "a"), ChatMessage("user", "q2")))


def test_jsonl_roundtrip_and_hf_rows(tmp_path: Path):
    dataset = CharacterSFTDataset((example(1), example(2)))
    path = tmp_path / "data.jsonl"
    dataset.to_jsonl(path)
    loaded = CharacterSFTDataset.from_jsonl(path)
    assert loaded == dataset
    assert loaded.to_hf_rows()[0]["messages"][-1]["role"] == "assistant"


def test_dataset_validation_duplicate_warning_and_stats():
    one = example(1)
    two = SFTExample("e2", one.messages)
    report = CharacterSFTDataset((one, two)).validate()
    assert report.valid
    assert report.stats.examples == 2
    assert report.stats.assistant_messages == 2
    assert any(issue.code == "duplicate_content" for issue in report.issues)


def test_dataset_fingerprint_is_order_independent_and_content_sensitive():
    a = CharacterSFTDataset((example(1), example(2)))
    b = CharacterSFTDataset((example(2), example(1)))
    c = CharacterSFTDataset((example(1), example(2, "不同回答")))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_deterministic_split_is_stable_and_nonempty():
    dataset = CharacterSFTDataset(tuple(example(i) for i in range(10)))
    t1, e1 = dataset.deterministic_split(0.2, seed=7)
    t2, e2 = dataset.deterministic_split(0.2, seed=7)
    assert [x.example_id for x in t1.examples] == [x.example_id for x in t2.examples]
    assert [x.example_id for x in e1.examples] == [x.example_id for x in e2.examples]
    assert len(t1.examples) == 8
    assert len(e1.examples) == 2


def test_training_config_validation_and_plan_warnings():
    dataset = CharacterSFTDataset(tuple(example(i) for i in range(6)))
    config = SFTTrainingConfig(
        base_model="org/model",
        output_dir="out",
        eval_ratio=0.2,
        quantization=QuantizationConfigSpec(mode="4bit"),
        lora=LoRAConfigSpec(r=8, alpha=16),
    )
    plan = build_training_plan(dataset, config)
    assert plan.train_examples + plan.eval_examples == 6
    assert any("chat template" in warning for warning in plan.warnings)
    assert any("bitsandbytes" in warning for warning in plan.warnings)


def test_backend_compile_is_gpu_free_and_contains_no_secret_fields():
    dataset = CharacterSFTDataset(tuple(example(i) for i in range(4)))
    config = SFTTrainingConfig(base_model="org/model", output_dir="out", eval_ratio=0.25)
    compiled = HuggingFaceSFTBackend().compile(dataset, config)
    assert compiled["backend"] == "huggingface-trl-peft"
    assert compiled["method"] == "lora"
    assert compiled["plan"]["train_examples"] == 3
    assert "api_key" not in json.dumps(compiled)


def test_adapter_acceptance_gate_blocks_regression():
    gate = AdapterAcceptanceGate()
    baseline = AdapterEvaluationMetrics(persona_consistency=0.96, task_success=0.9)
    candidate = AdapterEvaluationMetrics(persona_consistency=0.90, task_success=0.9)
    result = gate.evaluate(candidate, baseline=baseline)
    assert result.accepted is False
    assert any("regressed" in reason for reason in result.reasons)


def test_adapter_acceptance_gate_accepts_good_candidate():
    gate = AdapterAcceptanceGate()
    result = gate.evaluate(AdapterEvaluationMetrics(0.95, 0.9, 0.01))
    assert result.accepted is True
    assert result.reasons == ()


def test_cli_validate_and_plan(tmp_path: Path, monkeypatch):
    # Test core behavior through direct module invocation data, avoiding subprocess PYTHONPATH concerns.
    data = CharacterSFTDataset((example(1), example(2), example(3)))
    path = tmp_path / "dataset.jsonl"
    data.to_jsonl(path)
    assert CharacterSFTDataset.from_jsonl(path).validate().valid
    compiled = HuggingFaceSFTBackend().compile(
        data,
        SFTTrainingConfig(base_model="org/model", output_dir="out", eval_ratio=0.33),
    )
    assert compiled["schema_version"] == "0.21"
