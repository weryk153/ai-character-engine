from __future__ import annotations

from dataclasses import replace

from .dataset import CharacterSFTDataset
from .models import SFTTrainingConfig, TrainingPlan


def build_training_plan(dataset: CharacterSFTDataset, config: SFTTrainingConfig) -> TrainingPlan:
    report = dataset.validate()
    if not report.valid:
        raise ValueError("dataset contains validation errors")
    train, evaluation = dataset.deterministic_split(config.eval_ratio, seed=config.seed)
    warnings: list[str] = []
    if len(dataset.examples) < 50:
        warnings.append("dataset is very small; treat the run as a pipeline smoke test, not evidence of character quality")
    if config.assistant_only_loss:
        warnings.append(
            "assistant_only_loss requires a compatible chat template/generation mask; verify the base model tokenizer before training"
        )
    if config.quantization.mode in {"4bit", "8bit"}:
        warnings.append("quantized training requires backend/device support; bitsandbytes is not portable to every platform")
    if not config.lora.target_modules:
        warnings.append("LoRA target_modules are automatic; confirm the resolved modules for the chosen architecture")
    return TrainingPlan(
        config=config,
        train_examples=len(train.examples),
        eval_examples=len(evaluation.examples) if evaluation is not None else 0,
        warnings=tuple(warnings),
    )
