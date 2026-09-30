from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataset import CharacterSFTDataset
from .models import AdapterManifest, SFTTrainingConfig
from .planner import build_training_plan


class TrainingDependencyError(RuntimeError):
    pass


class HuggingFaceSFTBackend:
    """Optional TRL/PEFT backend.

    Imports are deliberately lazy so the character runtime never requires a
    training/GPU stack. Tests can exercise plan compilation without torch.
    """

    def compile(self, dataset: CharacterSFTDataset, config: SFTTrainingConfig) -> dict[str, Any]:
        plan = build_training_plan(dataset, config)
        method = "qlora" if config.quantization.mode != "none" else "lora"
        return {
            "schema_version": "0.21",
            "backend": "huggingface-trl-peft",
            "method": method,
            "dataset_fingerprint": dataset.fingerprint(),
            "plan": plan.to_dict(),
            "hf_dataset_columns": ["id", "messages", "metadata"],
        }

    def save_plan(self, dataset: CharacterSFTDataset, config: SFTTrainingConfig, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.compile(dataset, config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def train(self, dataset: CharacterSFTDataset, config: SFTTrainingConfig) -> AdapterManifest:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import BitsAndBytesConfig
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise TrainingDependencyError(
                "training dependencies are missing; install the optional 'training' extra in a supported GPU environment"
            ) from exc

        report = dataset.validate()
        if not report.valid:
            raise ValueError("dataset contains validation errors")
        train_dataset, eval_dataset = dataset.deterministic_split(config.eval_ratio, seed=config.seed)

        model_init_kwargs: dict[str, Any] = {}
        if config.quantization.mode != "none":
            dtype = getattr(torch, config.quantization.compute_dtype, None)
            if dtype is None:
                raise ValueError(f"unsupported torch dtype: {config.quantization.compute_dtype}")
            model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=config.quantization.mode == "4bit",
                load_in_8bit=config.quantization.mode == "8bit",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=config.quantization.quant_type,
                bnb_4bit_use_double_quant=config.quantization.double_quant,
            )

        lora_kwargs: dict[str, Any] = {
            "r": config.lora.r,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "bias": config.lora.bias,
            "task_type": "CAUSAL_LM",
            "use_rslora": config.lora.use_rslora,
        }
        if config.lora.target_modules:
            lora_kwargs["target_modules"] = list(config.lora.target_modules)
        if config.lora.modules_to_save:
            lora_kwargs["modules_to_save"] = list(config.lora.modules_to_save)
        peft_config = LoraConfig(**lora_kwargs)

        args = SFTConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            packing=config.packing,
            assistant_only_loss=config.assistant_only_loss,
            seed=config.seed,
            model_init_kwargs=model_init_kwargs or None,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=config.base_model,
            args=args,
            train_dataset=Dataset.from_list(train_dataset.to_hf_rows()),
            eval_dataset=Dataset.from_list(eval_dataset.to_hf_rows()) if eval_dataset else None,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(config.output_dir)

        manifest = AdapterManifest(
            adapter_path=config.output_dir,
            base_model=config.base_model,
            method="qlora" if config.quantization.mode != "none" else "lora",
            dataset_fingerprint=dataset.fingerprint(),
            examples=len(dataset.examples),
            metadata={"config": asdict(config)},
        )
        manifest.save(Path(config.output_dir) / "ai_character_engine_adapter.json")
        return manifest
