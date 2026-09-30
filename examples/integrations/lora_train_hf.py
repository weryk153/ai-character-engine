"""Optional real training entry point. Requires `pip install -e '.[training]'`.

This example intentionally does not auto-download or run during regression tests.
Choose a base model and hardware you are licensed and equipped to train.
"""
from pathlib import Path

from ai_character_engine.training import (
    CharacterSFTDataset,
    HuggingFaceSFTBackend,
    LoRAConfigSpec,
    SFTTrainingConfig,
)

path = Path(__file__).resolve().parents[2] / "examples" / "data" / "character_sft.jsonl"
dataset = CharacterSFTDataset.from_jsonl(path)
config = SFTTrainingConfig(
    base_model="Qwen/Qwen3-0.6B",
    output_dir="./outputs/mei-lora",
    lora=LoRAConfigSpec(r=16, alpha=32),
)
manifest = HuggingFaceSFTBackend().train(dataset, config)
print(manifest.to_dict())
