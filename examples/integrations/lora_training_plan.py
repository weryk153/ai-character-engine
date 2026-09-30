from pathlib import Path

from ai_character_engine.training import (
    CharacterSFTDataset,
    HuggingFaceSFTBackend,
    LoRAConfigSpec,
    QuantizationConfigSpec,
    SFTTrainingConfig,
)

path = Path(__file__).resolve().parents[2] / "examples" / "data" / "character_sft.jsonl"
dataset = CharacterSFTDataset.from_jsonl(path)
config = SFTTrainingConfig(
    base_model="Qwen/Qwen3-0.6B",
    output_dir="./outputs/mei-lora",
    max_length=2048,
    eval_ratio=0.2,
    lora=LoRAConfigSpec(r=16, alpha=32, dropout=0.05),
    quantization=QuantizationConfigSpec(mode="none"),
)
compiled = HuggingFaceSFTBackend().compile(dataset, config)
print("backend:", compiled["backend"])
print("method:", compiled["method"])
print("train examples:", compiled["plan"]["train_examples"])
print("eval examples:", compiled["plan"]["eval_examples"])
for warning in compiled["plan"]["warnings"]:
    print("warning:", warning)
