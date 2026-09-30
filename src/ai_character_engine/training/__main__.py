from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import CharacterSFTDataset
from .huggingface import HuggingFaceSFTBackend
from .models import LoRAConfigSpec, QuantizationConfigSpec, SFTTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="ai-character-engine SFT/LoRA dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a conversational JSONL dataset")
    validate.add_argument("dataset")

    plan = sub.add_parser("plan", help="compile a provider-neutral SFT/LoRA training plan")
    plan.add_argument("dataset")
    plan.add_argument("--base-model", required=True)
    plan.add_argument("--output-dir", default="./outputs/character-lora")
    plan.add_argument("--eval-ratio", type=float, default=0.1)
    plan.add_argument("--max-length", type=int, default=2048)
    plan.add_argument("--lora-r", type=int, default=16)
    plan.add_argument("--lora-alpha", type=int, default=32)
    plan.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    plan.add_argument("--output")

    args = parser.parse_args()
    dataset = CharacterSFTDataset.from_jsonl(args.dataset)
    if args.command == "validate":
        report = dataset.validate()
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.valid else 2

    config = SFTTrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        max_length=args.max_length,
        lora=LoRAConfigSpec(r=args.lora_r, alpha=args.lora_alpha),
        quantization=QuantizationConfigSpec(mode=args.quantization),
    )
    compiled = HuggingFaceSFTBackend().compile(dataset, config)
    text = json.dumps(compiled, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
