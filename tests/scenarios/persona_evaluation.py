"""Offline v0.12 example; no API key or network required.

Run: PYTHONPATH=src python tests/scenarios/persona_evaluation.py
"""
import asyncio
import json
from pathlib import Path

from ai_character_engine.evaluation import CharacterEvalDataset, evaluate_dataset


async def main() -> None:
    dataset = CharacterEvalDataset.from_jsonl(Path(__file__).resolve().parents[2] / "examples" / "data" / "character_eval.jsonl")
    report = await evaluate_dataset(dataset)
    print(json.dumps(report.to_dict()["metrics"], ensure_ascii=False, indent=2))
    for result in report.results:
        print(f"{result.case_id}: passed={result.passed}, score={result.consistency_score:.3f}")
        for failure in result.violations:
            print(f"  {failure.dimension.value}/{failure.constraint_id}: {failure.message}; evidence={failure.evidence}")
    # The fixture contains deliberate violations; detector label accuracy is
    # separate from character pass rate, which should therefore be low here.
    assert report.metrics.label_accuracy == 1.0
    assert report.metrics.passed_cases == 1
    assert report.metrics.failed_cases == 7


if __name__ == "__main__":
    asyncio.run(main())
