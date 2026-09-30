from pathlib import Path

from ai_character_engine.training import CharacterSFTDataset

path = Path(__file__).resolve().parents[2] / "examples" / "data" / "character_sft.jsonl"
dataset = CharacterSFTDataset.from_jsonl(path)
report = dataset.validate()
train, evaluation = dataset.deterministic_split(0.2, seed=42)

print("valid:", report.valid)
print("examples:", report.stats.examples)
print("fingerprint:", dataset.fingerprint()[:16])
print("train/eval:", len(train.examples), len(evaluation.examples) if evaluation else 0)
for issue in report.issues:
    print(issue.severity, issue.code, issue.example_id, issue.message)
