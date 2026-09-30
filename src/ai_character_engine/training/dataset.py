from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import DatasetIssue, DatasetStats, DatasetValidationReport, SFTExample


@dataclass(frozen=True, slots=True)
class CharacterSFTDataset:
    examples: tuple[SFTExample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "examples", tuple(self.examples))
        if not self.examples:
            raise ValueError("dataset cannot be empty")
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("example IDs must be unique")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CharacterSFTDataset":
        rows: list[SFTExample] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    rows.append(SFTExample.from_dict(payload))
                except Exception as exc:
                    raise ValueError(f"invalid SFT JSONL at line {line_number}: {exc}") from exc
        return cls(tuple(rows))

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            for example in self.examples:
                handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")

    def to_hf_rows(self) -> list[dict]:
        return [
            {
                "id": example.example_id,
                "messages": [message.to_dict() for message in example.messages],
                "metadata": dict(example.metadata),
            }
            for example in self.examples
        ]

    def validate(self) -> DatasetValidationReport:
        issues: list[DatasetIssue] = []
        message_count = 0
        users = 0
        assistants = 0
        assistant_chars = 0
        seen_normalized: dict[str, str] = {}

        for example in self.examples:
            message_count += len(example.messages)
            users += sum(m.role == "user" for m in example.messages)
            assistant_messages = [m for m in example.messages if m.role == "assistant"]
            assistants += len(assistant_messages)
            assistant_chars += sum(len(m.content) for m in assistant_messages)

            if len(example.messages) < 2:
                issues.append(DatasetIssue("error", "too_few_messages", "example needs at least two messages", example.example_id))
            if len(assistant_messages[-1].content.strip()) < 2:
                issues.append(DatasetIssue("error", "empty_target", "final assistant target is too short", example.example_id))
            if len(assistant_messages[-1].content) > 8_000:
                issues.append(DatasetIssue("warning", "long_target", "assistant target exceeds 8000 characters", example.example_id))

            normalized = "\n".join(f"{m.role}:{' '.join(m.content.lower().split())}" for m in example.messages)
            fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if fingerprint in seen_normalized:
                issues.append(DatasetIssue(
                    "warning", "duplicate_content",
                    f"content duplicates example {seen_normalized[fingerprint]}", example.example_id,
                ))
            else:
                seen_normalized[fingerprint] = example.example_id

        stats = DatasetStats(
            examples=len(self.examples),
            messages=message_count,
            user_messages=users,
            assistant_messages=assistants,
            assistant_chars=assistant_chars,
            average_messages_per_example=message_count / len(self.examples),
            average_assistant_chars=assistant_chars / assistants if assistants else 0.0,
        )
        return DatasetValidationReport(stats=stats, issues=tuple(issues))

    def deterministic_split(self, eval_ratio: float = 0.1, *, seed: int = 42) -> tuple["CharacterSFTDataset", "CharacterSFTDataset | None"]:
        if not 0 <= eval_ratio < 1:
            raise ValueError("eval_ratio must be in [0, 1)")
        if eval_ratio == 0 or len(self.examples) < 2:
            return self, None
        ranked = sorted(
            self.examples,
            key=lambda e: hashlib.sha256(f"{seed}:{e.example_id}".encode("utf-8")).digest(),
        )
        eval_count = max(1, round(len(ranked) * eval_ratio))
        eval_count = min(eval_count, len(ranked) - 1)
        eval_ids = {e.example_id for e in ranked[:eval_count]}
        train = tuple(e for e in self.examples if e.example_id not in eval_ids)
        evaluation = tuple(e for e in self.examples if e.example_id in eval_ids)
        return CharacterSFTDataset(train), CharacterSFTDataset(evaluation)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for example in sorted(self.examples, key=lambda e: e.example_id):
            digest.update(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def dataset_from_examples(examples: Iterable[SFTExample]) -> CharacterSFTDataset:
    return CharacterSFTDataset(tuple(examples))
