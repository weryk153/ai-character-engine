from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .models import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalRecord,
    GoalStatus,
    MotivationKind,
    MotivationSignal,
)


class GoalStore(Protocol):
    def add_goal(self, record: GoalRecord) -> None: ...
    def list_goals(self, character_id: str) -> list[GoalRecord]: ...
    def replace_goals(self, character_id: str, records: Iterable[GoalRecord]) -> None: ...


class InMemoryGoalStore:
    def __init__(self, *, goals: Iterable[GoalRecord] = ()) -> None:
        self._goals = list(goals)

    def add_goal(self, record: GoalRecord) -> None:
        self._goals.append(record)

    def list_goals(self, character_id: str) -> list[GoalRecord]:
        return [record for record in self._goals if record.character_id == character_id]

    def replace_goals(self, character_id: str, records: Iterable[GoalRecord]) -> None:
        kept = [record for record in self._goals if record.character_id != character_id]
        self._goals = [*kept, *list(records)]


class JsonlGoalStore(InMemoryGoalStore):
    """Single-process local/dev JSONL persistence for goals."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        goals: list[GoalRecord] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("record_type") != "goal":
                    raise ValueError(f"unknown goal JSONL record_type: {payload.get('record_type')!r}")
                payload = dict(migrate_persistence_payload(PersistenceSurface.GOAL, payload).payload)
                goals.append(_goal_from_dict(payload))
        super().__init__(goals=goals)

    def add_goal(self, record: GoalRecord) -> None:
        super().add_goal(record)
        self._append(_goal_to_dict(record))

    def replace_goals(self, character_id: str, records: Iterable[GoalRecord]) -> None:
        super().replace_goals(character_id, records)
        self._rewrite()

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(_goal_to_dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in self._goals
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)


def _evidence_to_dict(record: GoalEvidenceRef) -> dict[str, Any]:
    return {
        "source_type": record.source_type,
        "source_id": record.source_id,
        "excerpt": record.excerpt,
        "metadata": _jsonable(record.metadata),
    }


def _evidence_from_dict(payload: dict[str, Any]) -> GoalEvidenceRef:
    return GoalEvidenceRef(
        source_type=payload["source_type"],
        source_id=payload["source_id"],
        excerpt=payload["excerpt"],
        metadata=dict(payload.get("metadata", {})),
    )


def _signal_to_dict(signal: MotivationSignal) -> dict[str, Any]:
    return {
        "kind": signal.kind.value,
        "strength": signal.strength,
        "evidence": _evidence_to_dict(signal.evidence),
        "rationale": signal.rationale,
    }


def _signal_from_dict(payload: dict[str, Any]) -> MotivationSignal:
    return MotivationSignal(
        kind=MotivationKind(payload["kind"]),
        strength=float(payload["strength"]),
        evidence=_evidence_from_dict(payload["evidence"]),
        rationale=payload["rationale"],
    )


def _goal_to_dict(record: GoalRecord) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.GOAL, {
        "record_type": "goal",
        "id": record.id,
        "character_id": record.character_id,
        "objective": record.objective,
        "horizon": record.horizon.value,
        "urgency": record.urgency,
        "confidence": record.confidence,
        "motivation_signals": [_signal_to_dict(item) for item in record.motivation_signals],
        "status": record.status.value,
        "conflict_key": record.conflict_key,
        "source_proposal_ids": list(record.source_proposal_ids),
        "source_task_ids": list(record.source_task_ids),
        "base_revisions": list(record.base_revisions),
        "metadata": _jsonable(record.metadata),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    })


def _goal_from_dict(payload: dict[str, Any]) -> GoalRecord:
    return GoalRecord(
        id=payload["id"],
        character_id=payload["character_id"],
        objective=payload["objective"],
        horizon=GoalHorizon(payload["horizon"]),
        urgency=float(payload["urgency"]),
        confidence=float(payload["confidence"]),
        motivation_signals=tuple(_signal_from_dict(item) for item in payload.get("motivation_signals", [])),
        status=GoalStatus(payload.get("status", GoalStatus.ACTIVE.value)),
        conflict_key=payload.get("conflict_key"),
        source_proposal_ids=tuple(payload.get("source_proposal_ids", ())),
        source_task_ids=tuple(payload.get("source_task_ids", ())),
        base_revisions=tuple(int(value) for value in payload.get("base_revisions", ())),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
