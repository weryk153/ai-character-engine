from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema

from .models import (
    BeliefClaim,
    BeliefRecord,
    BeliefStatus,
    CognitionEvidenceRef,
    ReflectionRecord,
    ReflectionStatus,
)


class LongTermCognitionStore(Protocol):
    def add_reflection(self, record: ReflectionRecord) -> None: ...

    def list_reflections(self, character_id: str) -> list[ReflectionRecord]: ...

    def replace_reflections(
        self, character_id: str, records: Iterable[ReflectionRecord]
    ) -> None: ...

    def add_belief(self, record: BeliefRecord) -> None: ...

    def list_beliefs(self, character_id: str) -> list[BeliefRecord]: ...

    def replace_beliefs(self, character_id: str, records: Iterable[BeliefRecord]) -> None: ...


class InMemoryLongTermCognitionStore:
    def __init__(
        self,
        *,
        reflections: Iterable[ReflectionRecord] = (),
        beliefs: Iterable[BeliefRecord] = (),
    ) -> None:
        self._reflections = list(reflections)
        self._beliefs = list(beliefs)

    def add_reflection(self, record: ReflectionRecord) -> None:
        self._reflections.append(record)

    def list_reflections(self, character_id: str) -> list[ReflectionRecord]:
        return [record for record in self._reflections if record.character_id == character_id]

    def replace_reflections(
        self, character_id: str, records: Iterable[ReflectionRecord]
    ) -> None:
        kept = [record for record in self._reflections if record.character_id != character_id]
        self._reflections = [*kept, *list(records)]

    def add_belief(self, record: BeliefRecord) -> None:
        self._beliefs.append(record)

    def list_beliefs(self, character_id: str) -> list[BeliefRecord]:
        return [record for record in self._beliefs if record.character_id == character_id]

    def replace_beliefs(self, character_id: str, records: Iterable[BeliefRecord]) -> None:
        kept = [record for record in self._beliefs if record.character_id != character_id]
        self._beliefs = [*kept, *list(records)]


class JsonlLongTermCognitionStore(InMemoryLongTermCognitionStore):
    """Small local/dev persistence store for reflections and beliefs.

    This intentionally remains a single-process JSONL implementation. Production
    databases/distributed transactions stay outside the v0.34 core contract.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        reflections: list[ReflectionRecord] = []
        beliefs: list[BeliefRecord] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                record_type = payload.get("record_type")
                if record_type == "reflection":
                    payload = dict(migrate_persistence_payload(PersistenceSurface.REFLECTION, payload).payload)
                    reflections.append(_reflection_from_dict(payload))
                elif record_type == "belief":
                    payload = dict(migrate_persistence_payload(PersistenceSurface.BELIEF, payload).payload)
                    beliefs.append(_belief_from_dict(payload))
                else:
                    raise ValueError(f"unknown cognition JSONL record_type: {record_type!r}")
        super().__init__(reflections=reflections, beliefs=beliefs)

    def add_reflection(self, record: ReflectionRecord) -> None:
        super().add_reflection(record)
        self._append(_reflection_to_dict(record))

    def replace_reflections(
        self, character_id: str, records: Iterable[ReflectionRecord]
    ) -> None:
        super().replace_reflections(character_id, records)
        self._rewrite()

    def add_belief(self, record: BeliefRecord) -> None:
        super().add_belief(record)
        self._append(_belief_to_dict(record))

    def replace_beliefs(self, character_id: str, records: Iterable[BeliefRecord]) -> None:
        super().replace_beliefs(character_id, records)
        self._rewrite()

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            *(_reflection_to_dict(record) for record in self._reflections),
            *(_belief_to_dict(record) for record in self._beliefs),
        ]
        payload = "".join(
            json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n" for line in lines
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)


def _evidence_to_dict(record: CognitionEvidenceRef) -> dict[str, Any]:
    return {
        "source_type": record.source_type,
        "source_id": record.source_id,
        "evidence_type": record.evidence_type,
        "excerpt": record.excerpt,
        "confidence": record.confidence,
        "metadata": _jsonable(record.metadata),
    }


def _evidence_from_dict(payload: dict[str, Any]) -> CognitionEvidenceRef:
    return CognitionEvidenceRef(
        source_type=payload["source_type"],
        source_id=payload["source_id"],
        evidence_type=payload["evidence_type"],
        excerpt=payload["excerpt"],
        confidence=payload.get("confidence"),
        metadata=dict(payload.get("metadata", {})),
    )


def _claim_to_dict(claim: BeliefClaim | None) -> dict[str, str] | None:
    if claim is None:
        return None
    return {"subject": claim.subject, "predicate": claim.predicate, "object": claim.object}


def _claim_from_dict(payload: dict[str, Any] | None) -> BeliefClaim | None:
    if payload is None:
        return None
    return BeliefClaim(
        subject=payload["subject"],
        predicate=payload["predicate"],
        object=payload["object"],
    )


def _reflection_to_dict(record: ReflectionRecord) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.REFLECTION, {
        "record_type": "reflection",
        "id": record.id,
        "character_id": record.character_id,
        "insight": record.insight,
        "confidence": record.confidence,
        "evidence": [_evidence_to_dict(item) for item in record.evidence],
        "claim": _claim_to_dict(record.claim),
        "base_revision": record.base_revision,
        "source_proposal_id": record.source_proposal_id,
        "source_task_id": record.source_task_id,
        "status": record.status.value,
        "metadata": _jsonable(record.metadata),
        "created_at": record.created_at.isoformat(),
    })


def _reflection_from_dict(payload: dict[str, Any]) -> ReflectionRecord:
    return ReflectionRecord(
        id=payload["id"],
        character_id=payload["character_id"],
        insight=payload["insight"],
        confidence=payload["confidence"],
        evidence=tuple(_evidence_from_dict(item) for item in payload.get("evidence", [])),
        claim=_claim_from_dict(payload.get("claim")),
        base_revision=payload.get("base_revision", 0),
        source_proposal_id=payload.get("source_proposal_id"),
        source_task_id=payload.get("source_task_id"),
        status=ReflectionStatus(payload.get("status", ReflectionStatus.PROVISIONAL.value)),
        metadata=dict(payload.get("metadata", {})),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


def _belief_to_dict(record: BeliefRecord) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.BELIEF, {
        "record_type": "belief",
        "id": record.id,
        "character_id": record.character_id,
        "claim": _claim_to_dict(record.claim),
        "confidence": record.confidence,
        "evidence": [_evidence_to_dict(item) for item in record.evidence],
        "source_reflection_ids": list(record.source_reflection_ids),
        "support_count": record.support_count,
        "status": record.status.value,
        "metadata": _jsonable(record.metadata),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    })


def _belief_from_dict(payload: dict[str, Any]) -> BeliefRecord:
    claim = _claim_from_dict(payload.get("claim"))
    if claim is None:
        raise ValueError("belief record requires claim")
    return BeliefRecord(
        id=payload["id"],
        character_id=payload["character_id"],
        claim=claim,
        confidence=payload["confidence"],
        evidence=tuple(_evidence_from_dict(item) for item in payload.get("evidence", [])),
        source_reflection_ids=tuple(payload.get("source_reflection_ids", ())),
        support_count=int(payload.get("support_count", 0)),
        status=BeliefStatus(payload.get("status", BeliefStatus.ACTIVE.value)),
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
