from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from .models import GoalRecord, GoalStatus, MotivationSignal
from .store import GoalStore, InMemoryGoalStore


@dataclass(frozen=True, slots=True)
class GoalCommitResult:
    goal: GoalRecord
    created: bool
    blocked_goal_ids: tuple[str, ...] = ()
    active_goal_ids: tuple[str, ...] = ()


class GoalManager:
    """Owns durable action intentions separately from Memory/Belief truth layers."""

    def __init__(self, *, store: GoalStore | None = None) -> None:
        self.store = store or InMemoryGoalStore()
        self.last_commit_result: GoalCommitResult | None = None

    def commit_candidate(self, record: GoalRecord) -> GoalCommitResult:
        before = self.store.list_goals(record.character_id)
        try:
            records = list(before)
            existing_index = next(
                (index for index, item in enumerate(records) if item.semantic_key == record.semantic_key),
                None,
            )
            created = existing_index is None
            if existing_index is None:
                candidate = record
                records.append(candidate)
            else:
                candidate = self._merge(records[existing_index], record)
                records[existing_index] = candidate

            records = self._reconcile_conflicts(records)
            self.store.replace_goals(record.character_id, records)
        except Exception:
            self.store.replace_goals(record.character_id, before)
            raise

        committed = next(item for item in records if item.semantic_key == record.semantic_key)
        result = GoalCommitResult(
            goal=committed,
            created=created,
            blocked_goal_ids=tuple(item.id for item in records if item.status is GoalStatus.BLOCKED),
            active_goal_ids=tuple(item.id for item in records if item.status is GoalStatus.ACTIVE),
        )
        self.last_commit_result = result
        return result

    def transition(
        self,
        *,
        character_id: str,
        goal_id: str,
        status: GoalStatus,
        reason: str,
    ) -> GoalRecord:
        if status is GoalStatus.BLOCKED:
            raise ValueError("blocked status is computed by conflict reconciliation")
        cleaned_reason = reason.strip()
        if not cleaned_reason:
            raise ValueError("goal transition reason must not be empty")
        before = self.store.list_goals(character_id)
        index = next((i for i, item in enumerate(before) if item.id == goal_id), None)
        if index is None:
            raise KeyError(goal_id)
        records = list(before)
        current = records[index]
        metadata = dict(current.metadata)
        transitions = list(metadata.get("transitions", ()))
        transitions.append(
            {
                "from": current.status.value,
                "to": status.value,
                "reason": cleaned_reason,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        metadata["transitions"] = transitions
        records[index] = replace(
            current,
            status=status,
            metadata=metadata,
            updated_at=datetime.now(UTC),
        )
        try:
            records = self._reconcile_conflicts(records)
            self.store.replace_goals(character_id, records)
        except Exception:
            self.store.replace_goals(character_id, before)
            raise
        return next(item for item in records if item.id == goal_id)

    def active_goals(self, *, character_id: str) -> tuple[GoalRecord, ...]:
        records = [item for item in self.store.list_goals(character_id) if item.status is GoalStatus.ACTIVE]
        records.sort(
            key=lambda item: (
                -item.rank_score,
                -item.support_count,
                item.horizon.value,
                item.objective.casefold(),
                item.id,
            )
        )
        return tuple(records)

    def goals(self, *, character_id: str) -> tuple[GoalRecord, ...]:
        return tuple(self.store.list_goals(character_id))

    @staticmethod
    def _merge(old: GoalRecord, new: GoalRecord) -> GoalRecord:
        signals: dict[tuple[str, tuple[str, str]], MotivationSignal] = {
            signal.key: signal for signal in old.motivation_signals
        }
        for signal in new.motivation_signals:
            previous = signals.get(signal.key)
            if previous is None or signal.strength > previous.strength:
                signals[signal.key] = signal
        metadata = dict(old.metadata)
        metadata.update(dict(new.metadata))
        return replace(
            old,
            urgency=max(old.urgency, new.urgency),
            confidence=max(old.confidence, new.confidence),
            motivation_signals=tuple(signals[key] for key in sorted(signals)),
            source_proposal_ids=tuple(dict.fromkeys((*old.source_proposal_ids, *new.source_proposal_ids))),
            source_task_ids=tuple(dict.fromkeys((*old.source_task_ids, *new.source_task_ids))),
            base_revisions=tuple(dict.fromkeys((*old.base_revisions, *new.base_revisions))),
            metadata=metadata,
            updated_at=datetime.now(UTC),
        )

    @staticmethod
    def _reconcile_conflicts(records: list[GoalRecord]) -> list[GoalRecord]:
        conflict_groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            if record.status in {GoalStatus.COMPLETED, GoalStatus.RETIRED, GoalStatus.PAUSED}:
                continue
            if record.conflict_key:
                conflict_groups.setdefault(record.conflict_key.casefold(), []).append(index)

        conflicting_indexes = {
            index
            for indexes in conflict_groups.values()
            if len({records[i].semantic_key for i in indexes}) > 1
            for index in indexes
        }
        now = datetime.now(UTC)
        reconciled: list[GoalRecord] = []
        for index, record in enumerate(records):
            if record.status in {GoalStatus.COMPLETED, GoalStatus.RETIRED, GoalStatus.PAUSED}:
                reconciled.append(record)
                continue
            desired = GoalStatus.BLOCKED if index in conflicting_indexes else GoalStatus.ACTIVE
            reconciled.append(
                record if record.status is desired else replace(record, status=desired, updated_at=now)
            )
        return reconciled
