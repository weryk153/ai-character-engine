from __future__ import annotations

from dataclasses import dataclass

from .consolidation import (
    BeliefConsolidationPolicy,
    BeliefConsolidationResult,
    DeterministicBeliefConsolidator,
)
from .models import BeliefRecord, BeliefStatus, ReflectionRecord
from .store import InMemoryLongTermCognitionStore, LongTermCognitionStore


@dataclass(frozen=True, slots=True)
class ReflectionCommitResult:
    reflection: ReflectionRecord
    consolidation: BeliefConsolidationResult


class LongTermCognitionManager:
    """Owns durable reflection/belief data independently of MemoryManager."""

    def __init__(
        self,
        *,
        store: LongTermCognitionStore | None = None,
        consolidation_policy: BeliefConsolidationPolicy | None = None,
    ) -> None:
        self.store = store or InMemoryLongTermCognitionStore()
        self.consolidator = DeterministicBeliefConsolidator(
            self.store, policy=consolidation_policy
        )
        self.last_consolidation_result: BeliefConsolidationResult | None = None

    def commit_reflection(self, record: ReflectionRecord) -> ReflectionCommitResult:
        """Persist one provisional reflection and consolidate atomically per scope.

        JSONL/in-memory stores expose replace operations, so a failed append or
        consolidation can restore both cognition collections to their pre-commit
        state. External production stores should provide equivalent transactional
        semantics behind the same protocol.
        """

        before_reflections = self.store.list_reflections(record.character_id)
        before_beliefs = self.store.list_beliefs(record.character_id)
        try:
            self.store.add_reflection(record)
            consolidation = self.consolidator.consolidate(character_id=record.character_id)
        except Exception:
            self.store.replace_reflections(record.character_id, before_reflections)
            self.store.replace_beliefs(record.character_id, before_beliefs)
            raise
        self.last_consolidation_result = consolidation
        return ReflectionCommitResult(reflection=record, consolidation=consolidation)

    def consolidate(self, *, character_id: str) -> BeliefConsolidationResult:
        result = self.consolidator.consolidate(character_id=character_id)
        self.last_consolidation_result = result
        return result

    def active_beliefs(self, *, character_id: str) -> tuple[BeliefRecord, ...]:
        beliefs = [
            record
            for record in self.store.list_beliefs(character_id)
            if record.status is BeliefStatus.ACTIVE
        ]
        beliefs.sort(
            key=lambda record: (
                -record.confidence,
                -record.support_count,
                record.claim.key,
                record.claim.object.casefold(),
                record.id,
            )
        )
        return tuple(beliefs)

    def reflections(self, *, character_id: str) -> tuple[ReflectionRecord, ...]:
        return tuple(self.store.list_reflections(character_id))

    def beliefs(self, *, character_id: str) -> tuple[BeliefRecord, ...]:
        return tuple(self.store.list_beliefs(character_id))
