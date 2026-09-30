from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from .models import (
    SAFE_BELIEF_EVIDENCE_TYPES,
    BeliefClaim,
    BeliefRecord,
    BeliefStatus,
    CognitionEvidenceRef,
    ReflectionRecord,
    ReflectionStatus,
)
from .store import LongTermCognitionStore


@dataclass(frozen=True, slots=True)
class BeliefConsolidationPolicy:
    min_reflection_confidence: float = 0.80
    min_belief_confidence: float = 0.78
    min_independent_sources: int = 2
    safe_evidence_types: frozenset[str] = field(
        default_factory=lambda: SAFE_BELIEF_EVIDENCE_TYPES
    )

    def __post_init__(self) -> None:
        if not 0 <= self.min_reflection_confidence <= 1:
            raise ValueError("min_reflection_confidence must be between 0 and 1")
        if not 0 <= self.min_belief_confidence <= 1:
            raise ValueError("min_belief_confidence must be between 0 and 1")
        if self.min_independent_sources < 1:
            raise ValueError("min_independent_sources must be >= 1")
        cleaned = frozenset(str(item).strip().casefold() for item in self.safe_evidence_types if str(item).strip())
        if not cleaned:
            raise ValueError("safe_evidence_types must not be empty")
        object.__setattr__(self, "safe_evidence_types", cleaned)


@dataclass(frozen=True, slots=True)
class BeliefConsolidationResult:
    created_belief_ids: tuple[str, ...] = ()
    updated_belief_ids: tuple[str, ...] = ()
    contested_belief_ids: tuple[str, ...] = ()
    active_belief_ids: tuple[str, ...] = ()
    qualifying_claim_values: int = 0


@dataclass(slots=True)
class _Candidate:
    claim: BeliefClaim
    reflection_ids: set[str]
    evidence_by_source: dict[tuple[str, str], CognitionEvidenceRef]
    confidence_by_source: dict[tuple[str, str], float]

    @property
    def support_count(self) -> int:
        return len(self.evidence_by_source)

    @property
    def confidence(self) -> float:
        if not self.confidence_by_source:
            return 0.0
        return max(
            0.0,
            min(1.0, sum(self.confidence_by_source.values()) / len(self.confidence_by_source)),
        )


class DeterministicBeliefConsolidator:
    """Promotes safe repeated reflection evidence without asking another LLM."""

    def __init__(
        self,
        store: LongTermCognitionStore,
        *,
        policy: BeliefConsolidationPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or BeliefConsolidationPolicy()

    def consolidate(self, *, character_id: str) -> BeliefConsolidationResult:
        existing = self.store.list_beliefs(character_id)
        candidates = self._candidates(self.store.list_reflections(character_id))
        qualifying = {
            key: candidate
            for key, candidate in candidates.items()
            if candidate.support_count >= self.policy.min_independent_sources
            and candidate.confidence >= self.policy.min_belief_confidence
        }
        if not qualifying:
            return BeliefConsolidationResult(
                active_belief_ids=tuple(record.id for record in existing if record.is_active),
                qualifying_claim_values=0,
            )

        by_exact = {record.claim.value_key: record for record in existing}
        by_key: dict[tuple[str, str], list[tuple[tuple[str, str, str], _Candidate]]] = {}
        for exact_key, candidate in qualifying.items():
            by_key.setdefault(candidate.claim.key, []).append((exact_key, candidate))

        now = datetime.now(UTC)
        replacements: dict[tuple[str, str, str], BeliefRecord] = {}
        created: list[str] = []
        updated: list[str] = []
        contested: list[str] = []

        for claim_key, values in by_key.items():
            is_contested = len(values) > 1
            status = BeliefStatus.CONTESTED if is_contested else BeliefStatus.ACTIVE
            for exact_key, candidate in values:
                old = by_exact.get(exact_key)
                evidence = tuple(
                    candidate.evidence_by_source[key]
                    for key in sorted(candidate.evidence_by_source)
                )
                reflection_ids = tuple(sorted(candidate.reflection_ids))
                metadata = {
                    "consolidation": "deterministic",
                    "independence_rule": "source_type+source_id",
                }
                if old is None:
                    record = BeliefRecord(
                        character_id=character_id,
                        claim=candidate.claim,
                        confidence=candidate.confidence,
                        evidence=evidence,
                        source_reflection_ids=reflection_ids,
                        support_count=candidate.support_count,
                        status=status,
                        metadata=metadata,
                        created_at=now,
                        updated_at=now,
                    )
                    created.append(record.id)
                else:
                    record = replace(
                        old,
                        confidence=candidate.confidence,
                        evidence=evidence,
                        source_reflection_ids=reflection_ids,
                        support_count=candidate.support_count,
                        status=status,
                        metadata=metadata,
                        updated_at=now,
                    )
                    if record != old:
                        updated.append(record.id)
                replacements[exact_key] = record
                if status is BeliefStatus.CONTESTED:
                    contested.append(record.id)

            # If multiple values qualify, any existing belief with the same
            # (subject,predicate) key remains visibly contested rather than being
            # silently superseded by a newer value.
            if is_contested:
                for old in existing:
                    if old.claim.key != claim_key or old.claim.value_key in replacements:
                        continue
                    record = replace(old, status=BeliefStatus.CONTESTED, updated_at=now)
                    replacements[old.claim.value_key] = record
                    if record != old:
                        updated.append(record.id)
                    contested.append(record.id)

        merged: list[BeliefRecord] = []
        consumed: set[tuple[str, str, str]] = set()
        for old in existing:
            replacement = replacements.get(old.claim.value_key)
            if replacement is not None:
                merged.append(replacement)
                consumed.add(old.claim.value_key)
            else:
                merged.append(old)
        for key, record in replacements.items():
            if key not in consumed:
                merged.append(record)

        self.store.replace_beliefs(character_id, merged)
        return BeliefConsolidationResult(
            created_belief_ids=tuple(dict.fromkeys(created)),
            updated_belief_ids=tuple(dict.fromkeys(updated)),
            contested_belief_ids=tuple(dict.fromkeys(contested)),
            active_belief_ids=tuple(record.id for record in merged if record.is_active),
            qualifying_claim_values=len(qualifying),
        )

    def _candidates(
        self, reflections: list[ReflectionRecord]
    ) -> dict[tuple[str, str, str], _Candidate]:
        candidates: dict[tuple[str, str, str], _Candidate] = {}
        for reflection in reflections:
            if reflection.status is not ReflectionStatus.PROVISIONAL:
                continue
            if reflection.claim is None:
                continue
            if reflection.confidence < self.policy.min_reflection_confidence:
                continue
            safe_evidence = [
                evidence
                for evidence in reflection.evidence
                if evidence.evidence_type.casefold() in self.policy.safe_evidence_types
            ]
            if not safe_evidence:
                continue
            key = reflection.claim.value_key
            candidate = candidates.setdefault(
                key,
                _Candidate(
                    claim=reflection.claim,
                    reflection_ids=set(),
                    evidence_by_source={},
                    confidence_by_source={},
                ),
            )
            candidate.reflection_ids.add(reflection.id)
            for evidence in safe_evidence:
                source_key = evidence.independence_key
                evidence_confidence = (
                    reflection.confidence
                    if evidence.confidence is None
                    else min(reflection.confidence, evidence.confidence)
                )
                previous = candidate.confidence_by_source.get(source_key)
                if previous is None or evidence_confidence > previous:
                    candidate.evidence_by_source[source_key] = evidence
                    candidate.confidence_by_source[source_key] = evidence_confidence
        return candidates
