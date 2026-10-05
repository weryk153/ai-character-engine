from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime
from typing import Callable, Iterable
from uuid import uuid4

from ai_character_engine.memory.models import MemoryRecord
from ai_character_engine.long_term_cognition import (
    BeliefClaim,
    CognitionEvidenceRef,
    ReflectionRecord,
)
from ai_character_engine.goals import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalRecord,
    MotivationKind,
    MotivationSignal,
)
from ai_character_engine.state.models import StatePatch
from ai_character_engine.state.mood import (
    CHARACTER_MOODS,
    MOOD_TURN_KEY,
    blend_mood,
    effective_mood,
    seconds,
)
from ai_character_engine.goals.models import MOTIVATION_SOURCE_TYPES
from ai_character_engine.tasks.models import TaskProposal, TaskResult, TaskStatus
from ai_character_engine.tasks.runtime import MultiTaskRuntime

from .models import (
    CommitLifecycleEvent,
    CommitNextAction,
    CommitResult,
    CommitStatus,
    CommitTargetPolicy,
    StalePolicy,
)


_DEFAULT_POLICIES: dict[str, CommitTargetPolicy] = {
    "memory.append_candidate": CommitTargetPolicy(
        min_confidence=0.75,
        stale_policy=StalePolicy.ALLOW_MANUAL_REBASE,
        expected_worker_kind="memory_extraction",
        max_age_s=300.0,
    ),
    # What the character said about herself. Written into the memory scope
    # that is current when it is committed: a host that keeps these apart
    # from memory of the user (CharacterCompanion does) sets the scope for the
    # commit. Like memory of the user it stays true however late it arrives.
    "memory.self_candidate": CommitTargetPolicy(
        min_confidence=0.75,
        stale_policy=StalePolicy.ALLOW_MANUAL_REBASE,
        expected_worker_kind="self_memory_extraction",
        max_age_s=300.0,
    ),
    "state.emotion_candidate": CommitTargetPolicy(
        min_confidence=0.70,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="emotion_analysis",
        max_age_s=90.0,
    ),
    # The character's own mood, read from both sides of the conversation.
    "state.mood_candidate": CommitTargetPolicy(
        min_confidence=0.70,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="character_mood",
        max_age_s=90.0,
    ),
    "memory.conversation_summary_candidate": CommitTargetPolicy(
        min_confidence=0.70,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="conversation_summary",
        max_age_s=300.0,
    ),
    # Reflection remains review-only unless a LongTermCognitionManager is
    # explicitly configured. Vision interpretation remains review-only. Neither
    # model output is silently promoted into durable fact or Character State.
    "cognition.reflection_candidate": CommitTargetPolicy(
        min_confidence=0.80,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="reflection",
        max_age_s=300.0,
    ),
    "cognition.goal_candidate": CommitTargetPolicy(
        min_confidence=0.78,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="goal_motivation",
        max_age_s=180.0,
    ),
    "context.vision_interpretation_candidate": CommitTargetPolicy(
        min_confidence=0.80,
        stale_policy=StalePolicy.RERUN,
        expected_worker_kind="vision_interpretation",
        max_age_s=120.0,
    ),
}


class CognitiveCommitCoordinator:
    """Validates and serializes background proposals into authoritative writes.

    Background workers remain non-authoritative.  This coordinator is the bridge
    from ``TaskProposal`` to authoritative State/Memory and, when explicitly
    configured, the separate long-term cognition store.  It shares
    the MultiTaskRuntime authority lock with foreground turns so freshness is
    checked in the same critical section as the mutation.
    """

    def __init__(
        self,
        tasks: MultiTaskRuntime,
        *,
        policies: dict[str, CommitTargetPolicy] | None = None,
        event_history: int = 512,
    ) -> None:
        if event_history < 1:
            raise ValueError("event_history must be >= 1")
        self.tasks = tasks
        self.policies = dict(_DEFAULT_POLICIES)
        if policies:
            self.policies.update(policies)
        self._kept = event_history
        self._events: deque[CommitLifecycleEvent] = deque(maxlen=event_history)
        self._processed: dict[str, CommitResult] = {}
        self._semantic_commits: dict[str, str] = {}
        self._conflicts: dict[str, tuple[str, str]] = {}
        self._commit_sequence = 0
        self._attempts: dict[str, int] = {}
        # Dates her mood when it is written. A host with its own clock
        # (CharacterCompanion) sets it, so that her mood fades by that clock.
        self.clock: Callable[[], float] = time.time
        # How her mood fades before a new reading is weighed against it
        # (CompanionSettings.mood_half_life_seconds/mood_floor).
        self.mood_half_life_seconds = 300.0
        self.mood_floor = 0.15

    @property
    def commit_sequence(self) -> int:
        return self._commit_sequence

    def events(self) -> tuple[CommitLifecycleEvent, ...]:
        return tuple(self._events)

    @property
    def remembered_decisions(self) -> int:
        """How many proposals a second attempt is still answered for."""

        return len(self._processed)

    def _let_go_of_the_oldest(self) -> None:
        # The newest event_history of each, like the events. A proposal is
        # tried again within moments, and the keys that guard against
        # duplicates and conflicts name a turn that is long past by then.
        for kept in (self._processed, self._attempts, self._semantic_commits, self._conflicts):
            while len(kept) > self._kept:
                del kept[next(iter(kept))]

    def result_for(self, proposal_id: str) -> CommitResult | None:
        return self._processed.get(proposal_id)

    async def commit(self, proposal: TaskProposal, *, retry: bool = False) -> CommitResult:
        previous = self._processed.get(proposal.id)
        if previous is not None and previous.status is not CommitStatus.RETRYABLE_ERROR:
            return previous
        if previous is not None and not retry:
            return previous

        async with self.tasks.authority_guard():
            # Re-check after waiting for an in-flight foreground turn.
            previous = self._processed.get(proposal.id)
            if previous is not None and previous.status is not CommitStatus.RETRYABLE_ERROR:
                return previous

            current_revision = self.tasks.revision
            policy = self.policies.get(proposal.target)
            if policy is None:
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "unsupported_target",
                    current_revision,
                )

            validation = self._validate(proposal, policy, current_revision)
            if validation is not None:
                return validation

            existing_authority = self._existing_authority_decision(proposal, current_revision)
            if existing_authority is not None:
                return existing_authority

            # Reflection becomes persistable only when the host explicitly
            # configures the separate LongTermCognitionManager.  Without it,
            # reflection stays review-only. Vision remains review-only.
            if proposal.target == "cognition.reflection_candidate":
                if getattr(self.tasks.runtime, "long_term_cognition", None) is None:
                    return self._finalize(
                        proposal,
                        CommitStatus.REVIEW_REQUIRED,
                        "long_term_cognition_not_configured",
                        current_revision,
                        next_action=CommitNextAction.MANUAL_REVIEW,
                    )
            if proposal.target == "cognition.goal_candidate":
                if getattr(self.tasks.runtime, "goal_manager", None) is None:
                    return self._finalize(
                        proposal,
                        CommitStatus.REVIEW_REQUIRED,
                        "goal_manager_not_configured",
                        current_revision,
                        next_action=CommitNextAction.MANUAL_REVIEW,
                    )
            if proposal.target == "context.vision_interpretation_candidate":
                return self._finalize(
                    proposal,
                    CommitStatus.REVIEW_REQUIRED,
                    "interpretation_target_not_authoritatively_committed",
                    current_revision,
                    next_action=CommitNextAction.MANUAL_REVIEW,
                )

            semantic_key, conflict_key, fingerprint = self._identity(proposal)
            if semantic_key is not None and semantic_key in self._semantic_commits:
                return self._finalize(
                    proposal,
                    CommitStatus.DUPLICATE,
                    "semantic_duplicate",
                    current_revision,
                    metadata={"original_proposal_id": self._semantic_commits[semantic_key]},
                )
            if conflict_key is not None:
                existing = self._conflicts.get(conflict_key)
                if existing is not None:
                    existing_id, existing_fingerprint = existing
                    if existing_fingerprint == fingerprint:
                        return self._finalize(
                            proposal,
                            CommitStatus.DUPLICATE,
                            "conflict_key_duplicate",
                            current_revision,
                            metadata={"original_proposal_id": existing_id},
                        )
                    return self._finalize(
                        proposal,
                        CommitStatus.CONFLICT,
                        "conflicting_proposal_already_committed",
                        current_revision,
                        next_action=CommitNextAction.MANUAL_REVIEW,
                        metadata={"conflicting_proposal_id": existing_id},
                    )

            self._attempts[proposal.id] = self._attempts.get(proposal.id, 0) + 1
            try:
                applied_record_id = self._apply(proposal)
            except Exception as exc:
                return self._finalize(
                    proposal,
                    CommitStatus.RETRYABLE_ERROR,
                    f"{type(exc).__name__}: {exc}",
                    current_revision,
                    next_action=CommitNextAction.RETRY,
                    metadata={"attempt": self._attempts[proposal.id]},
                )

            self._commit_sequence += 1
            if semantic_key is not None:
                self._semantic_commits[semantic_key] = proposal.id
            if conflict_key is not None:
                self._conflicts[conflict_key] = (proposal.id, fingerprint)
            return self._finalize(
                proposal,
                CommitStatus.COMMITTED,
                "committed",
                current_revision,
                commit_sequence=self._commit_sequence,
                applied_record_id=applied_record_id,
                metadata={"attempt": self._attempts[proposal.id]},
            )

    async def retry(self, proposal: TaskProposal) -> CommitResult:
        return await self.commit(proposal, retry=True)

    def rebase(self, proposal: TaskProposal, *, reason: str) -> TaskProposal:
        """Create a new proposal at the current foreground revision.

        Rebase is explicit rather than automatic.  It is only allowed for targets
        whose policy says manual rebase is acceptable.  The caller is responsible
        for confirming the old evidence remains valid after intervening turns.
        """

        policy = self.policies.get(proposal.target)
        if policy is None or policy.stale_policy is not StalePolicy.ALLOW_MANUAL_REBASE:
            raise ValueError(f"target {proposal.target!r} does not allow manual rebase")
        cleaned_reason = reason.strip()
        if not cleaned_reason:
            raise ValueError("rebase reason must not be empty")
        provenance = dict(proposal.provenance)
        provenance.update(
            {
                "rebased_from_proposal_id": proposal.id,
                "rebased_from_revision": proposal.base_revision,
                "rebase_reason": cleaned_reason,
            }
        )
        return replace(
            proposal,
            id=uuid4().hex,
            base_revision=self.tasks.revision,
            provenance=provenance,
            created_at=datetime.now(UTC),
        )

    async def commit_task_result(self, result: TaskResult) -> tuple[CommitResult, ...]:
        if result.status is not TaskStatus.SUCCEEDED or result.output is None:
            return ()
        return tuple([await self.commit(proposal) for proposal in result.output.proposals])

    async def commit_task_results(
        self, results: Iterable[TaskResult]
    ) -> tuple[CommitResult, ...]:
        committed: list[CommitResult] = []
        for result in results:
            committed.extend(await self.commit_task_result(result))
        return tuple(committed)

    def _validate(
        self,
        proposal: TaskProposal,
        policy: CommitTargetPolicy,
        current_revision: int,
    ) -> CommitResult | None:
        if policy.require_source_task_id and not proposal.source_task_id:
            return self._finalize(
                proposal,
                CommitStatus.REJECTED,
                "missing_source_task_id",
                current_revision,
            )
        if policy.expected_worker_kind is not None:
            worker_kind = str(proposal.provenance.get("worker_kind", ""))
            if worker_kind != policy.expected_worker_kind:
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "unexpected_worker_kind",
                    current_revision,
                    metadata={"expected": policy.expected_worker_kind, "actual": worker_kind},
                )
        if proposal.confidence is None or proposal.confidence < policy.min_confidence:
            return self._finalize(
                proposal,
                CommitStatus.REJECTED,
                "confidence_below_threshold",
                current_revision,
                metadata={
                    "confidence": proposal.confidence,
                    "minimum": policy.min_confidence,
                },
            )
        if policy.max_age_s is not None:
            age_s = max(0.0, (datetime.now(UTC) - proposal.created_at).total_seconds())
            if age_s > policy.max_age_s:
                return self._finalize(
                    proposal,
                    CommitStatus.STALE,
                    "proposal_expired",
                    current_revision,
                    next_action=self._stale_next_action(policy),
                    metadata={"age_s": age_s, "max_age_s": policy.max_age_s},
                )
        if proposal.base_revision != current_revision:
            return self._finalize(
                proposal,
                CommitStatus.STALE,
                "foreground_revision_changed",
                current_revision,
                next_action=self._stale_next_action(policy),
            )
        if proposal.target == "memory.append_candidate":
            evidence_type = str(proposal.provenance.get("evidence_type", ""))
            if evidence_type != "asserted_fact":
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "memory_candidate_requires_asserted_fact",
                    current_revision,
                    metadata={"evidence_type": evidence_type},
                )
        if proposal.target == "memory.self_candidate":
            evidence_type = str(proposal.provenance.get("evidence_type", ""))
            if evidence_type != "character_statement":
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "self_memory_candidate_requires_character_statement",
                    current_revision,
                    metadata={"evidence_type": evidence_type},
                )
        if proposal.target == "state.emotion_candidate":
            evidence_type = str(proposal.provenance.get("evidence_type", ""))
            if evidence_type in {"quoted_reference", "user_instruction", "memory_operation"}:
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "emotion_candidate_has_unsafe_evidence_type",
                    current_revision,
                    metadata={"evidence_type": evidence_type},
                )
        if proposal.target == "state.mood_candidate":
            if str(proposal.payload.get("mood", "")) not in CHARACTER_MOODS:
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "mood_not_in_vocabulary",
                    current_revision,
                )
        if proposal.target == "cognition.reflection_candidate":
            insight = str(proposal.payload.get("insight", "")).strip()
            if not insight:
                return self._finalize(
                    proposal,
                    CommitStatus.REJECTED,
                    "reflection_insight_required",
                    current_revision,
                )
            raw_claim = proposal.payload.get("belief_candidate")
            if raw_claim is not None:
                if not isinstance(raw_claim, dict):
                    return self._finalize(
                        proposal,
                        CommitStatus.REJECTED,
                        "invalid_belief_candidate",
                        current_revision,
                    )
                if not all(str(raw_claim.get(key, "")).strip() for key in ("subject", "predicate", "object")):
                    return self._finalize(
                        proposal,
                        CommitStatus.REJECTED,
                        "invalid_belief_candidate",
                        current_revision,
                    )
        if proposal.target == "cognition.goal_candidate":
            invalid = self._validate_goal_candidate(proposal, current_revision)
            if invalid is not None:
                return invalid
        return None

    def _validate_goal_candidate(
        self, proposal: TaskProposal, current_revision: int
    ) -> CommitResult | None:
        objective = str(proposal.payload.get("objective", "")).strip()
        if not objective:
            return self._finalize(proposal, CommitStatus.REJECTED, "goal_objective_required", current_revision)
        try:
            GoalHorizon(str(proposal.payload.get("horizon", "")))
        except ValueError:
            return self._finalize(proposal, CommitStatus.REJECTED, "invalid_goal_horizon", current_revision)
        try:
            urgency = float(proposal.payload.get("urgency"))
        except (TypeError, ValueError):
            urgency = -1.0
        if not 0 <= urgency <= 1:
            return self._finalize(proposal, CommitStatus.REJECTED, "invalid_goal_urgency", current_revision)
        raw_signals = proposal.payload.get("motivation_signals")
        if not isinstance(raw_signals, list) or not raw_signals or len(raw_signals) > 8:
            return self._finalize(proposal, CommitStatus.REJECTED, "invalid_motivation_signals", current_revision)
        compatibility = MOTIVATION_SOURCE_TYPES
        for raw in raw_signals:
            if not isinstance(raw, dict):
                return self._finalize(proposal, CommitStatus.REJECTED, "invalid_motivation_signal", current_revision)
            try:
                kind = MotivationKind(str(raw.get("kind", "")))
            except ValueError:
                return self._finalize(proposal, CommitStatus.REJECTED, "invalid_motivation_kind", current_revision)
            source_type = str(raw.get("source_type", "")).strip().casefold()
            source_id = str(raw.get("source_id", "")).strip()
            rationale = str(raw.get("rationale", "")).strip()
            try:
                strength = float(raw.get("strength"))
            except (TypeError, ValueError):
                strength = -1.0
            if not source_id or not rationale or not 0 <= strength <= 1:
                return self._finalize(proposal, CommitStatus.REJECTED, "invalid_motivation_signal", current_revision)
            if source_type not in compatibility[kind]:
                return self._finalize(
                    proposal, CommitStatus.REJECTED, "goal_source_kind_mismatch", current_revision,
                    metadata={"kind": kind.value, "source_type": source_type},
                )
            if not self._goal_source_exists(source_type, source_id, proposal):
                return self._finalize(
                    proposal, CommitStatus.REJECTED, "goal_source_not_authoritative", current_revision,
                    metadata={"source_type": source_type, "source_id": source_id},
                )
        return None

    def _goal_source_exists(self, source_type: str, source_id: str, proposal: TaskProposal) -> bool:
        runtime = self.tasks.runtime
        if source_type == "event":
            if source_id != _optional_str(proposal.provenance.get("foreground_event_id")):
                return False
            evidence_type = str(proposal.provenance.get("evidence_type", "unknown")).casefold()
            return evidence_type not in {"quoted_reference", "memory_operation", "unknown"}
        if source_type == "memory":
            manager = runtime.memory_manager
            if manager is None:
                return False
            return any(
                record.id == source_id and record.is_active
                for record in manager.store.list_for_character(runtime.memory_scope_id)
            )
        if source_type == "belief":
            manager = getattr(runtime, "long_term_cognition", None)
            if manager is None:
                return False
            return any(record.id == source_id for record in manager.active_beliefs(character_id=runtime.cognition_scope_id))
        if source_type == "state":
            return _state_source_value(runtime.state.snapshot(), source_id)[0]
        return False

    def _resolve_goal_signal(self, raw: dict, proposal: TaskProposal) -> MotivationSignal:
        runtime = self.tasks.runtime
        source_type = str(raw["source_type"]).strip().casefold()
        source_id = str(raw["source_id"]).strip()
        if source_type == "event":
            excerpt = str(proposal.provenance.get("foreground_event_content", "")).strip()
            evidence = GoalEvidenceRef(
                source_type="event", source_id=source_id, excerpt=excerpt or source_id,
                metadata={
                    "event_type": proposal.provenance.get("foreground_event_type"),
                    "event_source": proposal.provenance.get("foreground_event_source"),
                    "evidence_type": proposal.provenance.get("evidence_type"),
                },
            )
        elif source_type == "memory":
            record = next(
                item for item in runtime.memory_manager.store.list_for_character(runtime.memory_scope_id)
                if item.id == source_id and item.is_active
            )
            evidence = GoalEvidenceRef(
                source_type="memory", source_id=record.id, excerpt=record.summary,
                metadata={"evidence_type": record.evidence_type, "kind": record.kind},
            )
        elif source_type == "belief":
            record = next(
                item for item in runtime.long_term_cognition.active_beliefs(character_id=runtime.cognition_scope_id)
                if item.id == source_id
            )
            evidence = GoalEvidenceRef(
                source_type="belief", source_id=record.id,
                excerpt=f"{record.claim.subject} {record.claim.predicate} {record.claim.object}",
                metadata={"confidence": record.confidence, "support_count": record.support_count},
            )
        elif source_type == "state":
            exists, value = _state_source_value(runtime.state.snapshot(), source_id)
            if not exists:
                raise ValueError(f"unknown state source {source_id!r}")
            evidence = GoalEvidenceRef(
                source_type="state", source_id=source_id, excerpt=f"{source_id}={value!r}"
            )
        else:
            raise ValueError(f"unsupported goal source_type {source_type!r}")
        return MotivationSignal(
            kind=MotivationKind(str(raw["kind"])),
            strength=float(raw["strength"]),
            evidence=evidence,
            rationale=str(raw["rationale"]),
        )

    @staticmethod
    def _stale_next_action(policy: CommitTargetPolicy) -> CommitNextAction:
        if policy.stale_policy is StalePolicy.ALLOW_MANUAL_REBASE:
            return CommitNextAction.REBASE
        if policy.stale_policy is StalePolicy.RERUN:
            return CommitNextAction.RERUN
        return CommitNextAction.NONE

    def _apply(self, proposal: TaskProposal) -> str | None:
        runtime = self.tasks.runtime
        if proposal.target in {"memory.append_candidate", "memory.self_candidate"}:
            manager = runtime.memory_manager
            if manager is None:
                raise RuntimeError("memory manager is not configured")
            before = manager.store.list_for_character(runtime.memory_scope_id)
            payload = dict(proposal.payload)
            provenance = dict(proposal.provenance)
            try:
                record = manager.append_background_candidate(
                    character_id=runtime.memory_scope_id,
                    summary=str(payload.get("summary", "")),
                    kind=str(payload.get("kind", "event")) or "event",
                    importance=float(payload.get("importance", 0.5)),
                    source_event_id=_optional_str(provenance.get("foreground_event_id")),
                    source_event_type=_optional_str(provenance.get("foreground_event_type")),
                    tags=(
                        "background_cognition",
                        str(provenance.get("worker_kind") or "memory_extraction"),
                    ),
                    metadata={
                        "source": "background_cognition",
                        "evidence_type": provenance.get("evidence_type"),
                        "proposal_id": proposal.id,
                        "source_task_id": proposal.source_task_id,
                        "base_revision": proposal.base_revision,
                        "confidence": proposal.confidence,
                        "background_provenance": provenance,
                    },
                )
            except Exception:
                manager.store.replace_for_character(runtime.memory_scope_id, before)
                raise
            return record.id

        if proposal.target == "memory.conversation_summary_candidate":
            manager = runtime.memory_manager
            if manager is None:
                raise RuntimeError("memory manager is not configured")
            before = manager.store.list_for_character(runtime.memory_scope_id)
            provenance = dict(proposal.provenance)
            try:
                record = manager.append_background_candidate(
                    character_id=runtime.memory_scope_id,
                    summary=str(proposal.payload.get("summary", "")),
                    kind="conversation_summary",
                    importance=0.45,
                    source_event_id=_optional_str(provenance.get("foreground_event_id")),
                    source_event_type=_optional_str(provenance.get("foreground_event_type")),
                    tags=("background_cognition", "conversation_summary"),
                    metadata={
                        "source": "background_cognition",
                        "evidence_type": "conversation_summary",
                        "proposal_id": proposal.id,
                        "source_task_id": proposal.source_task_id,
                        "base_revision": proposal.base_revision,
                        "confidence": proposal.confidence,
                        "background_provenance": provenance,
                    },
                )
            except Exception:
                manager.store.replace_for_character(runtime.memory_scope_id, before)
                raise
            return record.id

        if proposal.target == "state.emotion_candidate":
            # The emotion worker analyzes the *user's* expressed emotion.  Do not
            # overwrite CharacterState.emotion (the character's own emotion).
            # Store it as an explicit observation in custom state instead.
            before = runtime.state.snapshot()
            value = {
                "emotion": str(proposal.payload.get("emotion", "neutral")),
                "intensity": float(proposal.payload.get("intensity", 0.5)),
                "confidence": proposal.confidence,
                "base_revision": proposal.base_revision,
                # The turn observed, also when the proposal was moved onto a
                # later revision (rebase): a remark of hers in between is not
                # the turn the user's emotion was read from.
                "turn_revision": proposal.provenance.get(
                    "rebased_from_revision", proposal.base_revision
                ),
                "source_task_id": proposal.source_task_id,
                "proposal_id": proposal.id,
            }
            for key in ("valence", "stance"):
                score = proposal.payload.get(key)
                # Proposals can come from any host; bool is an int and NaN compares false.
                if isinstance(score, bool) or not isinstance(score, (int, float)) or score != score:
                    continue
                value[key] = max(-1.0, min(1.0, float(score)))
            turn = seconds(proposal.provenance.get("turn_ended_at"))
            if turn is not None:
                value["turn_ended_at"] = turn
            try:
                runtime.state.apply(
                    StatePatch(
                        custom_updates={"observed_user_emotion": value},
                        reason="background emotion analysis accepted by commit coordinator",
                    )
                )
            except Exception:
                runtime.state.restore(before)
                raise
            return None

        if proposal.target == "state.mood_candidate":
            # The character's own mood, weighed against her mood as it stands
            # now (blend_mood): it may leave her mood as it was. Either way
            # her mood was read for this turn, and the turn's rules do not
            # override that reading.
            before = runtime.state.snapshot()
            now = self.clock()
            current = effective_mood(
                before.emotion,
                before.mood_intensity,
                before.mood_updated_at,
                now=now,
                half_life_seconds=self.mood_half_life_seconds,
                floor=self.mood_floor,
            )
            blended = blend_mood(
                *current,
                str(proposal.payload.get("mood", "")),
                proposal.payload.get("intensity"),
            )
            custom_updates = {}
            turn = seconds(proposal.provenance.get("turn_ended_at"))
            if turn is not None:
                custom_updates[MOOD_TURN_KEY] = turn
            try:
                runtime.state.apply(
                    StatePatch(
                        emotion=None if blended is None else blended[0],
                        custom_updates=custom_updates,
                        reason="background character mood accepted by commit coordinator",
                        mood_intensity=None if blended is None else blended[1],
                        mood_updated_at=None if blended is None else now,
                    )
                )
            except Exception:
                runtime.state.restore(before)
                raise
            return None

        if proposal.target == "cognition.reflection_candidate":
            manager = getattr(runtime, "long_term_cognition", None)
            if manager is None:
                raise RuntimeError("long-term cognition manager is not configured")
            raw_claim = proposal.payload.get("belief_candidate")
            claim = None
            if isinstance(raw_claim, dict):
                claim = BeliefClaim(
                    subject=str(raw_claim["subject"]),
                    predicate=str(raw_claim["predicate"]),
                    object=str(raw_claim["object"]),
                )
            provenance = dict(proposal.provenance)
            evidence = _reflection_evidence(proposal)
            record = ReflectionRecord(
                character_id=runtime.cognition_scope_id,
                insight=str(proposal.payload.get("insight", "")),
                confidence=float(proposal.confidence),
                evidence=evidence,
                claim=claim,
                base_revision=proposal.base_revision,
                source_proposal_id=proposal.id,
                source_task_id=proposal.source_task_id,
                metadata={
                    "source": "background_cognition",
                    "worker_kind": provenance.get("worker_kind"),
                    "foreground_event_id": provenance.get("foreground_event_id"),
                    "foreground_event_type": provenance.get("foreground_event_type"),
                    "foreground_event_source": provenance.get("foreground_event_source"),
                    "evidence_type": provenance.get("evidence_type"),
                },
            )
            committed = manager.commit_reflection(record)
            return committed.reflection.id

        if proposal.target == "cognition.goal_candidate":
            manager = getattr(runtime, "goal_manager", None)
            if manager is None:
                raise RuntimeError("goal manager is not configured")
            signals = tuple(self._resolve_goal_signal(raw, proposal) for raw in proposal.payload["motivation_signals"])
            record = GoalRecord(
                character_id=runtime.goal_scope_id,
                objective=str(proposal.payload["objective"]),
                horizon=GoalHorizon(str(proposal.payload["horizon"])),
                urgency=float(proposal.payload["urgency"]),
                confidence=float(proposal.confidence),
                motivation_signals=signals,
                conflict_key=_optional_str(proposal.payload.get("conflict_key")),
                source_proposal_ids=(proposal.id,),
                source_task_ids=(proposal.source_task_id,) if proposal.source_task_id else (),
                base_revisions=(proposal.base_revision,),
                metadata={
                    "source": "background_cognition",
                    "worker_kind": proposal.provenance.get("worker_kind"),
                    "foreground_event_id": proposal.provenance.get("foreground_event_id"),
                },
            )
            committed = manager.commit_candidate(record)
            return committed.goal.id

        raise RuntimeError(f"no authoritative commit handler for target {proposal.target!r}")

    def _existing_authority_decision(
        self, proposal: TaskProposal, current_revision: int
    ) -> CommitResult | None:
        """Detect duplicates/conflicts already present in authoritative stores.

        The coordinator journal is intentionally in-memory, so this second
        check prevents duplicate writes after coordinator recreation and reconciles
        background extraction with memory already written by the foreground turn.
        """

        runtime = self.tasks.runtime
        if proposal.target in {
            "memory.append_candidate",
            "memory.self_candidate",
            "memory.conversation_summary_candidate",
        }:
            manager = runtime.memory_manager
            if manager is None:
                return None
            active = [
                record
                for record in manager.store.list_for_character(runtime.memory_scope_id)
                if record.is_active
            ]
            proposed_summary = _normalize_text(str(proposal.payload.get("summary", "")))

            if proposal.target == "memory.self_candidate":
                exact = [
                    record for record in active
                    if _normalize_text(record.summary) == proposed_summary
                ]
                if exact:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "authoritative_memory_already_contains_candidate",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in exact]},
                    )

            if proposal.target == "memory.append_candidate":
                event_id = _optional_str(proposal.provenance.get("foreground_event_id"))
                # Only a memory the foreground wrote for this event competes with
                # extraction. Items extracted from the same message are siblings:
                # "my name is Dawn and my cat is Bun" is two facts, not a conflict.
                same_source = [
                    record for record in active
                    if event_id is not None and record.source_event_id == event_id
                    and record.metadata.get("source") != "background_cognition"
                ]
                exact = [
                    record for record in active
                    if _normalize_text(record.summary) == proposed_summary
                ]
                if exact:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "authoritative_memory_already_contains_candidate",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in exact]},
                    )
                if same_source:
                    return self._finalize(
                        proposal,
                        CommitStatus.CONFLICT,
                        "foreground_memory_already_exists_for_source_event",
                        current_revision,
                        next_action=CommitNextAction.MANUAL_REVIEW,
                        metadata={"existing_record_ids": [record.id for record in same_source]},
                    )

            if proposal.target == "memory.conversation_summary_candidate":
                same_revision = [
                    record for record in active
                    if record.kind == "conversation_summary"
                    and record.metadata.get("base_revision") == proposal.base_revision
                ]
                if same_revision:
                    exact = [
                        record for record in same_revision
                        if _normalize_text(record.summary) == proposed_summary
                    ]
                    if exact:
                        return self._finalize(
                            proposal,
                            CommitStatus.DUPLICATE,
                            "authoritative_summary_already_committed",
                            current_revision,
                            metadata={"existing_record_ids": [record.id for record in exact]},
                        )
                    return self._finalize(
                        proposal,
                        CommitStatus.CONFLICT,
                        "different_summary_already_committed_for_revision",
                        current_revision,
                        next_action=CommitNextAction.MANUAL_REVIEW,
                        metadata={"existing_record_ids": [record.id for record in same_revision]},
                    )

        if proposal.target == "state.emotion_candidate":
            current = runtime.state.custom.get("observed_user_emotion")
            if isinstance(current, dict) and current.get("base_revision") == proposal.base_revision:
                proposed = {
                    "emotion": str(proposal.payload.get("emotion", "neutral")),
                    "intensity": float(proposal.payload.get("intensity", 0.5)),
                }
                existing = {
                    "emotion": str(current.get("emotion", "neutral")),
                    "intensity": float(current.get("intensity", 0.5)),
                }
                if proposed == existing:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "authoritative_emotion_observation_already_committed",
                        current_revision,
                        metadata={"existing_proposal_id": current.get("proposal_id")},
                    )
                return self._finalize(
                    proposal,
                    CommitStatus.CONFLICT,
                    "different_emotion_observation_already_committed_for_revision",
                    current_revision,
                    next_action=CommitNextAction.MANUAL_REVIEW,
                    metadata={"existing_proposal_id": current.get("proposal_id")},
                )

        if proposal.target == "cognition.reflection_candidate":
            manager = getattr(runtime, "long_term_cognition", None)
            if manager is not None:
                reflections = manager.store.list_reflections(runtime.cognition_scope_id)
                same_proposal = [
                    record for record in reflections
                    if record.source_proposal_id == proposal.id
                ]
                if same_proposal:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "reflection_proposal_already_committed",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in same_proposal]},
                    )
                event_id = _optional_str(proposal.provenance.get("foreground_event_id"))
                insight = _normalize_text(str(proposal.payload.get("insight", "")))
                claim = _claim_fingerprint(proposal.payload.get("belief_candidate"))
                semantic = [
                    record for record in reflections
                    if event_id is not None
                    and any(item.source_id == event_id for item in record.evidence)
                    and _normalize_text(record.insight) == insight
                    and _record_claim_fingerprint(record) == claim
                ]
                if semantic:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "authoritative_reflection_already_contains_candidate",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in semantic]},
                    )
        if proposal.target == "cognition.goal_candidate":
            manager = getattr(runtime, "goal_manager", None)
            if manager is not None:
                goals = manager.store.list_goals(runtime.goal_scope_id)
                same_proposal = [record for record in goals if proposal.id in record.source_proposal_ids]
                if same_proposal:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "goal_proposal_already_committed",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in same_proposal]},
                    )
                event_id = _optional_str(proposal.provenance.get("foreground_event_id"))
                objective = _normalize_text(str(proposal.payload.get("objective", "")))
                horizon = str(proposal.payload.get("horizon", ""))
                same_event = [
                    record for record in goals
                    if _normalize_text(record.objective) == objective
                    and record.horizon.value == horizon
                    and event_id is not None
                    and any(signal.evidence.source_type == "event" and signal.evidence.source_id == event_id for signal in record.motivation_signals)
                ]
                if same_event:
                    return self._finalize(
                        proposal,
                        CommitStatus.DUPLICATE,
                        "authoritative_goal_already_contains_candidate",
                        current_revision,
                        metadata={"existing_record_ids": [record.id for record in same_event]},
                    )
        if proposal.target == "state.mood_candidate":
            turn = seconds(proposal.provenance.get("turn_ended_at"))
            judged = seconds(runtime.state.custom.get(MOOD_TURN_KEY))
            if turn is not None and judged is not None and turn < judged:
                # A reading of an earlier turn that arrived late: her mood was
                # already set from a later one.
                return self._finalize(
                    proposal,
                    CommitStatus.STALE,
                    "newer_mood_already_committed",
                    current_revision,
                    metadata={"turn_ended_at": turn, "mood_turn_ended_at": judged},
                )
        return None

    def _identity(self, proposal: TaskProposal) -> tuple[str | None, str | None, str]:
        payload_json = json.dumps(
            _jsonable(dict(proposal.payload)), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        fingerprint = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        provenance = dict(proposal.provenance)
        event_id = str(provenance.get("foreground_event_id", ""))

        if proposal.target == "memory.append_candidate":
            summary = " ".join(str(proposal.payload.get("summary", "")).lower().split())
            semantic = hashlib.sha256(
                f"memory|{event_id}|{summary}".encode("utf-8")
            ).hexdigest()
            return semantic, None, fingerprint
        if proposal.target == "memory.self_candidate":
            summary = _normalize_text(str(proposal.payload.get("summary", "")))
            semantic = hashlib.sha256(f"self|{event_id}|{summary}".encode("utf-8")).hexdigest()
            return semantic, None, fingerprint
        if proposal.target == "memory.conversation_summary_candidate":
            return None, f"summary:{proposal.base_revision}", fingerprint
        if proposal.target == "state.emotion_candidate":
            return None, f"observed_user_emotion:{proposal.base_revision}", fingerprint
        if proposal.target == "state.mood_candidate":
            return None, f"character_mood:{proposal.base_revision}", fingerprint
        if proposal.target == "cognition.reflection_candidate":
            insight = _normalize_text(str(proposal.payload.get("insight", "")))
            claim = _claim_fingerprint(proposal.payload.get("belief_candidate"))
            semantic = hashlib.sha256(
                f"reflection|{event_id}|{insight}|{claim}".encode("utf-8")
            ).hexdigest()
            return semantic, None, fingerprint
        if proposal.target == "cognition.goal_candidate":
            objective = _normalize_text(str(proposal.payload.get("objective", "")))
            horizon = str(proposal.payload.get("horizon", ""))
            conflict_key = _normalize_text(str(proposal.payload.get("conflict_key") or ""))
            semantic = hashlib.sha256(
                f"goal|{event_id}|{horizon}|{objective}|{conflict_key}".encode("utf-8")
            ).hexdigest()
            return semantic, None, fingerprint
        return None, None, fingerprint

    def _finalize(
        self,
        proposal: TaskProposal,
        status: CommitStatus,
        reason: str,
        current_revision: int,
        *,
        next_action: CommitNextAction = CommitNextAction.NONE,
        commit_sequence: int | None = None,
        applied_record_id: str | None = None,
        metadata: dict | None = None,
    ) -> CommitResult:
        result = CommitResult(
            proposal_id=proposal.id,
            target=proposal.target,
            status=status,
            reason=reason,
            base_revision=proposal.base_revision,
            current_revision=current_revision,
            next_action=next_action,
            commit_sequence=commit_sequence,
            applied_record_id=applied_record_id,
            metadata=metadata or {},
        )
        self._processed[proposal.id] = result
        self._let_go_of_the_oldest()
        self._events.append(
            CommitLifecycleEvent(
                proposal_id=proposal.id,
                target=proposal.target,
                status=status,
                reason=reason,
                base_revision=proposal.base_revision,
                current_revision=current_revision,
                commit_sequence=commit_sequence,
            )
        )
        return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _state_source_value(snapshot, source_id: str) -> tuple[bool, object]:
    if source_id in {"emotion", "energy", "trust", "favorability", "relationship_stage"}:
        return True, getattr(snapshot, source_id)
    if source_id.startswith("custom:"):
        key = source_id.split(":", 1)[1]
        if key and key in snapshot.custom:
            return True, snapshot.custom[key]
    return False, None


def _reflection_evidence(proposal: TaskProposal) -> tuple[CognitionEvidenceRef, ...]:
    provenance = dict(proposal.provenance)
    raw = provenance.get("evidence", ())
    excerpts = (
        [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, (list, tuple))
        else []
    )
    if not excerpts:
        return ()
    event_id = _optional_str(provenance.get("foreground_event_id"))
    source_task_id = _optional_str(proposal.source_task_id)
    source_type = "event" if event_id is not None else "task"
    source_id = event_id or source_task_id
    if source_id is None:
        return ()
    evidence_type = str(provenance.get("evidence_type", "unknown")).strip() or "unknown"
    metadata = {
        "foreground_event_type": provenance.get("foreground_event_type"),
        "foreground_event_source": provenance.get("foreground_event_source"),
        "source_task_id": proposal.source_task_id,
    }
    return tuple(
        CognitionEvidenceRef(
            source_type=source_type,
            source_id=source_id,
            evidence_type=evidence_type,
            excerpt=excerpt,
            confidence=proposal.confidence,
            metadata=metadata,
        )
        for excerpt in excerpts
    )


def _claim_fingerprint(value) -> str:
    if not isinstance(value, dict):
        return ""
    return "|".join(
        _normalize_text(str(value.get(key, "")))
        for key in ("subject", "predicate", "object")
    )


def _record_claim_fingerprint(record: ReflectionRecord) -> str:
    if record.claim is None:
        return ""
    return "|".join(
        (_normalize_text(record.claim.subject), _normalize_text(record.claim.predicate), _normalize_text(record.claim.object))
    )
