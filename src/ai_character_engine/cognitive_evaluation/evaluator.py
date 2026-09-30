from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, Sequence

from ai_character_engine.collaboration.models import SpecialistRunStatus, VerificationDecision
from ai_character_engine.goals.models import GoalStatus, MotivationKind
from ai_character_engine.long_term_cognition.models import BeliefStatus

from .models import (
    CognitiveEvalCase,
    CognitiveEvalDimension,
    CognitiveEvalResult,
    CognitiveEvalTrace,
)
from ai_character_engine.evaluation.models import Severity


def _norm(value: str) -> str:
    return " ".join(str(value).casefold().split())


@dataclass(frozen=True, slots=True)
class CognitiveEvalPolicy:
    min_active_belief_support: int = 2
    require_known_sources: bool = True
    require_active_goal_memory_sources: bool = True
    require_active_goal_belief_sources: bool = True

    def __post_init__(self) -> None:
        if self.min_active_belief_support < 1:
            raise ValueError("min_active_belief_support must be >= 1")


@dataclass(frozen=True, slots=True)
class CognitiveJudgeVerdict:
    dimension: CognitiveEvalDimension
    status: str
    rationale: str
    severity: Severity | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", CognitiveEvalDimension(self.dimension))
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("invalid cognitive judge status")
        if not self.rationale.strip():
            raise ValueError("cognitive judge rationale must not be empty")
        if self.status == "failed":
            object.__setattr__(self, "severity", Severity(self.severity))
        elif self.severity is not None:
            raise ValueError("nonfailed cognitive judge verdict cannot have severity")
        object.__setattr__(self, "evidence", tuple(self.evidence))


class CognitiveJudgeAdapter(Protocol):
    async def evaluate(
        self,
        case: CognitiveEvalCase,
        *,
        dimensions: tuple[CognitiveEvalDimension, ...],
    ) -> Sequence[CognitiveJudgeVerdict]: ...


class CallableCognitiveJudgeAdapter:
    def __init__(self, func: Callable[..., Awaitable[Sequence[CognitiveJudgeVerdict]]]) -> None:
        self.func = func

    async def evaluate(self, case: CognitiveEvalCase, *, dimensions: tuple[CognitiveEvalDimension, ...]) -> Sequence[CognitiveJudgeVerdict]:
        return await self.func(case, dimensions=dimensions)


class CognitiveEvaluator:
    """Read-only cognition assessment across Memory/Reflection/Belief/Goal/Collaboration.

    Rule-based checks own structural/provenance invariants. An optional semantic
    judge may assess harder questions such as whether a specialist finding is
    actually relevant to its objective. Judge output never mutates runtime state
    and never becomes evidence for Memory, Belief, or Goal.
    """

    SEMANTIC_DIMENSIONS = (
        CognitiveEvalDimension.EVIDENCE_FAITHFULNESS,
        CognitiveEvalDimension.SPECIALIST_RELEVANCE,
        CognitiveEvalDimension.VERIFIER_QUALITY,
        CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
    )

    def __init__(
        self,
        *,
        policy: CognitiveEvalPolicy | None = None,
        judge: CognitiveJudgeAdapter | None = None,
        semantic_judge_enabled: bool = False,
    ) -> None:
        if type(semantic_judge_enabled) is not bool:
            raise ValueError("semantic_judge_enabled must be a boolean")
        if semantic_judge_enabled and judge is None:
            raise ValueError("semantic judge enabled but no judge adapter was provided")
        self.policy = policy or CognitiveEvalPolicy()
        self.judge = judge
        self.semantic_judge_enabled = semantic_judge_enabled

    async def evaluate(self, case: CognitiveEvalCase) -> CognitiveEvalResult:
        trace: list[CognitiveEvalTrace] = []
        trace.extend(self._evidence(case))
        trace.extend(self._beliefs(case))
        trace.extend(self._goals(case))
        trace.extend(self._specialists(case))
        trace.extend(self._verifier(case))
        trace.extend(self._long_horizon(case))
        executed = False
        if self.semantic_judge_enabled:
            assert self.judge is not None
            verdicts = tuple(await self.judge.evaluate(case, dimensions=self.SEMANTIC_DIMENSIONS))
            seen: set[CognitiveEvalDimension] = set()
            for verdict in verdicts:
                if not isinstance(verdict, CognitiveJudgeVerdict):
                    raise ValueError("cognitive judge must return CognitiveJudgeVerdict values")
                if verdict.dimension not in self.SEMANTIC_DIMENSIONS:
                    raise ValueError(f"cognitive judge returned unsupported dimension: {verdict.dimension.value}")
                if verdict.dimension in seen:
                    raise ValueError("cognitive judge returned duplicate dimension verdict")
                seen.add(verdict.dimension)
                trace.append(CognitiveEvalTrace(
                    code=f"semantic.{verdict.dimension.value}",
                    dimension=verdict.dimension,
                    status=verdict.status,
                    message=verdict.rationale,
                    severity=verdict.severity,
                    evidence=verdict.evidence,
                    source="semantic_judge",
                ))
            executed = True
        return CognitiveEvalResult(case.case_id, tuple(trace), semantic_judge_executed=executed)

    def _evidence(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        sources = case.source_map()
        def check(source_type: str, source_id: str, artifact_type: str, artifact_id: str, code: str) -> None:
            key = (_norm(source_type), _norm(source_id))
            exists = key in sources
            out.append(CognitiveEvalTrace(
                code=code,
                dimension=CognitiveEvalDimension.EVIDENCE_FAITHFULNESS,
                status="passed" if exists or not self.policy.require_known_sources else "failed",
                message=("Evidence source resolves in the evaluation source catalog." if exists else "Evidence source is missing from the evaluation source catalog."),
                severity=None if exists or not self.policy.require_known_sources else Severity.HIGH,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                evidence=(f"{source_type}:{source_id}",),
            ))
        for artifact_type, records in (("memory", case.memories), ("reflection", case.reflections), ("belief", case.beliefs), ("goal", case.goals)):
            for record in records:
                matches = record.character_id == case.character_id
                out.append(CognitiveEvalTrace(
                    "evidence.character_scope", CognitiveEvalDimension.EVIDENCE_FAITHFULNESS,
                    "passed" if matches else "failed",
                    "Artifact belongs to the evaluated character scope." if matches else "Artifact belongs to a different character scope.",
                    None if matches else Severity.CRITICAL, artifact_type, record.id,
                ))
        for memory in case.memories:
            if memory.source_event_id:
                check("event", memory.source_event_id, "memory", memory.id, "evidence.memory_source_resolves")
        for reflection in case.reflections:
            for ref in reflection.evidence:
                check(ref.source_type, ref.source_id, "reflection", reflection.id, "evidence.reflection_source_resolves")
        for belief in case.beliefs:
            for ref in belief.evidence:
                check(ref.source_type, ref.source_id, "belief", belief.id, "evidence.belief_source_resolves")
        for goal in case.goals:
            for signal in goal.motivation_signals:
                check(signal.evidence.source_type, signal.evidence.source_id, "goal", goal.id, "evidence.goal_source_resolves")
        if not out:
            out.append(CognitiveEvalTrace("evidence.no_references", CognitiveEvalDimension.EVIDENCE_FAITHFULNESS, "skipped", "No traceable cognition evidence references were supplied."))
        return out

    def _beliefs(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        active = [x for x in case.beliefs if x.status is BeliefStatus.ACTIVE]
        by_key: dict[tuple[str, str], list] = {}
        by_value: dict[tuple[str, str, str], list] = {}
        for belief in active:
            by_key.setdefault(belief.claim.key, []).append(belief)
            by_value.setdefault(belief.claim.value_key, []).append(belief)
            enough = belief.support_count >= self.policy.min_active_belief_support
            out.append(CognitiveEvalTrace(
                "belief.independent_support", CognitiveEvalDimension.BELIEF_CONSISTENCY,
                "passed" if enough else "failed",
                f"Active belief has {belief.support_count} independent evidence source(s).",
                None if enough else Severity.HIGH, "belief", belief.id,
            ))
            safe = all(ref.safe_for_belief for ref in belief.evidence)
            out.append(CognitiveEvalTrace(
                "belief.safe_evidence", CognitiveEvalDimension.BELIEF_CONSISTENCY,
                "passed" if safe else "failed",
                "Active belief uses only promotion-safe evidence." if safe else "Active belief contains evidence that is unsafe for automatic belief promotion.",
                None if safe else Severity.CRITICAL, "belief", belief.id,
            ))
        for key, records in by_key.items():
            values = {x.claim.value_key for x in records}
            conflict = len(values) > 1
            out.append(CognitiveEvalTrace(
                "belief.active_conflict", CognitiveEvalDimension.BELIEF_CONSISTENCY,
                "failed" if conflict else "passed",
                "Conflicting values are simultaneously active for one belief key." if conflict else "No simultaneous active-value conflict for this belief key.",
                Severity.CRITICAL if conflict else None, "belief_key", "|".join(key),
                tuple(x.id for x in records),
            ))
        for value_key, records in by_value.items():
            duplicate = len(records) > 1
            out.append(CognitiveEvalTrace(
                "belief.duplicate_active_value", CognitiveEvalDimension.BELIEF_CONSISTENCY,
                "failed" if duplicate else "passed",
                "Duplicate active belief records represent the same structured claim." if duplicate else "Structured active belief value is unique.",
                Severity.MEDIUM if duplicate else None, "belief_value", "|".join(value_key), tuple(x.id for x in records),
            ))
        if not out:
            out.append(CognitiveEvalTrace("belief.no_active_beliefs", CognitiveEvalDimension.BELIEF_CONSISTENCY, "skipped", "No active beliefs were supplied."))
        return out

    def _goals(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        sources = case.source_map()
        compatibility = {
            MotivationKind.EXPLICIT_REQUEST: {"event", "memory"},
            MotivationKind.UNFINISHED_INTENT: {"event", "memory"},
            MotivationKind.COMMITMENT: {"event", "memory"},
            MotivationKind.BELIEF_ALIGNMENT: {"belief"},
            MotivationKind.STATE_PRESSURE: {"state"},
        }
        active_or_blocked = [x for x in case.goals if x.status in {GoalStatus.ACTIVE, GoalStatus.BLOCKED}]
        for goal in active_or_blocked:
            for signal in goal.motivation_signals:
                source_type = _norm(signal.evidence.source_type)
                compatible = source_type in compatibility[signal.kind]
                out.append(CognitiveEvalTrace(
                    "goal.motivation_source_compatible", CognitiveEvalDimension.GOAL_GROUNDING,
                    "passed" if compatible else "failed",
                    "Motivation kind is compatible with its canonical source type." if compatible else "Motivation kind is incompatible with its source type.",
                    None if compatible else Severity.HIGH, "goal", goal.id,
                ))
                source = sources.get((source_type, _norm(signal.evidence.source_id)))
                active_source = True
                if source is not None and source_type == "memory" and self.policy.require_active_goal_memory_sources:
                    active_source = source.status == "active"
                if source is not None and source_type == "belief" and self.policy.require_active_goal_belief_sources:
                    active_source = source.status == "active"
                out.append(CognitiveEvalTrace(
                    "goal.source_is_active", CognitiveEvalDimension.GOAL_GROUNDING,
                    "passed" if active_source else "failed",
                    "Goal evidence points to an active canonical source." if active_source else "Goal evidence points to an inactive canonical source.",
                    None if active_source else Severity.HIGH, "goal", goal.id,
                ))
        groups: dict[str, list] = {}
        for goal in case.goals:
            if goal.status is GoalStatus.ACTIVE and goal.conflict_key:
                groups.setdefault(_norm(goal.conflict_key), []).append(goal)
        for key, records in groups.items():
            distinct = {x.semantic_key for x in records}
            conflict = len(distinct) > 1
            out.append(CognitiveEvalTrace(
                "goal.unreconciled_active_conflict", CognitiveEvalDimension.GOAL_GROUNDING,
                "failed" if conflict else "passed",
                "Mutually exclusive goals remain active instead of being blocked." if conflict else "No unreconciled active conflict for this goal conflict key.",
                Severity.HIGH if conflict else None, "goal_conflict", key, tuple(x.id for x in records),
            ))
        if not out:
            out.append(CognitiveEvalTrace("goal.no_active_goals", CognitiveEvalDimension.GOAL_GROUNDING, "skipped", "No active or blocked goals were supplied."))
        return out

    def _specialists(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        for collaboration in case.collaborations:
            work = {item.item_id: item for item in collaboration.plan.work_items}
            for finding in collaboration.findings:
                item = work.get(finding.work_item_id)
                aligned = item is not None and item.specialist_id == finding.specialist_id
                out.append(CognitiveEvalTrace(
                    "specialist.work_item_alignment", CognitiveEvalDimension.SPECIALIST_RELEVANCE,
                    "passed" if aligned else "failed",
                    "Finding maps to its planned specialist work item." if aligned else "Finding does not map to a matching planned specialist work item.",
                    None if aligned else Severity.HIGH, "collaboration", collaboration.collaboration_id,
                ))
                inside = item is not None and set(finding.evidence_source_ids).issubset(set(item.source_ids))
                out.append(CognitiveEvalTrace(
                    "specialist.evidence_slice", CognitiveEvalDimension.SPECIALIST_RELEVANCE,
                    "passed" if inside else "failed",
                    "Finding cites only sources assigned to its work item." if inside else "Finding cites evidence outside its assigned source slice.",
                    None if inside else Severity.CRITICAL, "collaboration", collaboration.collaboration_id,
                ))
            if not collaboration.findings:
                out.append(CognitiveEvalTrace("specialist.no_findings", CognitiveEvalDimension.SPECIALIST_RELEVANCE, "skipped", "Collaboration produced no specialist findings.", artifact_type="collaboration", artifact_id=collaboration.collaboration_id))
        if not case.collaborations:
            out.append(CognitiveEvalTrace("specialist.no_collaboration", CognitiveEvalDimension.SPECIALIST_RELEVANCE, "skipped", "No specialist collaboration results were supplied."))
        return out

    def _verifier(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        for collaboration in case.collaborations:
            successful = {x.work_item_id for x in collaboration.findings if x.status is SpecialistRunStatus.SUCCEEDED}
            accepted = set(collaboration.verification.accepted_work_item_ids)
            valid_ids = accepted.issubset(successful)
            out.append(CognitiveEvalTrace(
                "verifier.accepts_only_successful", CognitiveEvalDimension.VERIFIER_QUALITY,
                "passed" if valid_ids else "failed",
                "Verifier accepts only successfully completed work items." if valid_ids else "Verifier accepted an unknown, failed, or timed-out work item.",
                None if valid_ids else Severity.CRITICAL, "collaboration", collaboration.collaboration_id,
            ))
            decision = collaboration.verification.decision
            coherent = True
            if decision is VerificationDecision.ACCEPT: coherent = accepted == successful and bool(successful)
            elif decision is VerificationDecision.PARTIAL: coherent = bool(accepted) and accepted < successful
            elif decision in {VerificationDecision.REJECT, VerificationDecision.UNVERIFIED}: coherent = not accepted
            out.append(CognitiveEvalTrace(
                "verifier.decision_coherence", CognitiveEvalDimension.VERIFIER_QUALITY,
                "passed" if coherent else "failed",
                "Verifier decision is coherent with the accepted set." if coherent else "Verifier decision and accepted set are inconsistent.",
                None if coherent else Severity.HIGH, "collaboration", collaboration.collaboration_id,
            ))
            known_items = {x.item_id for x in collaboration.plan.work_items}
            issue_ids_valid = all(issue.work_item_id is None or issue.work_item_id in known_items for issue in collaboration.verification.issues)
            out.append(CognitiveEvalTrace(
                "verifier.issue_references", CognitiveEvalDimension.VERIFIER_QUALITY,
                "passed" if issue_ids_valid else "failed",
                "Verifier issues reference known work items." if issue_ids_valid else "Verifier issue references an unknown work item.",
                None if issue_ids_valid else Severity.MEDIUM, "collaboration", collaboration.collaboration_id,
            ))
        if not case.collaborations:
            out.append(CognitiveEvalTrace("verifier.no_collaboration", CognitiveEvalDimension.VERIFIER_QUALITY, "skipped", "No verifier reports were supplied."))
        return out

    def _long_horizon(self, case: CognitiveEvalCase) -> list[CognitiveEvalTrace]:
        out: list[CognitiveEvalTrace] = []
        if len(case.timeline) < 2:
            return [CognitiveEvalTrace("timeline.insufficient_frames", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY, "skipped", "At least two timeline frames are required for long-horizon evaluation.")]
        revisions = [frame.revision for frame in case.timeline]
        monotonic = all(b > a for a, b in zip(revisions, revisions[1:]))
        out.append(CognitiveEvalTrace(
            "timeline.revision_monotonic", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
            "passed" if monotonic else "failed",
            "Timeline revisions are strictly increasing." if monotonic else "Timeline revisions move backward or repeat.",
            None if monotonic else Severity.HIGH,
        ))
        belief_identity: dict[str, tuple] = {}
        retired_beliefs: set[str] = set()
        goal_identity: dict[str, tuple] = {}
        terminal_goals: set[str] = set()
        for frame in case.timeline:
            for belief in frame.beliefs:
                identity = belief.claim.value_key
                stable = belief.id not in belief_identity or belief_identity[belief.id] == identity
                out.append(CognitiveEvalTrace(
                    "timeline.belief_identity_stable", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
                    "passed" if stable else "failed",
                    "Belief id keeps a stable structured claim across frames." if stable else "Belief id changed its structured claim across frames.",
                    None if stable else Severity.CRITICAL, "belief", belief.id,
                ))
                belief_identity.setdefault(belief.id, identity)
                resurrected = belief.id in retired_beliefs and belief.status is not BeliefStatus.RETIRED
                out.append(CognitiveEvalTrace(
                    "timeline.retired_belief_terminal", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
                    "failed" if resurrected else "passed",
                    "Retired belief was reactivated." if resurrected else "Retired belief terminality is preserved.",
                    Severity.HIGH if resurrected else None, "belief", belief.id,
                ))
                if belief.status is BeliefStatus.RETIRED: retired_beliefs.add(belief.id)
            for goal in frame.goals:
                identity = goal.semantic_key
                stable = goal.id not in goal_identity or goal_identity[goal.id] == identity
                out.append(CognitiveEvalTrace(
                    "timeline.goal_identity_stable", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
                    "passed" if stable else "failed",
                    "Goal id keeps a stable semantic identity across frames." if stable else "Goal id changed objective/horizon/conflict identity across frames.",
                    None if stable else Severity.CRITICAL, "goal", goal.id,
                ))
                goal_identity.setdefault(goal.id, identity)
                resurrected = goal.id in terminal_goals and goal.status not in {GoalStatus.COMPLETED, GoalStatus.RETIRED}
                out.append(CognitiveEvalTrace(
                    "timeline.terminal_goal_not_reactivated", CognitiveEvalDimension.LONG_HORIZON_CONSISTENCY,
                    "failed" if resurrected else "passed",
                    "Completed/retired goal was reactivated." if resurrected else "Completed/retired goal terminality is preserved.",
                    Severity.HIGH if resurrected else None, "goal", goal.id,
                ))
                if goal.status in {GoalStatus.COMPLETED, GoalStatus.RETIRED}: terminal_goals.add(goal.id)
        return out
