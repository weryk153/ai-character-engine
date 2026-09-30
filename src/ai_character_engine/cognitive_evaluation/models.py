from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ai_character_engine.collaboration.models import (
    CollaborationResult,
    CognitiveWorkItem,
    CognitiveWorkPlan,
    SpecialistFinding,
    SpecialistKind,
    SpecialistRunStatus,
    VerificationDecision,
    VerificationIssue,
    VerificationReport,
)
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals.models import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalRecord,
    GoalStatus,
    MotivationKind,
    MotivationSignal,
)
from ai_character_engine.long_term_cognition.models import (
    BeliefClaim,
    BeliefRecord,
    BeliefStatus,
    CognitionEvidenceRef,
    ReflectionRecord,
    ReflectionStatus,
)
from ai_character_engine.memory.models import MemoryRecord
from ai_character_engine.state.models import CharacterState, CharacterStateSnapshot

from ai_character_engine.evaluation.models import Severity


def _required(value: str, name: str) -> str:
    cleaned = " ".join(str(value).strip().split())
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


class CognitiveEvalDimension(str, Enum):
    EVIDENCE_FAITHFULNESS = "evidence_faithfulness"
    BELIEF_CONSISTENCY = "belief_consistency"
    GOAL_GROUNDING = "goal_grounding"
    SPECIALIST_RELEVANCE = "specialist_relevance"
    VERIFIER_QUALITY = "verifier_quality"
    LONG_HORIZON_CONSISTENCY = "long_horizon_consistency"


@dataclass(frozen=True, slots=True)
class CognitiveEvalSource:
    source_type: str
    source_id: str
    content: str
    status: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", _required(self.source_type, "source_type"))
        object.__setattr__(self, "source_id", _required(self.source_id, "source_id"))
        if not isinstance(self.content, str):
            raise ValueError("source content must be a string")
        status = None if self.status is None else str(self.status).strip()
        object.__setattr__(self, "status", status or None)
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.source_type), _normalize(self.source_id))


@dataclass(frozen=True, slots=True)
class CognitiveTimelineFrame:
    revision: int
    beliefs: tuple[BeliefRecord, ...] = ()
    goals: tuple[GoalRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("timeline revision must be >= 0")
        object.__setattr__(self, "beliefs", tuple(self.beliefs))
        object.__setattr__(self, "goals", tuple(self.goals))


@dataclass(frozen=True, slots=True)
class CognitiveEvalCase:
    case_id: str
    character_id: str
    revision: int = 0
    state: CharacterStateSnapshot | CharacterState | None = None
    sources: tuple[CognitiveEvalSource, ...] = ()
    memories: tuple[MemoryRecord, ...] = ()
    reflections: tuple[ReflectionRecord, ...] = ()
    beliefs: tuple[BeliefRecord, ...] = ()
    goals: tuple[GoalRecord, ...] = ()
    collaborations: tuple[CollaborationResult, ...] = ()
    timeline: tuple[CognitiveTimelineFrame, ...] = ()
    expected_failures: tuple[CognitiveEvalDimension, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required(self.case_id, "case_id"))
        object.__setattr__(self, "character_id", _required(self.character_id, "character_id"))
        if self.revision < 0:
            raise ValueError("revision must be >= 0")
        state = self.state.snapshot() if isinstance(self.state, CharacterState) else self.state
        if state is not None and not isinstance(state, CharacterStateSnapshot):
            raise ValueError("state must be CharacterState or CharacterStateSnapshot")
        object.__setattr__(self, "state", copy.deepcopy(state))
        for name in ("sources", "memories", "reflections", "beliefs", "goals", "collaborations", "timeline"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        keys = [source.key for source in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("cognitive eval source keys must be unique")
        if self.expected_failures is not None:
            object.__setattr__(self, "expected_failures", tuple(CognitiveEvalDimension(x) for x in self.expected_failures))

    @classmethod
    def from_artifacts(
        cls,
        *,
        case_id: str,
        character_id: str,
        revision: int = 0,
        state: CharacterStateSnapshot | CharacterState | None = None,
        events: tuple[CharacterEvent, ...] = (),
        memories: tuple[MemoryRecord, ...] = (),
        reflections: tuple[ReflectionRecord, ...] = (),
        beliefs: tuple[BeliefRecord, ...] = (),
        goals: tuple[GoalRecord, ...] = (),
        collaborations: tuple[CollaborationResult, ...] = (),
        extra_sources: tuple[CognitiveEvalSource, ...] = (),
        timeline: tuple[CognitiveTimelineFrame, ...] = (),
        expected_failures: tuple[CognitiveEvalDimension, ...] | None = None,
    ) -> CognitiveEvalCase:
        state_snapshot = state.snapshot() if isinstance(state, CharacterState) else state
        sources: list[CognitiveEvalSource] = list(extra_sources)
        sources.extend(
            CognitiveEvalSource("event", event.id, event.content, metadata={"event_type": event.type, "event_source": event.source})
            for event in events
        )
        sources.extend(
            CognitiveEvalSource("memory", record.id, record.summary, status=record.status, metadata={"kind": record.kind})
            for record in memories
        )
        sources.extend(
            CognitiveEvalSource(
                "belief",
                record.id,
                f"{record.claim.subject} {record.claim.predicate} {record.claim.object}",
                status=record.status.value,
            )
            for record in beliefs
        )
        if state_snapshot is not None:
            for field_name in ("emotion", "energy", "trust", "favorability", "relationship_stage"):
                value = getattr(state_snapshot, field_name)
                sources.append(CognitiveEvalSource("state", field_name, f"{field_name}={value!r}"))
            for key, value in state_snapshot.custom.items():
                sources.append(CognitiveEvalSource("state", f"custom:{key}", f"custom:{key}={value!r}"))
        return cls(
            case_id=case_id,
            character_id=character_id,
            revision=revision,
            state=state_snapshot,
            sources=tuple(sources),
            memories=memories,
            reflections=reflections,
            beliefs=beliefs,
            goals=goals,
            collaborations=collaborations,
            timeline=timeline,
            expected_failures=expected_failures,
        )

    def source_map(self) -> dict[tuple[str, str], CognitiveEvalSource]:
        return {source.key: source for source in self.sources}

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "character_id": self.character_id,
            "revision": self.revision,
            "state": _state_to_dict(self.state),
            "sources": [_source_to_dict(x) for x in self.sources],
            "memories": [_memory_to_dict(x) for x in self.memories],
            "reflections": [_reflection_to_dict(x) for x in self.reflections],
            "beliefs": [_belief_to_dict(x) for x in self.beliefs],
            "goals": [_goal_to_dict(x) for x in self.goals],
            "collaborations": [_collaboration_to_dict(x) for x in self.collaborations],
            "timeline": [_frame_to_dict(x) for x in self.timeline],
            "expected_failures": None if self.expected_failures is None else [x.value for x in self.expected_failures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CognitiveEvalCase:
        raw = dict(data)
        return cls(
            case_id=raw["case_id"],
            character_id=raw["character_id"],
            revision=int(raw.get("revision", 0)),
            state=_state_from_dict(raw.get("state")),
            sources=tuple(_source_from_dict(x) for x in raw.get("sources", ())),
            memories=tuple(_memory_from_dict(x) for x in raw.get("memories", ())),
            reflections=tuple(_reflection_from_dict(x) for x in raw.get("reflections", ())),
            beliefs=tuple(_belief_from_dict(x) for x in raw.get("beliefs", ())),
            goals=tuple(_goal_from_dict(x) for x in raw.get("goals", ())),
            collaborations=tuple(_collaboration_from_dict(x) for x in raw.get("collaborations", ())),
            timeline=tuple(_frame_from_dict(x) for x in raw.get("timeline", ())),
            expected_failures=None if raw.get("expected_failures") is None else tuple(raw["expected_failures"]),
        )


@dataclass(frozen=True, slots=True)
class CognitiveEvalTrace:
    code: str
    dimension: CognitiveEvalDimension
    status: str
    message: str
    severity: Severity | None = None
    artifact_type: str | None = None
    artifact_id: str | None = None
    evidence: tuple[str, ...] = ()
    source: str = "rule_based"

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required(self.code, "trace code"))
        object.__setattr__(self, "message", _required(self.message, "trace message"))
        object.__setattr__(self, "dimension", CognitiveEvalDimension(self.dimension))
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("invalid cognitive eval trace status")
        if self.source not in {"rule_based", "semantic_judge"}:
            raise ValueError("invalid cognitive eval trace source")
        if self.status == "failed":
            object.__setattr__(self, "severity", Severity(self.severity))
        elif self.severity is not None:
            raise ValueError("nonfailed cognitive eval trace must not specify severity")
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True, slots=True)
class CognitiveEvalResult:
    case_id: str
    trace: tuple[CognitiveEvalTrace, ...]
    semantic_judge_executed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required(self.case_id, "case_id"))
        object.__setattr__(self, "trace", tuple(self.trace))

    @property
    def violations(self) -> tuple[CognitiveEvalTrace, ...]:
        return tuple(item for item in self.trace if item.status == "failed")

    @property
    def evaluated_dimensions(self) -> tuple[CognitiveEvalDimension, ...]:
        return tuple(
            dimension
            for dimension in CognitiveEvalDimension
            if any(item.dimension is dimension and item.status != "skipped" for item in self.trace)
        )

    @property
    def passed(self) -> bool | None:
        return not self.violations if self.evaluated_dimensions else None

    @property
    def severity(self) -> Severity | None:
        return max((item.severity for item in self.violations), key=lambda value: value.penalty, default=None)

    @property
    def quality_score(self) -> float | None:
        dimensions = self.evaluated_dimensions
        if not dimensions:
            return None
        penalties = []
        for dimension in dimensions:
            penalties.append(max((x.severity.penalty for x in self.violations if x.dimension is dimension), default=0.0))
        return sum(1.0 - penalty for penalty in penalties) / len(penalties)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "trace": [
                {
                    "code": x.code,
                    "dimension": x.dimension.value,
                    "status": x.status,
                    "message": x.message,
                    "severity": x.severity.value if x.severity else None,
                    "artifact_type": x.artifact_type,
                    "artifact_id": x.artifact_id,
                    "evidence": list(x.evidence),
                    "source": x.source,
                }
                for x in self.trace
            ],
            "semantic_judge_executed": self.semantic_judge_executed,
            "passed": self.passed,
            "severity": self.severity.value if self.severity else None,
            "quality_score": self.quality_score,
            "evaluated_dimensions": [x.value for x in self.evaluated_dimensions],
        }


# JSON helpers. They intentionally serialize only stable public fields used by evaluation.
def _iso(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _state_to_dict(state: CharacterStateSnapshot | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "emotion": state.emotion,
        "energy": state.energy,
        "trust": state.trust,
        "favorability": state.favorability,
        "relationship_stage": state.relationship_stage,
        "custom": copy.deepcopy(state.custom),
    }


def _state_from_dict(data: Mapping[str, Any] | None) -> CharacterStateSnapshot | None:
    return None if data is None else CharacterStateSnapshot(**dict(data))


def _source_to_dict(x: CognitiveEvalSource) -> dict[str, Any]:
    return {"source_type": x.source_type, "source_id": x.source_id, "content": x.content, "status": x.status, "metadata": dict(x.metadata)}


def _source_from_dict(x: Mapping[str, Any]) -> CognitiveEvalSource:
    return CognitiveEvalSource(**dict(x))


def _memory_to_dict(x: MemoryRecord) -> dict[str, Any]:
    return {
        "character_id": x.character_id, "summary": x.summary, "importance": x.importance, "kind": x.kind,
        "tags": list(x.tags), "source_event_id": x.source_event_id, "source_event_type": x.source_event_type,
        "metadata": copy.deepcopy(x.metadata), "status": x.status, "supersedes": list(x.supersedes),
        "superseded_by": x.superseded_by, "forgotten_at": None if x.forgotten_at is None else _iso(x.forgotten_at),
        "id": x.id, "created_at": _iso(x.created_at), "embedding": None if x.embedding is None else list(x.embedding),
        "embedding_metadata": copy.deepcopy(x.embedding_metadata), "vector_metadata": copy.deepcopy(x.vector_metadata),
    }


def _memory_from_dict(x: Mapping[str, Any]) -> MemoryRecord:
    d = dict(x)
    d["tags"] = tuple(d.get("tags", ()))
    d["supersedes"] = tuple(d.get("supersedes", ()))
    if d.get("forgotten_at") is not None: d["forgotten_at"] = _dt(d["forgotten_at"])
    if d.get("created_at") is not None: d["created_at"] = _dt(d["created_at"])
    if d.get("embedding") is not None: d["embedding"] = tuple(d["embedding"])
    return MemoryRecord(**d)


def _evidence_to_dict(x: CognitionEvidenceRef) -> dict[str, Any]:
    return {"source_type": x.source_type, "source_id": x.source_id, "evidence_type": x.evidence_type, "excerpt": x.excerpt, "confidence": x.confidence, "metadata": dict(x.metadata)}


def _evidence_from_dict(x: Mapping[str, Any]) -> CognitionEvidenceRef:
    return CognitionEvidenceRef(**dict(x))


def _claim_to_dict(x: BeliefClaim | None) -> dict[str, str] | None:
    return None if x is None else {"subject": x.subject, "predicate": x.predicate, "object": x.object}


def _reflection_to_dict(x: ReflectionRecord) -> dict[str, Any]:
    return {
        "character_id": x.character_id, "insight": x.insight, "confidence": x.confidence,
        "evidence": [_evidence_to_dict(v) for v in x.evidence], "claim": _claim_to_dict(x.claim),
        "base_revision": x.base_revision, "source_proposal_id": x.source_proposal_id, "source_task_id": x.source_task_id,
        "status": x.status.value, "metadata": dict(x.metadata), "id": x.id, "created_at": _iso(x.created_at),
    }


def _reflection_from_dict(x: Mapping[str, Any]) -> ReflectionRecord:
    d = dict(x); d["evidence"] = tuple(_evidence_from_dict(v) for v in d.get("evidence", ()))
    d["claim"] = None if d.get("claim") is None else BeliefClaim(**d["claim"]); d["status"] = ReflectionStatus(d["status"])
    d["created_at"] = _dt(d["created_at"])
    return ReflectionRecord(**d)


def _belief_to_dict(x: BeliefRecord) -> dict[str, Any]:
    return {
        "character_id": x.character_id, "claim": _claim_to_dict(x.claim), "confidence": x.confidence,
        "evidence": [_evidence_to_dict(v) for v in x.evidence], "source_reflection_ids": list(x.source_reflection_ids),
        "support_count": x.support_count, "status": x.status.value, "metadata": dict(x.metadata), "id": x.id,
        "created_at": _iso(x.created_at), "updated_at": _iso(x.updated_at),
    }


def _belief_from_dict(x: Mapping[str, Any]) -> BeliefRecord:
    d = dict(x); d["claim"] = BeliefClaim(**d["claim"]); d["evidence"] = tuple(_evidence_from_dict(v) for v in d.get("evidence", ()))
    d["source_reflection_ids"] = tuple(d.get("source_reflection_ids", ())); d["status"] = BeliefStatus(d["status"])
    d["created_at"] = _dt(d["created_at"]); d["updated_at"] = _dt(d["updated_at"])
    return BeliefRecord(**d)


def _goal_evidence_to_dict(x: GoalEvidenceRef) -> dict[str, Any]:
    return {"source_type": x.source_type, "source_id": x.source_id, "excerpt": x.excerpt, "metadata": dict(x.metadata)}


def _goal_evidence_from_dict(x: Mapping[str, Any]) -> GoalEvidenceRef:
    return GoalEvidenceRef(**dict(x))


def _signal_to_dict(x: MotivationSignal) -> dict[str, Any]:
    return {"kind": x.kind.value, "strength": x.strength, "evidence": _goal_evidence_to_dict(x.evidence), "rationale": x.rationale}


def _signal_from_dict(x: Mapping[str, Any]) -> MotivationSignal:
    d = dict(x); d["kind"] = MotivationKind(d["kind"]); d["evidence"] = _goal_evidence_from_dict(d["evidence"])
    return MotivationSignal(**d)


def _goal_to_dict(x: GoalRecord) -> dict[str, Any]:
    return {
        "character_id": x.character_id, "objective": x.objective, "horizon": x.horizon.value, "urgency": x.urgency,
        "confidence": x.confidence, "motivation_signals": [_signal_to_dict(v) for v in x.motivation_signals],
        "status": x.status.value, "conflict_key": x.conflict_key, "source_proposal_ids": list(x.source_proposal_ids),
        "source_task_ids": list(x.source_task_ids), "base_revisions": list(x.base_revisions), "metadata": dict(x.metadata),
        "id": x.id, "created_at": _iso(x.created_at), "updated_at": _iso(x.updated_at),
    }


def _goal_from_dict(x: Mapping[str, Any]) -> GoalRecord:
    d = dict(x); d["horizon"] = GoalHorizon(d["horizon"]); d["status"] = GoalStatus(d["status"])
    d["motivation_signals"] = tuple(_signal_from_dict(v) for v in d.get("motivation_signals", ()))
    for name in ("source_proposal_ids", "source_task_ids", "base_revisions"): d[name] = tuple(d.get(name, ()))
    d["created_at"] = _dt(d["created_at"]); d["updated_at"] = _dt(d["updated_at"])
    return GoalRecord(**d)


def _collaboration_to_dict(x: CollaborationResult) -> dict[str, Any]:
    return {
        "objective": x.objective,
        "plan": {
            "objective": x.plan.objective, "rationale": x.plan.rationale, "base_revision": x.plan.base_revision,
            "planner_model": x.plan.planner_model, "metadata": dict(x.plan.metadata),
            "work_items": [{"item_id": i.item_id, "specialist_id": i.specialist_id, "instruction": i.instruction, "source_ids": list(i.source_ids)} for i in x.plan.work_items],
        },
        "findings": [{
            "work_item_id": f.work_item_id, "specialist_id": f.specialist_id, "kind": f.kind.value, "status": f.status.value,
            "summary": f.summary, "confidence": f.confidence, "evidence_source_ids": list(f.evidence_source_ids),
            "recommendations": list(f.recommendations), "model": f.model, "error": f.error, "metadata": dict(f.metadata),
        } for f in x.findings],
        "verification": {
            "decision": x.verification.decision.value, "accepted_work_item_ids": list(x.verification.accepted_work_item_ids),
            "issues": [{"code": i.code, "message": i.message, "work_item_id": i.work_item_id} for i in x.verification.issues],
            "summary": x.verification.summary, "confidence": x.verification.confidence, "model": x.verification.model,
        },
        "base_revision": x.base_revision, "collaboration_id": x.collaboration_id,
    }


def _collaboration_from_dict(x: Mapping[str, Any]) -> CollaborationResult:
    d = dict(x); p = dict(d["plan"])
    items = tuple(CognitiveWorkItem(item_id=i["item_id"], specialist_id=i["specialist_id"], instruction=i["instruction"], source_ids=tuple(i.get("source_ids", ()))) for i in p.get("work_items", ()))
    plan = CognitiveWorkPlan(objective=p["objective"], rationale=p["rationale"], work_items=items, base_revision=p["base_revision"], planner_model=p.get("planner_model"), metadata=p.get("metadata", {}))
    findings = tuple(SpecialistFinding(work_item_id=f["work_item_id"], specialist_id=f["specialist_id"], kind=SpecialistKind(f["kind"]), status=SpecialistRunStatus(f["status"]), summary=f.get("summary"), confidence=f.get("confidence"), evidence_source_ids=tuple(f.get("evidence_source_ids", ())), recommendations=tuple(f.get("recommendations", ())), model=f.get("model"), error=f.get("error"), metadata=f.get("metadata", {})) for f in d.get("findings", ()))
    v = dict(d["verification"])
    verification = VerificationReport(decision=VerificationDecision(v["decision"]), accepted_work_item_ids=tuple(v.get("accepted_work_item_ids", ())), issues=tuple(VerificationIssue(**i) for i in v.get("issues", ())), summary=v["summary"], confidence=v.get("confidence"), model=v.get("model"))
    return CollaborationResult(objective=d["objective"], plan=plan, findings=findings, verification=verification, base_revision=d["base_revision"], collaboration_id=d["collaboration_id"])


def _frame_to_dict(x: CognitiveTimelineFrame) -> dict[str, Any]:
    return {"revision": x.revision, "beliefs": [_belief_to_dict(v) for v in x.beliefs], "goals": [_goal_to_dict(v) for v in x.goals]}


def _frame_from_dict(x: Mapping[str, Any]) -> CognitiveTimelineFrame:
    d = dict(x); return CognitiveTimelineFrame(revision=int(d["revision"]), beliefs=tuple(_belief_from_dict(v) for v in d.get("beliefs", ())), goals=tuple(_goal_from_dict(v) for v in d.get("goals", ())))
