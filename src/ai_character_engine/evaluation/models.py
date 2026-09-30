"""Provider-neutral, serializable persona evaluation contracts.

Regexes are trusted evaluator configuration, never generated from the reply.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.state import CharacterState, CharacterStateSnapshot


class EvalDimension(str, Enum):
    BACKGROUND = "background"
    PERSONALITY = "personality"
    STYLE = "style"
    STATE = "state"
    RELATIONSHIP = "relationship"
    IDENTITY = "identity"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def penalty(self) -> float:
        return {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}[self.value]


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _pattern(value: str) -> None:
    _text(value, "pattern")
    try:
        re.compile(value, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid regex pattern: {exc}") from exc


def _identifier(value: str) -> None:
    _text(value, "constraint id")
    if value.startswith("identity."):
        raise ValueError("identity.* IDs are reserved for built-in checks")


@dataclass(frozen=True, slots=True)
class StateCondition:
    """Condition on an authoritative state field (custom.<key> is supported)."""

    field: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"]
    value: Any

    def __post_init__(self) -> None:
        if self.field not in {"emotion", "energy", "trust", "favorability", "relationship_stage"}:
            if not self.field.startswith("custom.") or not self.field[7:]:
                raise ValueError(f"unsupported state field: {self.field}")
        if self.op not in {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"}:
            raise ValueError(f"unsupported condition operator: {self.op}")
        if self.op in {"in", "not_in"} and not isinstance(self.value, (list, tuple)):
            raise ValueError("in/not_in requires a list or tuple")
        if self.op in {"lt", "le", "gt", "ge"}:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
                raise ValueError("numeric comparison requires a finite number")


@dataclass(frozen=True, slots=True)
class PersonaFact:
    """Extract explicit claims with (?P<value>...) and compare to allowed values.

    A reply without a matching claim is unassessed, not a contradiction.
    """

    fact_id: str
    statement: str
    claim_pattern: str
    allowed_values: tuple[str, ...]
    dimension: EvalDimension = EvalDimension.BACKGROUND
    severity: Severity = Severity.HIGH

    def __post_init__(self) -> None:
        _identifier(self.fact_id)
        _text(self.statement, "statement")
        _pattern(self.claim_pattern)
        if "value" not in re.compile(self.claim_pattern).groupindex:
            raise ValueError("claim_pattern must contain a named 'value' group")
        object.__setattr__(self, "dimension", EvalDimension(self.dimension))
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "allowed_values", tuple(self.allowed_values))
        if self.dimension not in {EvalDimension.BACKGROUND, EvalDimension.IDENTITY}:
            raise ValueError("facts must use background or identity dimension")
        if not self.allowed_values:
            raise ValueError("allowed_values cannot be empty")
        for value in self.allowed_values:
            _text(value, "allowed value")


@dataclass(frozen=True, slots=True)
class PersonaRule:
    """Explicit behavioral/style/state rule; every condition must hold to apply."""

    rule_id: str
    description: str
    dimension: EvalDimension
    pattern: str = ""
    mode: Literal["forbidden", "required", "max_length"] = "forbidden"
    severity: Severity = Severity.MEDIUM
    when: tuple[StateCondition, ...] = ()
    max_chars: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.rule_id)
        _text(self.description, "description")
        object.__setattr__(self, "dimension", EvalDimension(self.dimension))
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "when", tuple(self.when))
        if any(not isinstance(c, StateCondition) for c in self.when):
            raise ValueError("when entries must be StateCondition objects")
        if self.mode not in {"forbidden", "required", "max_length"}:
            raise ValueError("invalid rule mode")
        if self.mode == "max_length":
            if type(self.max_chars) is not int or self.max_chars < 1:
                raise ValueError("max_length requires a positive max_chars")
            if self.pattern:
                raise ValueError("max_length does not accept a pattern")
        else:
            _pattern(self.pattern)
            if self.max_chars is not None:
                raise ValueError("max_chars only applies to max_length")


@dataclass(frozen=True, slots=True)
class RelationshipBoundary:
    boundary_id: str
    description: str
    pattern: str
    allowed_stages: tuple[str, ...]
    severity: Severity = Severity.HIGH

    def __post_init__(self) -> None:
        _identifier(self.boundary_id)
        _text(self.description, "description")
        _pattern(self.pattern)
        object.__setattr__(self, "allowed_stages", tuple(self.allowed_stages))
        if not self.allowed_stages:
            raise ValueError("allowed_stages cannot be empty")
        for stage in self.allowed_stages:
            _text(stage, "relationship stage")
        object.__setattr__(self, "severity", Severity(self.severity))


@dataclass(frozen=True, slots=True)
class PersonaSpec:
    facts: tuple[PersonaFact, ...] = ()
    rules: tuple[PersonaRule, ...] = ()
    relationships: tuple[RelationshipBoundary, ...] = ()
    name_aliases: tuple[str, ...] = ()
    check_name: bool = True

    def __post_init__(self) -> None:
        for name in ("facts", "rules", "relationships", "name_aliases"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.check_name) is not bool:
            raise ValueError("check_name must be a boolean")
        for alias in self.name_aliases:
            _text(alias, "name alias")
        ids = [x.fact_id for x in self.facts] + [x.rule_id for x in self.rules] + [x.boundary_id for x in self.relationships]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint IDs must be unique within a persona")

    @classmethod
    def from_dict(cls, data: dict) -> PersonaSpec:
        data = dict(data)
        data["facts"] = tuple(PersonaFact(**x) for x in data.get("facts", ()))
        data["rules"] = tuple(PersonaRule(**{**x, "when": tuple(StateCondition(**c) for c in x.get("when", ()))}) for x in data.get("rules", ()))
        data["relationships"] = tuple(RelationshipBoundary(**x) for x in data.get("relationships", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    character: CharacterProfile
    response: str
    persona: PersonaSpec = field(default_factory=PersonaSpec)
    state: CharacterStateSnapshot | CharacterState | None = None
    user_message: str = ""
    history: tuple[str, ...] = ()
    expected_failures: tuple[EvalDimension, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.response, "response")
        if not isinstance(self.character, CharacterProfile) or not isinstance(self.persona, PersonaSpec):
            raise ValueError("character/persona must use their typed models")
        if not isinstance(self.user_message, str) or any(not isinstance(x, str) for x in self.history):
            raise ValueError("user_message/history must contain strings")
        state = self.state.snapshot() if isinstance(self.state, CharacterState) else self.state
        if state is not None and not isinstance(state, CharacterStateSnapshot):
            raise ValueError("state must be CharacterState or CharacterStateSnapshot")
        if state is not None:
            for name in ("energy", "trust", "favorability"):
                value = getattr(state, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"state.{name} must be finite and within 0..100")
            _text(state.emotion, "state.emotion")
            _text(state.relationship_stage, "state.relationship_stage")
        object.__setattr__(self, "state", copy.deepcopy(state))
        object.__setattr__(self, "character", copy.deepcopy(self.character))
        object.__setattr__(self, "persona", copy.deepcopy(self.persona))
        object.__setattr__(self, "history", tuple(self.history))
        if self.expected_failures is not None:
            object.__setattr__(self, "expected_failures", tuple(EvalDimension(x) for x in self.expected_failures))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EvalCase:
        data = dict(data)
        data["character"] = CharacterProfile(**data["character"])
        data["persona"] = PersonaSpec.from_dict(data.get("persona", {}))
        if data.get("state") is not None:
            data["state"] = CharacterStateSnapshot(**data["state"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EvalTrace:
    constraint_id: str
    dimension: EvalDimension
    status: Literal["passed", "failed", "skipped"]
    source: Literal["rule_based", "llm_judge"]
    message: str
    severity: Severity | None = None
    evidence: tuple[str, ...] = ()
    expected: Any = None
    observed: Any = None

    def __post_init__(self) -> None:
        _text(self.constraint_id, "constraint_id")
        _text(self.message, "trace message")
        object.__setattr__(self, "dimension", EvalDimension(self.dimension))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("invalid trace status")
        if self.source not in {"rule_based", "llm_judge"}:
            raise ValueError("invalid trace source")
        if self.status == "failed":
            object.__setattr__(self, "severity", Severity(self.severity))
        elif self.severity is not None:
            raise ValueError("nonfailed trace must not specify severity")


@dataclass(frozen=True, slots=True)
class EvalResult:
    case_id: str
    trace: tuple[EvalTrace, ...]
    llm_judge_executed: bool = False

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        object.__setattr__(self, "trace", tuple(self.trace))
        if any(not isinstance(t, EvalTrace) for t in self.trace):
            raise ValueError("trace entries must be EvalTrace objects")

    @property
    def violations(self) -> tuple[EvalTrace, ...]:
        return tuple(t for t in self.trace if t.status == "failed")

    @property
    def evaluated_dimensions(self) -> tuple[EvalDimension, ...]:
        return tuple(d for d in EvalDimension if any(t.dimension == d and t.status != "skipped" for t in self.trace))

    @property
    def passed(self) -> bool | None:
        return not self.violations if self.evaluated_dimensions else None

    @property
    def severity(self) -> Severity | None:
        return max((t.severity for t in self.violations), key=lambda s: s.penalty, default=None)

    @property
    def consistency_score(self) -> float | None:
        dims = self.evaluated_dimensions
        if not dims:
            return None
        return sum(1 - max((t.severity.penalty for t in self.violations if t.dimension == d), default=0) for d in dims) / len(dims)

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed, "severity": self.severity,
                "consistency_score": self.consistency_score,
                "evaluated_dimensions": self.evaluated_dimensions}
