from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


@dataclass(slots=True, frozen=True)
class CharacterStateSnapshot:
    """Immutable snapshot of a character's current mutable state."""

    emotion: str
    energy: float
    trust: float
    favorability: float
    relationship_stage: str
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class StatePatch:
    """A deterministic state transition requested by engine policy.

    Numeric relationship fields use deltas so that host applications can define
    rules without replacing the entire state object. Absolute fields such as
    emotion and relationship_stage are optional replacements.
    """

    emotion: str | None = None
    energy_delta: float = 0.0
    trust_delta: float = 0.0
    favorability_delta: float = 0.0
    relationship_stage: str | None = None
    custom_updates: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def is_noop(self) -> bool:
        return (
            self.emotion is None
            and self.energy_delta == 0
            and self.trust_delta == 0
            and self.favorability_delta == 0
            and self.relationship_stage is None
            and not self.custom_updates
        )


@dataclass(slots=True)
class CharacterState:
    """Mutable runtime state for one character session.

    Profile describes who the character *is*. State describes how the character
    *currently is*. Numeric fields are clamped to 0..100.
    """

    emotion: str = "neutral"
    energy: float = 100.0
    trust: float = 50.0
    favorability: float = 50.0
    relationship_stage: str = "stranger"
    custom: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.emotion = self._clean_label(self.emotion, field_name="emotion")
        self.relationship_stage = self._clean_label(
            self.relationship_stage,
            field_name="relationship_stage",
        )
        self.energy = _clamp_percent(self.energy)
        self.trust = _clamp_percent(self.trust)
        self.favorability = _clamp_percent(self.favorability)

    @staticmethod
    def _clean_label(value: str, *, field_name: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return cleaned

    def snapshot(self) -> CharacterStateSnapshot:
        return CharacterStateSnapshot(
            emotion=self.emotion,
            energy=self.energy,
            trust=self.trust,
            favorability=self.favorability,
            relationship_stage=self.relationship_stage,
            custom=dict(self.custom),
        )

    def restore(self, snapshot: CharacterStateSnapshot) -> CharacterStateSnapshot:
        """Restore an authoritative snapshot for transactional rollback paths."""

        self.emotion = self._clean_label(snapshot.emotion, field_name="emotion")
        self.relationship_stage = self._clean_label(
            snapshot.relationship_stage, field_name="relationship_stage"
        )
        self.energy = _clamp_percent(snapshot.energy)
        self.trust = _clamp_percent(snapshot.trust)
        self.favorability = _clamp_percent(snapshot.favorability)
        self.custom = dict(snapshot.custom)
        return self.snapshot()

    def apply(self, patch: StatePatch) -> CharacterStateSnapshot:
        if patch.emotion is not None:
            self.emotion = self._clean_label(patch.emotion, field_name="emotion")
        if patch.relationship_stage is not None:
            self.relationship_stage = self._clean_label(
                patch.relationship_stage,
                field_name="relationship_stage",
            )

        self.energy = _clamp_percent(self.energy + patch.energy_delta)
        self.trust = _clamp_percent(self.trust + patch.trust_delta)
        self.favorability = _clamp_percent(
            self.favorability + patch.favorability_delta
        )
        self.custom.update(patch.custom_updates)
        return self.snapshot()
