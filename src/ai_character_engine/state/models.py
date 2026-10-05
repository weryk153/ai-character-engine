from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .mood import DEFAULT_MOOD_INTENSITY, mood_intensity, seconds


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
    # How strongly she feels her mood (``emotion``), 0..1, as of
    # ``mood_updated_at`` in seconds since the epoch. It fades with time; see
    # ai_character_engine.state.mood.
    mood_intensity: float = 0.0
    mood_updated_at: float | None = None


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
    # With ``emotion``: how strongly she feels it and since when. Left out, a
    # new mood has DEFAULT_MOOD_INTENSITY as of the moment it is applied.
    mood_intensity: float | None = None
    mood_updated_at: float | None = None

    @property
    def is_noop(self) -> bool:
        return (
            self.emotion is None
            and self.energy_delta == 0
            and self.trust_delta == 0
            and self.favorability_delta == 0
            and self.relationship_stage is None
            and not self.custom_updates
            and self.mood_intensity is None
            and self.mood_updated_at is None
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
    mood_intensity: float = 0.0
    mood_updated_at: float | None = None

    def __post_init__(self) -> None:
        self.emotion = self._clean_label(self.emotion, field_name="emotion")
        self.relationship_stage = self._clean_label(
            self.relationship_stage,
            field_name="relationship_stage",
        )
        self.energy = _clamp_percent(self.energy)
        self.trust = _clamp_percent(self.trust)
        self.favorability = _clamp_percent(self.favorability)
        self.mood_intensity = mood_intensity(self.emotion, self.mood_intensity)
        self.mood_updated_at = seconds(self.mood_updated_at)

    @staticmethod
    def _clean_label(value: str, *, field_name: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return cleaned

    def _stamp_mood(self, mood_updated_at: float | None) -> None:
        stamped = seconds(mood_updated_at)
        self.mood_updated_at = time.time() if stamped is None else stamped

    def snapshot(self) -> CharacterStateSnapshot:
        return CharacterStateSnapshot(
            emotion=self.emotion,
            energy=self.energy,
            trust=self.trust,
            favorability=self.favorability,
            relationship_stage=self.relationship_stage,
            custom=dict(self.custom),
            mood_intensity=self.mood_intensity,
            mood_updated_at=self.mood_updated_at,
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
        self.mood_intensity = mood_intensity(self.emotion, snapshot.mood_intensity)
        self.mood_updated_at = seconds(snapshot.mood_updated_at)
        return self.snapshot()

    def apply(self, patch: StatePatch) -> CharacterStateSnapshot:
        if patch.emotion is not None:
            self.emotion = self._clean_label(patch.emotion, field_name="emotion")
            self.mood_intensity = mood_intensity(
                self.emotion,
                DEFAULT_MOOD_INTENSITY if patch.mood_intensity is None else patch.mood_intensity,
            )
            self._stamp_mood(patch.mood_updated_at)
        elif patch.mood_intensity is not None:
            self.mood_intensity = mood_intensity(self.emotion, patch.mood_intensity)
            self._stamp_mood(patch.mood_updated_at)
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
