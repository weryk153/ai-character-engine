"""A ready-made policy for how the character's own state moves.

The engine observes the user (``observed_user_emotion`` in custom state, written
by the commit coordinator) but deliberately does not decide what that does to
the character. Hosts that do not want to invent rules can use these.

The rules read two signed scores and never the emotion label. The label is free
text in whatever language the model chose; one model returned "疲倦" (tired), "joy" and
"curiosity" within the same conversation.

- ``valence``: how pleasant the user feels, -1 to 1.
- ``stance``: how the user treats the character, -1 hostile to 1 warm, and 0
  when they are not addressing the character's person.

They are separate because they mean different things. Someone who was shouted
at all day and comes to talk feels bad and is warm toward the character: that
should earn trust, not cost favorability.

Her mood is one of ai_character_engine.state.mood.CHARACTER_MOODS: sad when
the user turns on her, worried when the user feels bad, happy when the user is
warm, as strong as the user's emotion was; an unremarkable turn leaves it as it
was. A reading of her mood from both sides of the same turn (the companion's
mood worker) stands.

The default numbers are a starting point measured against eight-turn
conversations on a local 9B model, not a calibrated model of affection.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.state.mood import MOOD_TURN_KEY, seconds
from ai_character_engine.state.models import CharacterStateSnapshot, StatePatch
from ai_character_engine.tools.models import ToolResult

OBSERVATION_KEY = "observed_user_emotion"
# Leading underscore: bookkeeping, kept out of the model's context.
APPLIED_OBSERVATION_KEY = "_relationship_applied_observation"

# Lowest first. The stage follows the average of trust and favorability.
DEFAULT_STAGES: tuple[tuple[str, float], ...] = (
    ("stranger", 0.0),
    ("acquaintance", 58.0),
    ("friend", 68.0),
    ("close", 82.0),
)


@dataclass(frozen=True, slots=True)
class RelationshipRules:
    trust_per_turn: float = 0.3
    favorability_per_warmth: float = 4.0
    trust_per_warmth: float = 2.0
    # Hostility costs more trust than warmth earns: trust is slow to build.
    trust_per_hostility: float = 3.0
    trust_per_confiding: float = 1.0
    # Below this magnitude a score counts as no clear leaning. An unremarkable
    # turn measured between 0.0 and 0.2.
    notable: float = 0.3
    stages: tuple[tuple[str, float], ...] = DEFAULT_STAGES
    # A stage drops only this far below its threshold, otherwise a value that
    # hovers at a threshold changes the stage every turn.
    downgrade_margin: float = 3.0

    def __post_init__(self) -> None:
        if not self.stages or self.stages[0][1] != 0:
            raise ValueError("stages must start with a stage whose threshold is 0")
        thresholds = [threshold for _, threshold in self.stages]
        if thresholds != sorted(thresholds):
            raise ValueError("stages must be ordered from lowest to highest threshold")
        if not 0 <= self.notable <= 1:
            raise ValueError("notable must be between 0 and 1")
        if self.downgrade_margin < 0:
            raise ValueError("downgrade_margin must be >= 0")


def _score(value: Any) -> float | None:
    """A score between -1 and 1, or None when it cannot be read.

    Observations come from model output through JSON; bool is an int, so True
    would otherwise count as 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return max(-1.0, min(1.0, number))


def _unit(value: Any, fallback: float) -> float:
    score = _score(value)
    return fallback if score is None else max(0.0, score)


def _percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def _stage(bond: float, current: str, rules: RelationshipRules) -> str | None:
    names = [name for name, _ in rules.stages]
    current_index = names.index(current) if current in names else 0
    reached = max(index for index, (_, floor) in enumerate(rules.stages) if bond >= floor)
    if reached > current_index:
        return names[reached]
    if reached < current_index and bond < rules.stages[current_index][1] - rules.downgrade_margin:
        return names[reached]
    if current not in names:
        return names[reached]
    return None


def relationship_patch(
    state: CharacterStateSnapshot,
    *,
    count_turn: bool,
    rules: RelationshipRules | None = None,
    now: float | None = None,
) -> StatePatch | None:
    """The change to apply now, or None when there is nothing to change.

    ``count_turn=False`` is for reacting between turns: an observation is
    committed in the background, usually before the next turn starts, and
    reacting then lets the next reply already carry the new mood. That moment
    is not another turn, so it earns no per-turn trust.

    ``now`` dates a new mood, in seconds since the epoch; the system clock
    when not given.
    """
    rules = rules or RelationshipRules()
    trust_delta = rules.trust_per_turn if count_turn else 0.0
    favorability_delta = 0.0
    emotion = None
    intensity = 0.5
    custom_updates: dict[str, Any] = {}

    observation = state.custom.get(OBSERVATION_KEY)
    if isinstance(observation, dict):
        observation_id = str(observation.get("proposal_id") or "") or None
        if observation_id and observation_id != state.custom.get(APPLIED_OBSERVATION_KEY):
            # An observation without readable scores is consumed too, or it
            # would be inspected again on every turn.
            custom_updates[APPLIED_OBSERVATION_KEY] = observation_id
            stance = _score(observation.get("stance"))
            valence = _score(observation.get("valence"))
            intensity = _unit(observation.get("intensity"), 0.5)
            confidence = _unit(observation.get("confidence"), 0.5)
            if stance is not None and valence is not None:
                warmth = stance * intensity * confidence
                favorability_delta += rules.favorability_per_warmth * warmth
                trust_delta += (
                    rules.trust_per_warmth if warmth > 0 else rules.trust_per_hostility
                ) * warmth
                if stance <= -rules.notable:
                    emotion = "sad"
                elif valence <= -rules.notable:
                    emotion = "worried"
                    trust_delta += rules.trust_per_confiding * intensity
                elif stance >= rules.notable:
                    emotion = "happy"
                # Otherwise her mood stays as it was: setting one on every
                # unremarkable turn flipped her face between readings of it.
                turn = seconds(observation.get("turn_ended_at"))
                judged = seconds(state.custom.get(MOOD_TURN_KEY))
                if emotion is not None and turn is not None:
                    if judged is not None and turn <= judged:
                        # Her mood was read from both sides of this turn, or of
                        # a later one. The observation reads the user alone:
                        # what he did still moves trust, not her mood.
                        emotion = None
                    else:
                        custom_updates[MOOD_TURN_KEY] = turn

    bond = (
        _percent(state.trust + trust_delta) + _percent(state.favorability + favorability_delta)
    ) / 2
    patch = StatePatch(
        emotion=emotion,
        trust_delta=trust_delta,
        favorability_delta=favorability_delta,
        relationship_stage=_stage(bond, state.relationship_stage, rules),
        custom_updates=custom_updates,
        reason="relationship rules",
        mood_intensity=intensity if emotion is not None else None,
        mood_updated_at=(time.time() if now is None else now) if emotion is not None else None,
    )
    return None if patch.is_noop else patch


class RelationshipStatePolicy:
    """CharacterStatePolicy that applies the rules on every user turn."""

    def __init__(self, rules: RelationshipRules | None = None) -> None:
        self.rules = rules or RelationshipRules()
        # Where the time of a new mood comes from; CharacterCompanion sets its own.
        self.clock: Callable[[], float] = time.time

    def on_event(
        self, event: CharacterEvent, state: CharacterStateSnapshot
    ) -> StatePatch | None:
        if event.type not in {"user_message", "multimodal_user_message"}:
            return None
        return relationship_patch(state, count_turn=True, rules=self.rules, now=self.clock())

    def on_tool_result(
        self, result: ToolResult, state: CharacterStateSnapshot
    ) -> StatePatch | None:
        return None
