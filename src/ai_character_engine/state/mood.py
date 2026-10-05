"""Her mood: a small fixed vocabulary and how strongly she feels it.

The engine decides which moods a character has, not any host's table of
expressions; a host maps these words onto faces of its own.
"""

from __future__ import annotations

import math

# In this order wherever they are listed: prompts, documentation, hosts.
CHARACTER_MOODS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "embarrassed",
    "calm",
    "worried",
)
NEUTRAL = "neutral"
# The strength of a mood set without one, as by a host's own state policy.
DEFAULT_MOOD_INTENSITY = 0.5
# Custom state: when the turn ended whose reading set her mood, in seconds
# since the epoch. A reading of an earlier turn that arrives later does not
# replace it. Leading underscore: bookkeeping, kept out of the model's context.
MOOD_TURN_KEY = "_mood_turn_ended_at"


def mood_intensity(mood: str, value: object) -> float:
    """``value`` as an intensity between 0 and 1: 0 for neutral and for what
    is not a number. Saved state and model output both reach here."""
    if mood == NEUTRAL or isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def seconds(value: object) -> float | None:
    """A time in seconds since the epoch, or None when it cannot be read."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
