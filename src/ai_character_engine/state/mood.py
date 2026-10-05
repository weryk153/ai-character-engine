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
# Near words a model answers with instead of one of the eight, lower case.
# Matched whole, never fuzzily: anything else is no reading of her mood.
MOOD_SYNONYMS: dict[str, str] = {
    **dict.fromkeys(
        ("joyful", "glad", "pleased", "excited", "cheerful", "amused", "delighted"), "happy"
    ),
    **dict.fromkeys(("relieved", "relaxed", "content", "peaceful", "serene", "at ease"), "calm"),
    **dict.fromkeys(("upset", "hurt", "lonely", "disappointed", "sorrowful", "down"), "sad"),
    **dict.fromkeys(("annoyed", "irritated", "frustrated", "mad", "furious"), "angry"),
    **dict.fromkeys(
        ("anxious", "concerned", "nervous", "uneasy", "afraid", "scared"), "worried"
    ),
    **dict.fromkeys(("shy", "flustered", "bashful", "awkward"), "embarrassed"),
    **dict.fromkeys(("shocked", "startled", "astonished", "amazed"), "surprised"),
}
# The strength of a mood set without one, as by a host's own state policy.
DEFAULT_MOOD_INTENSITY = 0.5
# How her mood fades unless a host says otherwise (CompanionSettings): the
# intensity halves every DEFAULT_MOOD_HALF_LIFE_SECONDS, and below
# DEFAULT_MOOD_FLOOR she is neutral again. Every reader of her mood uses these.
DEFAULT_MOOD_HALF_LIFE_SECONDS = 300.0
DEFAULT_MOOD_FLOOR = 0.15
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


def effective_mood(
    mood: str,
    intensity: float,
    updated_at: float | None,
    *,
    now: float,
    half_life_seconds: float = DEFAULT_MOOD_HALF_LIFE_SECONDS,
    floor: float = DEFAULT_MOOD_FLOOR,
) -> tuple[str, float]:
    """Her mood as it stands at ``now``: the intensity halves every
    ``half_life_seconds``, and below ``floor`` she is neutral again.

    Worked out when read, never stored: nothing runs while nobody talks to
    her. A time ahead of ``now`` (another machine's clock) counts as no time
    at all, and a mood of unknown time is as of now.
    """
    strength = mood_intensity(mood, intensity)
    if strength <= 0:
        return NEUTRAL, 0.0
    elapsed = 0.0 if updated_at is None else max(0.0, now - updated_at)
    faded = strength * 0.5 ** (elapsed / half_life_seconds)
    if faded < floor:
        return NEUTRAL, 0.0
    return mood, faded


def blend_mood(
    mood: str, intensity: float, new_mood: str, new_intensity: object
) -> tuple[str, float] | None:
    """What a new reading of her mood makes of it: the mood and intensity to
    store as of now, or None to leave her mood exactly as stored.

    ``mood``/``intensity`` are her mood as it stands now (effective_mood).
    A mood holds until something at least as strong comes along, so one
    reading does not wipe out the last: a neutral reading lets her mood fade
    by itself, the same mood again is as strong as the stronger of the two,
    and another mood takes over only when it is at least as strong.
    """
    strength = mood_intensity(new_mood, new_intensity)
    if strength <= 0:
        return None
    current = mood_intensity(mood, intensity)
    if new_mood == mood:
        return mood, max(strength, current)
    if strength >= current:
        return new_mood, strength
    return None
