"""The face and the gesture of each line she says, picked by her background model.

The host knows what its avatar can do; the engine knows how she feels. The
host passes the expressions and motions of its avatar (``AvatarChoices``),
asks for a ``ReplyActions`` at the start of a reply, and asks it about each
line as the line is spoken: her background model reads the line, the one
before it and her mood, and picks from the host's lists.

A pick never holds her voice: ``pick`` does not raise, answers ``None`` when
the model is slow, failing, unsure or already busy with a line of this reply
(one line at a time, nothing queues), and the host decides how long it waits
for an answer. ``voice`` gives the tone of a line for a voice with one
reference per feeling: the first expression picked in this reply, or her mood
as the host's ``mood_faces`` shows it, or ``None``.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_character_engine.llm.models import Message

SYSTEM_PROMPT = """\
You direct an animated character's face and body while she speaks. For the line she is saying now, pick the facial expression that fits the feeling of that line, and a gesture only when the line clearly calls for one (a greeting, agreeing, pointing something out); most lines have no gesture. Use the previous line and her current mood only as context: the expression follows the line she is saying now. Pick only from the lists given. intensity is how strongly the expression shows, from 0 (barely) to 1 (fully).
Reply with JSON only, on one line without spaces:
{"expression":"<one of the expressions, or null>","motion":"<one of the motions, or null>","intensity":<0 to 1>}"""

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?")
_NOTHING = {"", "null", "none", "nil", "n/a", "-"}


@dataclass(frozen=True, slots=True)
class AvatarChoices:
    """What the host's avatar can do: its expressions, its motions (keyword to
    a short description, which may be empty), and which of its expressions
    shows each of her moods (``CHARACTER_MOODS``); moods left out have none."""

    expressions: Sequence[str] = ()
    motions: Mapping[str, str] = field(default_factory=dict)
    mood_faces: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expressions", tuple(str(name) for name in self.expressions))
        object.__setattr__(self, "motions", {str(key): str(label or "") for key, label in self.motions.items()})
        object.__setattr__(
            self, "mood_faces", {str(mood).lower(): str(face) for mood, face in self.mood_faces.items()}
        )


@dataclass(frozen=True, slots=True)
class LineActions:
    """What was picked for one line: an expression and a motion from the
    host's lists (``None`` where nothing fits), and how strongly the
    expression shows, between 0 and 1."""

    expression: str | None
    motion: str | None
    intensity: float = 1.0


def _first_object(raw: str) -> dict | None:
    """The first JSON object in a reply; a think block, ``` fences and a
    repeat after it are ignored."""
    body = _FENCE.sub("", _THINK.sub("", raw or "")).strip()
    start = body.find("{")
    if start < 0:
        return None
    try:
        value = json.JSONDecoder().raw_decode(body[start:])[0]
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _choice(value: Any, allowed: Sequence[str]) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().strip("[]").strip().lower()
    if key in _NOTHING:
        return None
    for name in allowed:
        if str(name).lower() == key:
            return str(name)
    return None


def _intensity(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(number):
        return 1.0
    return max(0.0, min(1.0, number))


def read_pick(raw: str, choices: AvatarChoices) -> LineActions | None:
    """A background model's answer as a pick: only what is on the host's
    lists is kept. ``None`` when the answer holds no JSON object."""
    data = _first_object(raw)
    if data is None:
        return None
    return LineActions(
        expression=_choice(data.get("expression"), choices.expressions),
        motion=_choice(data.get("motion"), list(choices.motions)),
        intensity=_intensity(data.get("intensity")),
    )


class ReplyActions:
    """The picks of one reply. Made by ``CharacterCompanion.reply_actions``."""

    def __init__(
        self,
        *,
        client: Any,
        choices: AvatarChoices,
        mood: Callable[[], tuple[str, float]],
        timeout_seconds: float,
    ) -> None:
        self.choices = choices
        self._client = client
        self._mood = mood
        self._timeout = timeout_seconds
        self._previous = ""
        self._asking = False
        self._first_expression: str | None = None

    @property
    def can_pick(self) -> bool:
        """Whether a line can get anything at all: a model, and something to pick."""
        return self._client is not None and bool(self.choices.expressions or self.choices.motions)

    def _messages(self, line: str, previous: str) -> list[Message]:
        try:
            mood, intensity = self._mood()
            feeling = f"{mood} ({intensity:.2f})"
        except Exception:
            feeling = "unknown"
        motions = ", ".join(
            f"{key} ({label})" if label else key for key, label in self.choices.motions.items()
        )
        user = "\n".join(
            [
                f"Expressions: {', '.join(self.choices.expressions) or '(none)'}",
                f"Motions: {motions or '(none)'}",
                f"Her mood: {feeling}",
                f"Previous line: {previous.strip() or '(none)'}",
                f"Line: {line.strip()}",
            ]
        )
        return [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)]

    async def pick(self, line: str) -> LineActions | None:
        """The expression and motion for ``line``, or ``None``: an empty line,
        nothing to pick from, no model, a model that is slow
        (``actions_timeout_seconds``), failing or answers no JSON, or a line of
        this reply already being picked. Never raises."""
        previous, self._previous = self._previous, line
        if not line.strip() or not self.can_pick or self._asking:
            return None
        self._asking = True
        try:
            response = await asyncio.wait_for(
                self._client.generate(self._messages(line, previous)), timeout=self._timeout
            )
            picked = read_pick(getattr(response, "text", "") or "", self.choices)
        except Exception:  # noqa: BLE001 - a pick that fails is a line without a face
            return None
        finally:
            self._asking = False
        if picked is not None and picked.expression and self._first_expression is None:
            self._first_expression = picked.expression
        return picked

    def voice(self) -> str | None:
        """The tone of the next line, as one of the host's expressions: the
        first expression picked in this reply, else her mood through
        ``mood_faces``, else ``None``."""
        if self._first_expression is not None:
            return self._first_expression
        try:
            mood, _ = self._mood()
        except Exception:
            return None
        return self.choices.mood_faces.get(str(mood).lower())
