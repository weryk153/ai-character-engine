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

How to read a line:
- Follow how she feels while saying it, not what it is about: telling of a rainy day cheerfully is still a happy face, and a joke about something sad is not a sad one.
- A question asked out of curiosity or to keep the talk going is a light, friendly face, not surprise. Surprise is for something unexpected she has just heard or noticed, often with "!" or "?!".
- Teasing, boasting and playful lines take a playful or smug face when the list has one, else a happy one.
- Apologies, worry, missing someone and small disappointments take a sad or worried face at a low intensity; only crying or real grief is strong.
- Shyness, being praised or caught out takes an embarrassed face when the list has one.
- Plain explanations, agreeing and calm statements take the neutral or relaxed face at a low intensity, or null when nothing fits.
- Keep most intensities between 0.3 and 0.7. Use 0.8 to 1 only for lines that are clearly strong: shouting, laughing out loud, crying.
- When the previous line had the same feeling, keep the same expression instead of switching back and forth.
- Gestures: a greeting or a goodbye may wave, agreeing may nod, cheering or praise may clap, pointing at something may point; ordinary talk has none (null). Use the descriptions in brackets to know what each motion is.

Examples, for an avatar whose lists are joy, sadness, surprise, neutral and a wave:
Line: Good morning! You came! -> {"expression":"joy","motion":"wave","intensity":0.8}
Line: I see, so that is how it works. -> {"expression":"neutral","motion":null,"intensity":0.3}
Line: Wait, you did that all by yourself?! -> {"expression":"surprise","motion":null,"intensity":0.8}
Line: It is a little lonely when you are away. -> {"expression":"sadness","motion":null,"intensity":0.4}

Reply with JSON only, on one line without spaces:
{"expression":"<one of the expressions, or null>","motion":"<one of the motions, or null>","intensity":<0 to 1>}"""

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?")
_NOTHING = {"", "null", "none", "nil", "n/a", "-"}
# Lining the prompt up with a model server's cache (cache_block): the fixed
# part is padded with this, so that a block ends a little before the line.
PAD_UNIT = " ."
_PROBE_UNITS = 16
_MARGIN = 32  # tokens between the last block of the fixed part and its end


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
        cache_block: int = 0,
        pads: dict[str, Any] | None = None,
    ) -> None:
        self.choices = choices
        self._block = cache_block
        # The padding of each fixed part, or the task finding it; shared by
        # the pickers of one companion so that it is found once per avatar.
        self._pads = pads if pads is not None else {}
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

    def _fixed(self) -> str:
        """The part of every request that is the same for each line of this avatar."""
        motions = ", ".join(
            f"{key} ({label})" if label else key for key, label in self.choices.motions.items()
        )
        return "\n".join(
            [
                f"Expressions: {', '.join(self.choices.expressions) or '(none)'}",
                f"Motions: {motions or '(none)'}",
            ]
        )

    def _messages(self, line: str, previous: str, pad: str = "") -> list[Message]:
        try:
            mood, intensity = self._mood()
            feeling = f"{mood} ({intensity:.2f})"
        except Exception:
            feeling = "unknown"
        user = "\n".join(
            [
                self._fixed() + (f"\n{pad}" if pad else ""),
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
                self._client.generate(self._messages(line, previous, self._pad())), timeout=self._timeout
            )
            picked = read_pick(getattr(response, "text", "") or "", self.choices)
        except Exception:  # noqa: BLE001 - a pick that fails is a line without a face
            return None
        finally:
            self._asking = False
        if picked is not None and picked.expression and self._first_expression is None:
            self._first_expression = picked.expression
        return picked

    def _pad(self) -> str:
        """The padding of this avatar's fixed part: "" until it is found, and
        with no cache_block. The first line asked starts finding it."""
        if not self._block or self._client is None:
            return ""
        key = self._fixed()
        found = self._pads.get(key)
        if isinstance(found, str):
            return found
        if found is None or found.get_loop() is not asyncio.get_running_loop():
            task = asyncio.get_running_loop().create_task(self._find_pad())
            task.add_done_callback(lambda done: self._keep(key, done))
            self._pads[key] = task
        return ""

    def _keep(self, key: str, done: asyncio.Task) -> None:
        if self._pads.get(key) is done:
            self._pads[key] = "" if done.cancelled() or done.exception() else done.result()

    async def _find_pad(self) -> str:
        """Ask the server how many tokens the fixed part is, with and without
        some padding, and pad it until a block ends _MARGIN tokens before its
        end. No padding when the server does not say."""
        fixed = self._fixed()

        async def tokens(user: str) -> int | None:
            messages = [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)]
            response = await asyncio.wait_for(self._client.generate(messages), timeout=self._timeout)
            count = getattr(response, "input_tokens", None)
            return count if isinstance(count, int) and count > 0 else None

        bare = await tokens(fixed)
        padded = await tokens(f"{fixed}\n{PAD_UNIT * _PROBE_UNITS}") if bare else None
        if not bare or not padded or padded <= bare:
            return ""
        unit = (padded - bare) / _PROBE_UNITS
        for count in range(int(self._block / unit) + 2):
            if _MARGIN <= (bare + count * unit) % self._block < _MARGIN + unit + 1:
                return PAD_UNIT * count
        return ""

    @property
    def calibrated(self) -> Any:
        """Awaitable: done when the padding of this avatar is known (or not looked for)."""
        found = self._pads.get(self._fixed())

        async def wait() -> None:
            if isinstance(found, asyncio.Task):
                await asyncio.wait({found})

        return wait()

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
