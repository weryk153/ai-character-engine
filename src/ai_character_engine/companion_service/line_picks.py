"""Her reply, line by line, to the faces and gestures a game shows.

As her reply streams in, the text is cut into lines (。！？!?…, a new line,
or a full stop before a space ends one; the rest when she has finished). Each line is picked in order, one
after the other (``ReplyActions``), and each pick is handed on as it is made:
a line that gets nothing is skipped, and picks may come after her reply has
ended.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

from ai_character_engine.companion import ReplyActions

ENDS = "。！？!?…\n"
CLOSERS = "」』）)]\"'”’"   # stay with the line they close
_WORDS = re.compile(r"\w")


def cut_lines(text: str) -> tuple[list[str], str]:
    """The finished lines of ``text`` and what is left of a line not yet finished."""
    lines, start, index = [], 0, 0
    while index < len(text):
        char = text[index]
        # A full stop ends a line only before a space: 3.5 is one number, and a
        # stop at the end of what has come so far may be one.
        if char in ENDS or (char == "." and text[index + 1 : index + 2].isspace()):
            end = index + 1
            while end < len(text) and (text[end] in ENDS or text[end] in CLOSERS):
                end += 1
            if end == len(text) and char != "\n":
                break  # more marks or a closing quote may be on their way
            lines.append(text[start:end])
            start = index = end
            continue
        index += 1
    return lines, text[start:]


class LinePicks:
    def __init__(self, actions: ReplyActions, hand_on: Callable[[dict], None]) -> None:
        self._actions = actions
        self._hand_on = hand_on
        self._rest = ""
        self._count = 0
        self._lines: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
        self.picked: list[dict] = []
        self.task = asyncio.create_task(self._pick_each())

    def feed(self, text: str) -> None:
        lines, self._rest = cut_lines(self._rest + text)
        for line in lines:
            self._queue(line)

    def finish(self) -> None:
        """Her reply has ended: what is left is the last line."""
        if self._rest:
            self._queue(self._rest)
            self._rest = ""
        self._lines.put_nowait(None)

    def _queue(self, line: str) -> None:
        line = line.strip()
        if _WORDS.search(line):  # a line of 」 or … alone has nothing to show
            self._lines.put_nowait((self._count, line))
            self._count += 1

    async def _pick_each(self) -> None:
        while (item := await self._lines.get()) is not None:
            index, line = item
            picked = await self._actions.pick(line)
            if picked is None or (picked.expression is None and picked.motion is None):
                continue
            message = {
                "index": index,
                "line": line,
                "expression": picked.expression,
                "motion": picked.motion,
                "intensity": picked.intensity,
                "voice": self._actions.voice(),
            }
            self.picked.append(message)
            self._hand_on(message)
