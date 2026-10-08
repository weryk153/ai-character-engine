"""What she remembers from before the world began again.

A game that restarts its world (a new game, another run) may let her keep
something of the runs before. Only the game writes these: no background
worker reads her conversations into them. They live in ``meta_dir``, apart
from her state, and saves neither hold nor replace them.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_FRAMING = "From before this world began again, you remember:"
FILE_NAME = "across_runs.jsonl"


@dataclass(frozen=True, slots=True)
class AcrossRunsMemory:
    """``created_at``: seconds since the epoch, her clock; ``run``: which run
    it was written in, when the game says."""

    id: str
    text: str
    tags: tuple[str, ...] = ()
    created_at: float = 0.0
    run: int | None = None


class AcrossRuns:
    """The memories of one ``meta_dir``, oldest first. Read again when the
    file changed since: another companion of hers may have written it."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / FILE_NAME
        self._seen: tuple[int, int, int, int] | None = None
        self._memories: list[AcrossRunsMemory] = []
        self._refresh()

    @property
    def memories(self) -> list[AcrossRunsMemory]:
        self._refresh()
        return self._memories

    def _stamp(self) -> tuple[int, int, int, int] | None:
        """Every write moves a new file into place: its inode changes even
        where the clock is too coarse for the time to, and its change time
        moves even where the file system hands the freed inode number to the
        next new file (ext4). Nothing can set the change time back."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

    def _refresh(self) -> None:
        stamp = self._stamp()
        if stamp != self._seen:
            self._memories = self._load()
            self._seen = stamp

    def _load(self) -> list[AcrossRunsMemory]:
        if not self.path.is_file():
            return []
        memories = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                run = raw.get("run")
                memories.append(
                    AcrossRunsMemory(
                        id=str(raw["id"]),
                        text=str(raw["text"]),
                        tags=tuple(str(tag) for tag in raw.get("tags") or ()),
                        created_at=float(raw.get("created_at") or 0.0),
                        run=run if isinstance(run, int) and not isinstance(run, bool) else None,
                    )
                )
            except Exception as exc:
                # A damaged line must not cost her the others.
                logger.warning("memory across runs unreadable, skipped: %s", exc)
        return memories

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(
                    {
                        "id": memory.id,
                        "text": memory.text,
                        "tags": list(memory.tags),
                        "created_at": memory.created_at,
                        "run": memory.run,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for memory in self._memories
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self._seen = self._stamp()

    def add(self, text: str, *, tags, created_at: float, run: int | None) -> AcrossRunsMemory:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            raise ValueError("a memory across runs needs text")
        memory = AcrossRunsMemory(
            id=uuid4().hex,
            text=cleaned,
            # One tag given alone is one tag, not its letters.
            tags=(tags,) if isinstance(tags, str) else tuple(str(tag) for tag in tags),
            created_at=created_at,
            run=run,
        )
        self._refresh()
        self._memories.append(memory)
        try:
            self._write()
        except BaseException:
            self._memories.pop()  # not kept on disk: not kept
            raise
        return memory

    def forget(self, memory_id: str) -> bool:
        kept = [memory for memory in self.memories if memory.id != memory_id]
        if len(kept) == len(self._memories):
            return False
        before, self._memories = self._memories, kept
        try:
            self._write()
        except BaseException:
            self._memories = before
            raise
        return True


def section(memories: list[AcrossRunsMemory], *, framing: str | None, shown: int) -> str | None:
    """For her system prompt: the newest ``shown``, oldest first, under the framing."""
    newest = memories[-shown:] if shown > 0 else []
    if not newest:
        return None
    lines = "\n".join(f"- {memory.text}" for memory in newest)
    return f"{(framing or DEFAULT_FRAMING).strip()}\n{lines}"
