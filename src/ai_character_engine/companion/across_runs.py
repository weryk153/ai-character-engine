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
    """The memories of one ``meta_dir``, oldest first."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / FILE_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.memories: list[AcrossRunsMemory] = self._load()

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
                for memory in self.memories
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def add(self, text: str, *, tags, created_at: float, run: int | None) -> AcrossRunsMemory:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            raise ValueError("a memory across runs needs text")
        memory = AcrossRunsMemory(
            id=uuid4().hex,
            text=cleaned,
            tags=tuple(str(tag) for tag in tags),
            created_at=created_at,
            run=run,
        )
        self.memories.append(memory)
        self._write()
        return memory

    def forget(self, memory_id: str) -> bool:
        kept = [memory for memory in self.memories if memory.id != memory_id]
        if len(kept) == len(self.memories):
            return False
        self.memories = kept
        self._write()
        return True


def section(memories: list[AcrossRunsMemory], *, framing: str | None, shown: int) -> str | None:
    """For her system prompt: the newest ``shown``, oldest first, under the framing."""
    newest = memories[-shown:] if shown > 0 else []
    if not newest:
        return None
    lines = "\n".join(f"- {memory.text}" for memory in newest)
    return f"{(framing or DEFAULT_FRAMING).strip()}\n{lines}"
