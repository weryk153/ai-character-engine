"""The NPCs a service holds: one companion per save slot and NPC.

An NPC is opened by the game with its persona (``open``) and its companion is
built when first needed, closed after ``idle_close_seconds`` without use and
built again from its ``storage_dir`` when needed again. All of them share one
``ModelAccess``: background work of any NPC waits while any NPC replies.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import (
    AvatarChoices,
    CharacterCompanion,
    CompanionSnapshot,
    ModelAccess,
)
from ai_character_engine.companion.across_runs import AcrossRuns
from ai_character_engine.llm.local import OpenAICompatibleChatClient

from .config import ModelConfig, ServiceConfig

logger = logging.getLogger(__name__)

# Lowercase only: "Slot1" and "slot1" are one directory on macOS and Windows.
NAME = re.compile(r"[a-z0-9_-]{1,64}")


class ServiceError(Exception):
    status = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequest(ServiceError):
    status, code = 400, "invalid_request"


class Unauthorized(ServiceError):
    status, code = 401, "unauthorized"


class NpcNotOpen(ServiceError):
    status, code = 404, "npc_not_open"


def checked(name: str, what: str) -> str:
    if not NAME.fullmatch(name):
        raise InvalidRequest(f"{what} must be 1-64 lowercase letters, digits, '-' or '_'")
    return name


class GameClock:
    """The game's time for one NPC: the last ``game_time`` a request brought,
    standing still between requests; the system clock until one does."""

    def __init__(self) -> None:
        self.now: float | None = None

    def __call__(self) -> float:
        return time.time() if self.now is None else self.now

    def set(self, game_time: float | None) -> None:
        if game_time is not None:
            self.now = float(game_time)


def openai_client(model: ModelConfig) -> Any:
    return OpenAICompatibleChatClient(
        model=model.model,
        base_url=model.base_url,
        api_key=model.api_key,
        request_options=model.request_options(),
    )


@dataclass(slots=True)
class _Npc:
    character: CharacterProfile
    clock: GameClock = field(default_factory=GameClock)
    # What the game's avatar of her can do: her lines get faces and gestures
    # (line_picks). None: the game asked for none.
    avatar: AvatarChoices | None = None
    companion: CharacterCompanion | None = None
    used_at: float = field(default_factory=time.monotonic)
    # Held while she is replaced: two loads at once must not leave one of
    # the companions they built running unseen.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def take_character(self) -> None:
        """Called when a turn has her to itself: a persona the game sent while
        she was replying is hers from the next turn."""
        if self.companion is not None and self.companion.character != self.character:
            self.companion.character = self.character


async def _close_client(client: Any) -> None:
    """The HTTP connections of a model client: OpenAICompatibleChatClient
    keeps them in its AsyncOpenAI ``client``."""
    for owner in (client, getattr(client, "client", None)):
        close = getattr(owner, "aclose", None) or getattr(owner, "close", None)
        if owner is not None and callable(close):
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.warning("closing a model client failed", exc_info=True)
            return


class NpcRegistry:
    def __init__(
        self, config: ServiceConfig, *, make_llm: Callable[[ModelConfig], Any] = openai_client
    ) -> None:
        self.config = config
        self.settings = config.companion_settings()
        self.access = ModelAccess(self.settings.foreground_patience_seconds)
        self._foreground = make_llm(config.foreground)
        background = config.background
        if isinstance(background, ModelConfig):
            self._background: Any = make_llm(background)
        elif background is not None:
            self._background = {name: make_llm(model) for name, model in background.items()}
        else:
            self._background = make_llm(config.foreground.for_background())
        self._npcs: dict[tuple[str, str], _Npc] = {}
        self._listeners: dict[str, list[Callable[[str, CompanionSnapshot], None]]] = {}
        self._across_runs: dict[str, AcrossRuns] = {}

    # --- where things are -----------------------------------------------------------

    def storage_dir(self, slot: str, npc: str) -> Path:
        return self.config.data_dir / "slots" / checked(slot, "slot") / checked(npc, "npc")

    def meta_dir(self, npc: str) -> Path:
        return self.config.data_dir / "meta" / checked(npc, "npc")

    def across_runs(self, npc: str) -> AcrossRuns:
        if npc not in self._across_runs:
            self._across_runs[npc] = AcrossRuns(self.meta_dir(npc))
        return self._across_runs[npc]

    # --- NPCs -----------------------------------------------------------------------

    def open(
        self,
        slot: str,
        npc: str,
        character: CharacterProfile,
        game_time: float | None = None,
        avatar: AvatarChoices | None = None,
    ) -> bool:
        """Open an NPC in a slot, or give it a new persona (and avatar). True when it was not open."""
        self.storage_dir(slot, npc)  # checks both names
        entry = self._npcs.get((slot, npc))
        opened = entry is None
        if entry is None:
            entry = self._npcs[(slot, npc)] = _Npc(character)
        else:
            entry.character = character
            if entry.companion is not None and not entry.companion.busy:
                entry.take_character()
        entry.avatar = avatar
        entry.clock.set(game_time)
        return opened

    def entry(self, slot: str, npc: str) -> _Npc:
        self.storage_dir(slot, npc)
        entry = self._npcs.get((slot, npc))
        if entry is None:
            raise NpcNotOpen(f"{npc} is not open in slot {slot}: PUT it first")
        return entry

    def companion(self, slot: str, npc: str, game_time: float | None = None) -> CharacterCompanion:
        entry = self.entry(slot, npc)
        entry.clock.set(game_time)
        entry.used_at = time.monotonic()
        if entry.companion is None:
            entry.companion = self._build(slot, npc, entry)
        return entry.companion

    def _build(self, slot: str, npc: str, entry: _Npc) -> CharacterCompanion:
        companion = CharacterCompanion(
            character=entry.character,
            llm=self._foreground,
            background_llm=self._background,
            storage_dir=self.storage_dir(slot, npc),
            settings=self.settings,
            clock=entry.clock,
            meta_dir=self.meta_dir(npc),
            model_access=self.access,
        )
        companion.on_mood_change = lambda snapshot: self._tell(slot, npc, snapshot)
        return companion

    @staticmethod
    async def _retire(companion: CharacterCompanion) -> None:
        """Close her now: a line she is in the middle of is cut, not waited
        for (the game is loading or starting over)."""
        if companion.busy:
            companion.interrupt("")
        await companion.close()

    async def reopen(self, slot: str, npc: str, *, fresh: bool = False) -> CharacterCompanion:
        """A new companion for the NPC, before her first word: to load a save
        into (``fresh``: with her storage of this slot emptied)."""
        entry = self.entry(slot, npc)
        async with entry.lock:
            old, entry.companion = entry.companion, None
            if old is not None:
                await self._retire(old)
            if fresh:
                shutil.rmtree(self.storage_dir(slot, npc), ignore_errors=True)
            return self.companion(slot, npc)

    def npcs_of(self, slot: str) -> list[str]:
        checked(slot, "slot")
        return sorted(npc for (where, npc) in self._npcs if where == slot)

    async def delete_slot(self, slot: str) -> None:
        """A new game in this slot: its NPCs closed and forgotten, its data gone."""
        for npc in self.npcs_of(slot):
            entry = self._npcs.pop((slot, npc))
            old, entry.companion = entry.companion, None
            if old is not None:
                await self._retire(old)
        shutil.rmtree(self.config.data_dir / "slots" / slot, ignore_errors=True)

    async def close_idle(self, now: float | None = None) -> list[tuple[str, str]]:
        """Close the companions not used for ``idle_close_seconds``; their
        state is in storage, and they are built again when needed."""
        now = time.monotonic() if now is None else now
        closed = []
        # A copy: a game may open or drop NPCs while one is being closed.
        for key, entry in list(self._npcs.items()):
            companion = entry.companion
            if companion is None or companion.busy or companion.working:
                continue  # closing would cancel what she is in the middle of
            if now - entry.used_at >= self.config.idle_close_seconds:
                entry.companion = None
                try:
                    await companion.close()
                except Exception:
                    logger.exception("closing idle NPC %s of slot %s failed", key[1], key[0])
                closed.append(key)
        return closed

    async def close(self) -> None:
        for entry in list(self._npcs.values()):
            old, entry.companion = entry.companion, None
            if old is not None:
                await self._retire(old)
        clients = [self._foreground]
        if isinstance(self._background, dict):
            clients += self._background.values()
        elif self._background is not None:
            clients.append(self._background)
        for client in {id(client): client for client in clients}.values():
            await _close_client(client)

    # --- listeners ------------------------------------------------------------------

    def listen(self, slot: str, listener: Callable[[str, CompanionSnapshot], None]) -> Callable[[], None]:
        """``listener(npc, snapshot)`` when an NPC's mood changed between replies."""
        self._listeners.setdefault(slot, []).append(listener)
        return lambda: self._listeners.get(slot, []).remove(listener)

    def _tell(self, slot: str, npc: str, snapshot: CompanionSnapshot) -> None:
        for listener in tuple(self._listeners.get(slot, ())):
            listener(npc, snapshot)
