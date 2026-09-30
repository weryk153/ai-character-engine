"""CharacterCompanion: the engine assembled the way a host needs it.

The engine's parts each do one thing: CharacterRuntime runs a turn, the host
bridge streams and interrupts it, BackgroundCognitionRuntime thinks afterwards,
the commit coordinator decides what may change the character. Put together in a
real voice application on a local model, the defaults did not survive contact:
background results were stale before they could be committed, background calls
slowed the reply, memory landed in the wrong conversation, and nothing ever
changed the character's own state. This class is that assembly, with the
policies that were needed.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.cognition import (
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
)
from ai_character_engine.commit import CognitiveCommitCoordinator, CommitStatus
from ai_character_engine.commit.models import StalePolicy
from ai_character_engine.context.builder import ContextBuilder, one_line
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals import GoalManager
from ai_character_engine.goals.models import GoalRecord
from ai_character_engine.goals.store import InMemoryGoalStore, JsonlGoalStore
from ai_character_engine.host import CharacterHostBridge, HostBridgeConfig, HostBridgeError
from ai_character_engine.llm import ModelEndpoint
from ai_character_engine.llm.models import Message
from ai_character_engine.long_term_cognition import LongTermCognitionManager
from ai_character_engine.long_term_cognition.store import (
    InMemoryLongTermCognitionStore,
    JsonlLongTermCognitionStore,
)
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.memory.retriever import MemoryRetriever
from ai_character_engine.memory.trace import RetrievalResult
from ai_character_engine.memory.store import JsonlMemoryStore
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.session.serialization import state_from_dict, state_to_dict
from ai_character_engine.state.models import CharacterState
from ai_character_engine.state.policy import CharacterStatePolicy
from ai_character_engine.state.relationship import RelationshipStatePolicy, relationship_patch
from ai_character_engine.tasks import MultiTaskRuntime, MultiTaskRuntimeConfig, TaskPriority
from ai_character_engine.tasks.errors import UnknownTaskError
from ai_character_engine.tools.registry import ToolRegistry
from ai_character_engine.vision import VisionFrame, VisionPipeline

from .access import ModelAccess, PoliteClient
from .settings import CompanionSettings

logger = logging.getLogger(__name__)

MEMORY_TARGET = "memory.append_candidate"
# These two write into memory, and memory belongs to a conversation.
CONVERSATION_SCOPED_TARGETS = (MEMORY_TARGET, "memory.conversation_summary_candidate")
# Memory jobs can queue up while the user talks faster than the model works.
# Each holds a slot of the task runtime while it waits; without room to spare
# the emotion job of the newest turn would wait behind them for a slot.
SPARE_TASK_SLOTS = 8


class CompanionClosed(RuntimeError):
    """The companion was retired or closed. A turn that waited for the one
    before it never began; a host that replaced the companion asks its
    successor."""


class TurnInterrupted(HostBridgeError):
    """interrupt() ended the reply while it was being generated."""


@dataclass(frozen=True, slots=True)
class CompanionSnapshot:
    emotion: str
    trust: float
    favorability: float
    relationship_stage: str
    goals: tuple[str, ...] = ()
    thoughts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Worker:
    name: str
    kind: BackgroundCognitionKind
    role: CognitiveRole
    priority: TaskPriority
    target: str


# Most urgent first; the order is also the order in which they get the model.
_WORKERS = (
    _Worker(
        "emotion",
        BackgroundCognitionKind.EMOTION_ANALYSIS,
        CognitiveRole.EMOTION,
        TaskPriority.HIGH,
        "state.emotion_candidate",
    ),
    _Worker(
        "memory",
        BackgroundCognitionKind.MEMORY_EXTRACTION,
        CognitiveRole.MEMORY,
        TaskPriority.HIGH,
        MEMORY_TARGET,
    ),
    _Worker(
        "goal",
        BackgroundCognitionKind.GOAL_MOTIVATION,
        CognitiveRole.GOAL,
        TaskPriority.NORMAL,
        "cognition.goal_candidate",
    ),
    _Worker(
        "reflection",
        BackgroundCognitionKind.REFLECTION,
        CognitiveRole.REFLECTION,
        TaskPriority.LOW,
        "cognition.reflection_candidate",
    ),
    _Worker(
        "summary",
        BackgroundCognitionKind.CONVERSATION_SUMMARY,
        CognitiveRole.SUMMARY,
        TaskPriority.LOW,
        "memory.conversation_summary_candidate",
    ),
)


_TARGET_OF = {worker.kind: worker.target for worker in _WORKERS}


@dataclass(slots=True)
class _Turn:
    """A reply that was asked for and has not ended."""

    conversation_id: str | None
    text: str
    # The host's own name for the turn, to tell two turns of one conversation apart.
    turn_id: Any = None
    # Set by interrupt(): what the user had heard by then.
    heard: str | None = None
    generating: bool = False


_ANY = object()
# After every worker: the workers are ranked by their place in _WORKERS.
_HOST_RANK = 1_000


def _interrupted(heard: str) -> Message:
    return Message("assistant", f"{heard.strip()} [Interrupted by user]".strip())


def _cut(history: list[Message], reply: Message, heard: str) -> bool:
    """Replace that very reply, wherever it is, with what was heard of it."""
    for index, message in enumerate(history):
        if message is reply:
            history[index] = _interrupted(heard)
            return True
    return False


def _under_way(handle) -> bool:
    try:
        return not handle.status.terminal
    except UnknownTaskError:
        return False  # finished so long ago that the task runtime let it go


class _RecentEvents:
    """EventLedger that keeps the newest events. The ledger of the engine keeps
    every event with the state before and after it, for audit; nothing in a
    companion reads further back than the last few."""

    def __init__(self, kept: int) -> None:
        self._entries: deque = deque(maxlen=kept)

    def append(self, entry) -> None:
        self._entries.append(entry)

    def list_for_character(self, character_id: str) -> list:
        return [entry for entry in self._entries if entry.character_id == character_id]


class _GoalsInMind(GoalManager):
    """The goals she still has: none left untouched for too long. The store
    keeps them all; the goal worker reads the store. How many of them she
    keeps in mind is the context builder's ``goals_shown``: a goal pushed out
    by a more pressing one is not given up, only not in mind."""

    def __init__(self, *, store, max_age_days: int) -> None:
        super().__init__(store=store)
        self._max_age = timedelta(days=max_age_days)

    def active_goals(self, *, character_id: str) -> tuple[GoalRecord, ...]:
        cutoff = datetime.now(UTC) - self._max_age
        return tuple(
            goal
            for goal in super().active_goals(character_id=character_id)
            if goal.updated_at >= cutoff
        )


class _AllOfTheConversation(MemoryRetriever):
    """What the newest message is about first, then everything else she
    remembers of the conversation, what matters most first.

    The default retriever takes what shares words with the newest message. A
    conversation has tens of memories, not thousands; leaving out the ones
    that share no word with "hello" makes her forget for no gain. Each is
    written into the conversation once.
    """

    def retrieve_with_trace(self, *, character_id, query, limit=5, now=None):
        found = super().retrieve_with_trace(
            character_id=character_id, query=query, limit=limit, now=now
        )
        chosen = list(found.memories)
        taken = {item.record.id for item in chosen}
        rest = [
            record
            for record in self.store.list_for_character(character_id)
            if record.character_id == character_id
            and record.is_active
            and record.id not in taken
        ]
        rest.sort(key=lambda record: (record.importance, record.created_at), reverse=True)
        chosen += [RetrievedMemory(record=record, score=0.0) for record in rest]
        chosen = chosen[: max(0, limit)]
        return RetrievalResult(
            tuple(chosen),
            replace(
                found.trace, selected_memory_ids=tuple(item.record.id for item in chosen)
            ),
        )


class _BackgroundExtractionOnly:
    """MemoryWritePolicy: the foreground writes no memory.

    By default the foreground stores "User said: <the words>" for every turn,
    and the commit coordinator then refuses what the extraction worker found
    in the same turn as a conflict. The weaker record won.
    """

    def importance(self, *, event, response, state_before, state_after):
        return None


def _clean(messages: Sequence[Message]) -> list[Message]:
    """User and assistant lines only, no consecutive lines from the same side."""
    cleaned: list[Message] = []
    for message in messages:
        if message.role not in ("user", "assistant") or not message.content.strip():
            continue
        if cleaned and cleaned[-1].role == message.role:
            continue
        cleaned.append(message)
    return cleaned


class CharacterCompanion:
    def __init__(
        self,
        *,
        character: CharacterProfile,
        llm: Any,
        background_llm: Any | Mapping[str, Any] | None = None,
        storage_dir: str | Path | None = None,
        settings: CompanionSettings | None = None,
        context_builder: ContextBuilder | None = None,
        tool_registry: ToolRegistry | None = None,
        vision: VisionPipeline | None = None,
        state_policy: CharacterStatePolicy | None = None,
        bridge_config: HostBridgeConfig | None = None,
    ) -> None:
        """``background_llm`` is the model for background cognition: one client
        for every worker, or a mapping from worker name (emotion, memory, goal,
        reflection, summary) to a client; a worker without a client does not
        run. It defaults to ``llm``. Background workers must return JSON, so a
        client with a low temperature serves them better than the one tuned for
        conversation.

        Without ``storage_dir`` everything is kept in memory only.
        """
        self.settings = settings or CompanionSettings()
        self._character = character
        self._notes: tuple[str, ...] = ()
        self._dir = Path(storage_dir) if storage_dir is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        # None is a conversation too: the one of a host that names none.
        self._active: str | None = None
        self._started = False
        # Per conversation: what was said, and the notes written into it.
        self._kept: OrderedDict[str | None, tuple[list[Message], list]] = OrderedDict()
        self._loaded: dict[str | None, list[Message]] = {}
        self._turn_lock = asyncio.Lock()
        self._pending: set[asyncio.Task[None]] = set()
        self._closed = False
        self._released = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turns: list[_Turn] = []
        self._unfinished: tuple[str | None, str] | None = None
        # Per conversation: the lines the host passed as what it knows.
        self._told: dict[str | None, frozenset[str]] = {}
        # Per conversation: the turn that made its newest reply, and the reply.
        self._newest_reply: dict[str | None, tuple[Any, Message]] = {}
        # Per conversation: the serial of its last turn, whatever came of it,
        # and of the turn that made the newest reply. A turn that failed or
        # was kept out of memory leaves the reply before it in place, and
        # that reply was heard whole: it is not the one to take back.
        self._serial = itertools.count()
        self._last_turn: dict[str | None, int] = {}
        self._newest_turn: dict[str | None, int] = {}
        # A reply of the conversation at hand, interrupted while the next reply
        # was being generated: cut once that turn has let go of the history.
        self._cut_later: tuple[Message, str] | None = None
        self._access = ModelAccess(self.settings.foreground_patience_seconds)
        # target -> the turn after which a job for it was last scheduled
        self._scheduled_at: dict[str, int] = {}
        self._live_by_kind: dict[BackgroundCognitionKind, list] = {}
        self._conversation_by_task: dict[str, str | None] = {}

        builder = context_builder or ContextBuilder()
        builder.goals_shown = self.settings.goals_shown
        notes_of_the_host = builder.turn_notes
        builder.turn_notes = lambda: [
            *(notes_of_the_host() if notes_of_the_host is not None else ()),
            *self._thoughts_for_context(),
            # Notes stay in the conversation; an instruction for one reply
            # must not read as a standing one. What the host knows is no
            # instruction.
            *(
                note if note.startswith("- ") else f"For the next reply only: {note}"
                for note in self._notes
            ),
        ]
        # Kept here as well: a turn kept out of memory takes the memory
        # manager away from the runtime for its duration, and a host's memory
        # page is open meanwhile.
        memory = self._memory_store = self._store(
            JsonlMemoryStore, InMemoryMemoryStore, "memory.jsonl"
        )
        self.runtime = CharacterRuntime(
            character=character,
            llm=llm,
            context_builder=builder,
            tool_registry=tool_registry,
            state=self._load_state(),
            state_policy=state_policy or RelationshipStatePolicy(),
            memory_manager=MemoryManager(
                store=memory,
                ledger=_RecentEvents(self.settings.records_kept),
                retriever=_AllOfTheConversation(memory),
                retrieval_limit=self.settings.memories_recalled,
                write_policy=_BackgroundExtractionOnly(),
            ),
            goal_manager=_GoalsInMind(
                store=self._store(JsonlGoalStore, InMemoryGoalStore, "goals.jsonl"),
                max_age_days=self.settings.goal_max_age_days,
            ),
            long_term_cognition=LongTermCognitionManager(
                store=self._store(
                    JsonlLongTermCognitionStore, InMemoryLongTermCognitionStore, "cognition.jsonl"
                )
            ),
            max_history_messages=self.settings.max_history_messages,
            memory_scope_id=character.id,
            cognition_scope_id=character.id,
            goal_scope_id=character.id,
        )
        self.runtime.stream_text_with_tools = self.settings.stream_text_with_tools
        self._bridge = CharacterHostBridge(self.runtime, vision=vision, config=bridge_config)

        clients = self._background_clients(llm if background_llm is None else background_llm)
        # Who gets the model is decided by ModelAccess, not by the number of
        # slots. With a single slot, a job waiting to be redone after the
        # character spoke would hold up everything behind it.
        self._tasks = MultiTaskRuntime(
            self.runtime,
            config=MultiTaskRuntimeConfig(
                worker_count=len(clients) + SPARE_TASK_SLOTS,
                lifecycle_history=self.settings.records_kept,
            ),
        )
        self._background = self._build_background(clients)
        if self._background is not None:
            self._background.output_language = self.settings.language
        self._commits = CognitiveCommitCoordinator(
            self._tasks, event_history=self.settings.records_kept
        )
        for target, policy in tuple(self._commits.policies.items()):
            # The defaults ask for a rerun whenever a turn happened in between.
            # A local model needs 30 s or more for the background work of one
            # turn, so nearly nothing would ever be committed.
            self._commits.policies[target] = replace(
                policy, stale_policy=StalePolicy.ALLOW_MANUAL_REBASE
            )

    # --- assembly --------------------------------------------------------------

    def _store(self, persistent, in_memory, name: str):
        return persistent(self._dir / name) if self._dir is not None else in_memory()

    def _state_file(self) -> Path | None:
        return self._dir / "state.json" if self._dir is not None else None

    def _load_state(self) -> CharacterState:
        state = CharacterState()
        path = self._state_file()
        if path is not None and path.is_file():
            try:
                state.restore(state_from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception as exc:
                # A damaged file must not keep the character from starting.
                logger.warning("state file unreadable, starting fresh: %s", exc)
        return state

    def _save_state(self) -> None:
        path = self._state_file()
        if path is None or self._closed:
            return
        payload = json.dumps(
            state_to_dict(self.runtime.state.snapshot()), ensure_ascii=False, indent=2
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def _background_clients(self, source: Any) -> list[tuple[_Worker, Any]]:
        clients = []
        for worker in _WORKERS:
            if int(getattr(self.settings, f"{worker.name}_every")) <= 0:
                continue
            client = source.get(worker.name) if isinstance(source, Mapping) else source
            if client is not None:
                clients.append((worker, client))
        return clients

    def _build_background(self, clients) -> BackgroundCognitionRuntime | None:
        if not clients:
            return None
        specs, endpoints, policies = [], [], {}
        for rank, (worker, client) in enumerate(clients):
            every = int(getattr(self.settings, f"{worker.name}_every"))
            specs.append(
                BackgroundWorkerSpec(
                    worker.kind,
                    priority=worker.priority,
                    # The task timeout also covers waiting for the character to
                    # finish and for a turn at the model; the call itself is
                    # timed by ModelAccess.
                    timeout_s=self.settings.call_timeout_seconds
                    + self.settings.foreground_patience_seconds,
                    every_n_revisions=every,
                )
            )
            endpoint_id = f"companion-{worker.name}"
            endpoints.append(
                ModelEndpoint(
                    endpoint_id=endpoint_id,
                    client=PoliteClient(
                        client,
                        self._access,
                        rank=rank,
                        timeout=self.settings.call_timeout_seconds,
                        own_lane=worker.name == "emotion",
                    ),
                )
            )
            policies[worker.role] = CognitiveRolePolicy(primary_endpoint_ids=(endpoint_id,))
        return BackgroundCognitionRuntime(
            self._tasks,
            CognitiveModelRuntime(
                endpoints=tuple(endpoints), router=CognitiveModelRouter(policies=policies)
            ),
            config=BackgroundCognitionConfig(
                worker_specs=tuple(specs), event_history=self.settings.records_kept
            ),
        )

    def _thoughts_for_context(self) -> list[str]:
        thoughts = self.snapshot().thoughts
        return ["\n".join(f"- thought: {thought}" for thought in thoughts)] * bool(thoughts)

    @property
    def character(self) -> CharacterProfile:
        return self._character

    @character.setter
    def character(self, profile: CharacterProfile) -> None:
        """Rewrite the character between turns. The id names the stored state,
        memory and goals, so it cannot change."""
        if profile.id != self._character.id:
            raise ValueError("a companion keeps the character id it was created with")
        if self._bridge.busy:
            raise HostBridgeError("Cannot rewrite the character during a turn.")
        self._character = profile
        self.runtime.character = profile

    # --- conversations ---------------------------------------------------------

    def _scope(self, conversation_id: str | None) -> str:
        """Memory follows the conversation; state, goals and thoughts belong to
        the character."""
        if not conversation_id:
            return self.character.id
        return f"{self.character.id}:{conversation_id}"

    def has_conversation(self, conversation_id: str | None) -> bool:
        """Whether the companion holds this conversation, seen or loaded."""
        return (
            (self._started and conversation_id == self._active)
            or conversation_id in self._kept
            or conversation_id in self._loaded
        )

    def load_conversation(
        self, conversation_id: str | None, messages: Sequence[Message]
    ) -> None:
        """Give the companion a conversation the host has on record.

        Used only if the companion has not seen that conversation itself. What
        it has seen is newer: a page reload loads the record again, and using
        it would erase the turns made since.
        """
        seen = self._started and conversation_id == self._active
        if not seen and conversation_id not in self._kept:
            self._loaded[conversation_id] = _clean(messages)

    def _switch_to(self, conversation_id: str | None) -> None:
        """Call with the turn lock held."""
        if conversation_id != self._active or not self._started:
            # Taken out before room is made: it may be the oldest one kept.
            history, notes = self._kept.pop(conversation_id, (None, []))
            if self._started:
                self._kept[self._active] = (
                    list(self.runtime.history),
                    list(self.runtime.context_notes),
                )
                while len(self._kept) > self.settings.conversations_kept:
                    forgotten, _ = self._kept.popitem(last=False)
                    self._told.pop(forgotten, None)
                    self._newest_reply.pop(forgotten, None)
                    self._last_turn.pop(forgotten, None)
                    self._newest_turn.pop(forgotten, None)
            if history is None:
                history = self._loaded.pop(conversation_id, [])
            self._bridge.restore_history(history)
            self.runtime.context_notes = notes
            self._active = conversation_id
            self._started = True
        self.runtime.memory_scope_id = self._scope(conversation_id)

    # --- the turn --------------------------------------------------------------

    @property
    def tools(self) -> ToolRegistry:
        return self.runtime.tool_registry

    @property
    def busy(self) -> bool:
        return self._bridge.busy

    @property
    def sees(self) -> bool:
        """Whether reply() takes pictures: a vision pipeline was given."""
        return self._bridge.vision is not None

    async def reply(
        self,
        text: str,
        *,
        conversation_id: str | None = None,
        frames: tuple[VisionFrame, ...] = (),
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
        skip_memory: bool = False,
        proactive: bool = False,
        notes: Sequence[str] = (),
        before_turn: Callable[[], None] | None = None,
        remember_as: Callable[[str], str] | None = None,
        turn_id: Any = None,
    ) -> CharacterRunResult:
        """``notes`` are instructions of the host for this reply only.

        A note that starts with "- " is something the host knows about the
        character or the user, one line each. It is written into the
        conversation once, however often it is passed, and taken out again
        when the host no longer passes it: pass what is known on every turn.

        ``before_turn`` is called when the turn has the companion to itself,
        which is where a host shared by several callers rewrites the character
        or registers its tools. ``remember_as`` turns the reply into what the
        conversation keeps of it, for a host that shows replies normalized.
        ``turn_id`` is the host's own name for this turn, for interrupt().
        """
        if self._closed:
            raise CompanionClosed("companion is closed")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        turn = _Turn(conversation_id, text, turn_id)
        self._turns.append(turn)
        try:
            async with self._turn_lock:
                if self._closed:
                    # Retired while this turn waited for the one before it.
                    raise CompanionClosed("companion is closed")
                self._switch_to(conversation_id)
                if turn.heard is not None:
                    # Interrupted before she began to answer. It is still the
                    # conversation's last turn: the reply before it was heard.
                    self._last_turn[conversation_id] = next(self._serial)
                    self._bridge.record_interrupted_turn(text, turn.heard)
                    raise TurnInterrupted("The reply was interrupted.")
                self._unfinished = None
                if before_turn is not None:
                    before_turn()
                serial = next(self._serial)
                self._last_turn[conversation_id] = serial
                self._take_back_what_the_host_no_longer_knows(conversation_id, notes)
                self._notes = tuple(note for note in notes if note.strip())
                self._access.foreground_started()
                try:
                    turn.generating = True
                    try:
                        result = await self._tasks.run_foreground_turn(
                            lambda: self._bridge.process(
                                text,
                                frames=frames,
                                skip_memory=skip_memory,
                                proactive=proactive,
                                on_text_delta=on_text_delta,
                            )
                        )
                    except asyncio.CancelledError:
                        if turn.heard is None:
                            # The host cancelled its own task. It may tell us
                            # later how much was heard; see interrupt().
                            self._unfinished = (conversation_id, text)
                            raise
                        self._bridge.record_interrupted_turn(text, turn.heard)
                        self._save_state()
                        raise TurnInterrupted("The reply was interrupted.") from None
                    finally:
                        turn.generating = False
                        self._cut_what_waited()
                    if remember_as is not None and not skip_memory:
                        self._bridge.replace_reply(remember_as(result.text))
                    if not skip_memory and self.runtime.history:
                        self._newest_reply[conversation_id] = (
                            turn_id,
                            self.runtime.history[-1],
                        )
                        self._newest_turn[conversation_id] = serial
                    handles: tuple = ()
                    if self._background is not None and not skip_memory and not self._closed:
                        handles = await self._background.schedule_after_foreground(result)
                    self._take_on(handles)
                finally:
                    self._notes = ()
                    # Not before the new jobs are taken on: work that waited for
                    # the reply to end wakes up here and must find the hold for
                    # this turn's mood in place.
                    self._access.foreground_finished()
                self._save_state()
        finally:
            self._turns.remove(turn)
        for handle in handles:
            self._conversation_by_task[handle.task_id] = conversation_id
            task = asyncio.create_task(self._collect(handle))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        return result

    def _take_back_what_the_host_no_longer_knows(
        self, conversation_id: str | None, notes: Sequence[str]
    ) -> None:
        known = frozenset(
            line for note in notes if note.startswith("- ") for line in note.splitlines()
        )
        taken_back = self._told.get(conversation_id, frozenset()) - known
        if taken_back:
            self.runtime.withdraw_from_notes(taken_back.__contains__)
        self._told[conversation_id] = known

    def replace_reply(self, text: str) -> None:
        """Keep the newest reply the way the host displayed it."""
        history = self.runtime.history
        before = history[-1] if history else None
        self._bridge.replace_reply(text)
        newest = self._newest_reply.get(self._active)
        if newest and before is not None and newest[1] is before and history[-1] is not before:
            self._newest_reply[self._active] = (newest[0], history[-1])

    async def remember_remark(self, conversation_id: str | None, remark: str) -> None:
        """Keep a remark she made on her own in the conversation.

        A host that prompts her to speak up often sends a long instruction it
        does not want kept, and makes that turn with ``skip_memory``. The remark
        was then forgotten with the instruction: she repeated her remarks, and
        when the user answered one she said it again. This keeps what she
        actually said, after a short event saying she spoke up on her own, and
        makes it the conversation's newest reply (``take_back`` and ``interrupt``
        treat it like one). Waits for a reply under way.
        """
        remark = remark.strip()
        if not remark:
            return
        async with self._turn_lock:
            if self._closed:
                raise CompanionClosed("companion is closed")
            self._switch_to(conversation_id)
            event = self.runtime.context_builder.event_to_message(
                CharacterEvent(
                    type="proactive_remark",
                    source="host",
                    content="The user had been quiet for a while; you spoke up on your own.",
                )
            )
            reply = Message("assistant", remark)
            self.runtime.history.extend([event, reply])
            serial = next(self._serial)
            self._last_turn[conversation_id] = serial
            self._newest_turn[conversation_id] = serial
            self._newest_reply[conversation_id] = (None, reply)
            self._save_state()

    def take_back(self, conversation_id: str | None = None) -> bool:
        """The host did not use the newest reply of the conversation.

        A host may throw a reply away, one that repeats what she just said,
        and ask again with a hint. The reply and the user's words it answered
        leave the conversation, so that the exchange does not stand twice,
        once with a reply nobody heard. Only the newest exchange, and only
        while its reply is still the newest message; returns whether anything
        was taken back. What the background took from the exchange stays.
        """
        newest = self._newest_reply.get(conversation_id)
        if newest is None:
            return False
        if self._newest_turn.get(conversation_id) != self._last_turn.get(conversation_id):
            # A later turn failed or was kept out of memory: the newest reply
            # is from before it, and was heard whole.
            return False
        if conversation_id == self._active:
            if self._bridge.busy:
                return False
            history, notes = self.runtime.history, self.runtime.context_notes
        elif conversation_id in self._kept:
            history, notes = self._kept[conversation_id]
        else:
            return False
        if not history or history[-1] is not newest[1]:
            return False
        gone = []
        # Back to what the reply answered: the user's words, or the event a
        # remark of her own was made for.
        while history and history[-1].role not in ("user", "event"):
            gone.append(history.pop())
        if history:
            gone.append(history.pop())
        left = [(anchor, note) for anchor, note in notes if not any(anchor is m for m in gone)]
        if conversation_id == self._active:
            self.runtime.context_notes = left
        else:
            self._kept[conversation_id] = (history, left)
        self._newest_reply.pop(conversation_id, None)
        self._newest_turn.pop(conversation_id, None)
        self._save_state()
        return True

    def interrupt(
        self,
        heard_response: str,
        *,
        conversation_id: Any = _ANY,
        turn_id: Any = None,
    ) -> None:
        """The user interrupted. ``heard_response`` is what was actually played.

        Works at any of the moments a host can be in: before she began to
        answer, while the reply is being generated, after the host cancelled
        its own task, or during playback of a finished reply.

        A host with one listener needs no more. A host that serves several at
        once says whom it means: the conversation, and the name it gave the
        turn when two turns of one conversation can be under way. Without
        either, the interruption is for the reply being generated, because
        that is the one that can be heard, and otherwise for the turn asked
        for last.
        """

        def meant(conversation: str | None) -> bool:
            return conversation_id is _ANY or conversation == conversation_id

        turns = [
            turn
            for turn in self._turns
            if meant(turn.conversation_id) and (turn_id is None or turn.turn_id == turn_id)
        ]
        if turns:
            turn = next((turn for turn in turns if turn.generating), turns[-1])
            turn.heard = heard_response
            if turn.generating:
                self._bridge.cancel()
            return
        if conversation_id is not _ANY and conversation_id != self._active:
            # Its reply was played while another conversation went on. Only the
            # reply that the named turn made: a remark kept out of memory is
            # not in the conversation, and the reply before it was heard whole.
            kept = self._kept.get(conversation_id)
            newest = self._newest_reply.get(conversation_id)
            if kept and newest and (turn_id is None or newest[0] == turn_id):
                history, _ = kept
                if _cut(history, newest[1], heard_response):
                    self._newest_reply.pop(conversation_id, None)
            return
        if self._bridge.busy:
            # The next reply is being generated. What is meant here has ended:
            # either in another conversation, or in this one, played in a
            # second window. The turn under way holds the history until it
            # ends; the cut waits for it.
            newest = self._newest_reply.get(self._active)
            if meant(self._active) and newest and (turn_id is None or newest[0] == turn_id):
                self._cut_later = (newest[1], heard_response)
                self._newest_reply.pop(self._active, None)
            return
        if self._unfinished is not None and meant(self._unfinished[0]):
            conversation, text = self._unfinished
            self._unfinished = None
            if conversation == self._active:
                self._bridge.record_interrupted_turn(text, heard_response)
        elif meant(self._active):
            newest = self._newest_reply.get(self._active)
            if turn_id is None or (newest and newest[0] == turn_id):
                self._bridge.interrupt(heard_response)
        self._save_state()

    def _cut_what_waited(self) -> None:
        if self._cut_later is None:
            return
        reply, cut = self._cut_later
        self._cut_later = None
        _cut(self.runtime.history, reply, cut)

    # --- background results ----------------------------------------------------

    def _take_on(self, handles) -> None:
        """Make room for the jobs of the turn that just ended.

        A newer job of the same kind replaces an older one, queued or running.
        When the user speaks faster than the background can think, the older
        result would only give way to the newer one (see _superseded) after
        using the model. Memory is the exception: every turn has facts of its
        own.

        The mood is read before anything else uses the model.
        """
        if not handles or self._background is None:
            return
        kind_of = {
            event.task_id: event.kind
            for event in self._background.events()
            if event.action == "scheduled" and event.task_id
        }
        for handle in handles:
            kind = kind_of.get(handle.task_id)
            if kind is None:
                continue
            self._scheduled_at[_TARGET_OF[kind]] = self._tasks.revision
            if kind is BackgroundCognitionKind.EMOTION_ANALYSIS:
                self._access.hold(handle.task_id, self.settings.call_timeout_seconds)
            live = self._live_by_kind.setdefault(kind, [])
            live[:] = [item for item in live if _under_way(item)]
            if kind is not BackgroundCognitionKind.MEMORY_EXTRACTION:
                for older in live:
                    older.cancel()
                live.clear()
            live.append(handle)

    def _superseded(self, proposal) -> bool:
        """Whether a newer job of the same kind was scheduled after this one.

        The coordinator accepts one emotion observation or summary per
        revision. An older result moved onto the current revision takes that
        place, and the result that belongs there is refused as a conflict.
        """
        if proposal.target == MEMORY_TARGET:
            return False
        # Not every turn schedules work: a remark of her own does not.
        return self._scheduled_at.get(proposal.target, -1) > proposal.base_revision

    async def _collect(self, handle) -> None:
        try:
            try:
                result = await handle.wait()
            finally:
                self._access.release(handle.task_id)
            conversation_id = self._conversation_by_task.pop(handle.task_id, None)
            if result.output is None:
                logger.debug("%s %s: %s", result.task_type, result.status.value, result.error)
                return
            # The same lock as a turn: the conversation must not change under
            # a commit.
            async with self._turn_lock:
                if self._closed:
                    return
                outcomes = [
                    await self._commit(proposal, conversation_id)
                    for proposal in result.output.proposals
                ]
                if any(o.committed and o.target == "state.emotion_candidate" for o in outcomes):
                    await self._react_to_observation()
                if any(o.committed for o in outcomes):
                    self._save_state()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("background result dropped (%s: %s)", type(exc).__name__, exc)
        finally:
            self._conversation_by_task.pop(handle.task_id, None)

    async def _react_to_observation(self) -> None:
        rules = getattr(self.runtime.state_policy, "rules", None)
        if not isinstance(self.runtime.state_policy, RelationshipStatePolicy):
            return
        async with self._tasks.authority_guard():
            patch = relationship_patch(
                self.runtime.state.snapshot(), count_turn=False, rules=rules
            )
            if patch is not None:
                self.runtime.state.apply(patch)

    async def _commit(self, proposal, conversation_id: str | None):
        """Call with the turn lock held.

        The coordinator writes memory into the scope that is current when it
        commits. A result that arrives after the user moved to another
        conversation would land there, so the scope is set back to the
        conversation the job came from for the duration of the commit.
        """
        current_scope = self.runtime.memory_scope_id
        if proposal.target in CONVERSATION_SCOPED_TARGETS:
            self.runtime.memory_scope_id = self._scope(conversation_id)
        try:
            outcome = await self._commits.commit(proposal)
            turns_late = self._tasks.revision - proposal.base_revision
            if (
                outcome.status is CommitStatus.STALE
                and outcome.reason == "foreground_revision_changed"
                and turns_late > 0
                and (
                    proposal.target == MEMORY_TARGET
                    or turns_late <= self.settings.max_turns_late
                )
                and not self._superseded(proposal)
            ):
                rebased = self._commits.rebase(
                    proposal,
                    reason=f"{turns_late} turn(s) old and still valid evidence",
                )
                outcome = await self._commits.commit(rebased)
        finally:
            self.runtime.memory_scope_id = current_scope
        if not outcome.committed:
            logger.debug(
                "%s from turn %s %s at turn %s: %s %s",
                proposal.target,
                proposal.base_revision,
                outcome.status.value,
                self._tasks.revision,
                outcome.reason,
                dict(outcome.metadata) or "",
            )
        return outcome

    # --- lifecycle -------------------------------------------------------------

    async def settle(self) -> None:
        """Wait for background work and commits that are under way."""
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)

    async def aside(self, make_call: Callable[[], Awaitable[Any]], *, timeout: float | None = None) -> Any:
        """Run a model call of the host's own the way her background work runs.

        A host with a single local model has calls of its own beside hers: a
        memory of its own to tidy, a translation to make. Made here, the call
        waits while she replies, gives way to a reply that starts once, and
        takes its turn after her workers. Raises ``TimeoutError`` after
        ``timeout`` seconds of the call itself (``call_timeout_seconds`` when
        not given), and ``CompanionClosed`` once she is closed.
        """
        if self._closed:
            raise CompanionClosed("The companion is closed.")
        # Her workers of the turn that just ended come first. They are queued
        # only once the reply has ended, so waiting for the model alone would
        # let this call in ahead of them. Bounded: a user who keeps talking
        # keeps the workers busy.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.foreground_patience_seconds
        while (self._access.foreground_active or self._pending) and loop.time() < deadline:
            if self._access.foreground_active:
                try:
                    await asyncio.wait_for(self._access.quiet(), deadline - loop.time())
                except TimeoutError:
                    break
                continue
            await asyncio.wait(tuple(self._pending), timeout=deadline - loop.time())
        if self._closed:
            raise CompanionClosed("The companion is closed.")

        async def unless_closed():
            if self._closed:
                raise CompanionClosed("The companion is closed.")
            return await make_call()

        return await self._access.call(
            _HOST_RANK,
            unless_closed,
            self.settings.call_timeout_seconds if timeout is None else timeout,
        )

    def flush(self) -> None:
        """Save state; the companion keeps running."""
        self._save_state()

    def usable_in_running_loop(self) -> bool:
        """Whether this companion can take a turn where the caller is running.

        Its locks and workers belong to the event loop of its first turn. A
        host that keeps companions across restarts of its loop asks before
        reusing one.
        """
        if self._closed:
            return False
        if self._loop is None:
            return True
        if self._loop.is_closed():
            return False
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return True

    def retire(self) -> None:
        """Stop at once, from synchronous code: state is saved, no further turn
        is taken and nothing under way is committed. A reply she is in the
        middle of is finished, so that the host's turn ends the way it would
        have. close() still has to be awaited to release the workers."""
        if self._closed:
            return
        self._save_state()
        self._closed = True
        for task in tuple(self._pending):
            task.cancel()
        for handles in self._live_by_kind.values():
            for handle in handles:
                if _under_way(handle):
                    handle.cancel()

    async def close(self) -> None:
        if self._released:
            return
        self._released = True
        self.retire()
        await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        await self._tasks.close()
        await self._bridge.close()

    # --- reading ---------------------------------------------------------------

    def snapshot(self) -> CompanionSnapshot:
        state = self.runtime.state
        newest_first = sorted(
            self.runtime.long_term_cognition.reflections(character_id=self.character.id),
            key=lambda record: record.created_at,
            reverse=True,
        )
        goals = [
            goal.objective
            for goal in self.runtime.goal_manager.active_goals(character_id=self.character.id)
        ][: self.settings.goals_shown]
        thoughts = [record.insight for record in newest_first[: self.settings.thoughts_shown]]
        return CompanionSnapshot(
            emotion=state.emotion,
            trust=state.trust,
            favorability=state.favorability,
            relationship_stage=state.relationship_stage,
            goals=tuple(goals),
            thoughts=tuple(thoughts),
        )

    def rewrite_memories(
        self,
        conversation_id: str | None,
        summaries: Sequence[str],
        *,
        edited_from: Sequence[str] | None = None,
    ) -> None:
        """What she remembers of a conversation, as the host's user edited it.

        Lines that are still there stay as they are, lines that are gone are
        forgotten, new lines are remembered as something the user stated.

        ``edited_from`` is what the user was shown: only a line that was shown
        and is gone is forgotten. A memory that arrived while the page was
        open was never removed by anyone.
        """
        scope = self._scope(conversation_id)
        store = self._memory_store
        wanted = list(dict.fromkeys(one_line(line) for line in summaries if line.strip()))
        removable = None if edited_from is None else {one_line(line) for line in edited_from}
        records = []
        for record in store.list_for_character(scope):
            shown = one_line(record.summary)
            if record.is_active and shown not in wanted and (removable is None or shown in removable):
                record = replace(record, status="forgotten", forgotten_at=datetime.now(UTC))
            records.append(record)
        remembered = {one_line(record.summary) for record in records if record.is_active}
        records += [
            MemoryRecord(
                character_id=scope,
                summary=summary,
                kind="fact",
                importance=0.7,
                metadata={"evidence_type": "asserted_fact", "source": "host"},
            )
            for summary in wanted
            if summary not in remembered
        ]
        store.replace_for_character(scope, records)

    def memories(self, conversation_id: str | None = None) -> list[str]:
        records = self._memory_store.list_for_character(self._scope(conversation_id))
        return [one_line(record.summary) for record in records if record.is_active]
