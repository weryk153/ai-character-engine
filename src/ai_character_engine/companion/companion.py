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
import copy
import inspect
import itertools
import json
import logging
import os
import re
import time
import unicodedata
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from datetime import UTC, datetime, timedelta
from datetime import date as Date
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
from ai_character_engine.cognition.background import (
    DIARY_TARGET,
    MEMORY_CONFLICT_TARGET,
    MOOD_TARGET,
    REPLY_NOTE_TARGET,
    SELF_MEMORY_TARGET,
    USER_STATE_TARGET,
    said_in,
    script_of,
    script_of_language,
)
from ai_character_engine.commit import CognitiveCommitCoordinator, CommitStatus
from ai_character_engine.commit.models import StalePolicy
from ai_character_engine.context.builder import (
    SELF_MEMORY_LINE,
    USER_LATELY_LINE,
    ContextBuilder,
    one_line,
)
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals import GoalManager
from ai_character_engine.goals.models import GoalHorizon, GoalRecord
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
from ai_character_engine.memory.conflicts import (
    CONFLICTS_LOG,
    resolve_conflict,
    take_unasked_conflicts,
    unresolved_conflicts,
)
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.memory.self_kinds import (
    CONVERSATION_KEY,
    self_memory_kind,
    stays_in_conversation,
)
from ai_character_engine.memory.retriever import MemoryRetriever
from ai_character_engine.memory.trace import RetrievalResult
from ai_character_engine.memory.store import JsonlMemoryStore
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.session.serialization import state_from_dict, state_to_dict
from ai_character_engine.state.models import CharacterState
from ai_character_engine.state.mood import (
    DEFAULT_MOOD_FLOOR,
    DEFAULT_MOOD_HALF_LIFE_SECONDS,
    NEUTRAL,
    effective_mood,
)
from ai_character_engine.state.policy import CharacterStatePolicy
from ai_character_engine.state.relationship import RelationshipStatePolicy
from ai_character_engine.tasks import MultiTaskRuntime, MultiTaskRuntimeConfig, TaskPriority
from ai_character_engine.tasks.errors import UnknownTaskError
from ai_character_engine.tools.registry import ToolRegistry
from ai_character_engine.vision import VisionFrame, VisionPipeline

from . import save_state
from .access import ModelAccess, PoliteClient
from .across_runs import AcrossRuns, AcrossRunsMemory
from .across_runs import section as across_runs_section
from .save_state import StateBusy, StateFormatError
from .settings import CompanionSettings

logger = logging.getLogger(__name__)

MEMORY_TARGET = "memory.append_candidate"
# These write into memory, and memory belongs to a conversation.
CONVERSATION_SCOPED_TARGETS = (
    MEMORY_TARGET,
    "memory.conversation_summary_candidate",
    MEMORY_CONFLICT_TARGET,
)
# Each job of these reads lines no other job reads, and what was said stays
# said: a newer job does not replace it, and its result is used however late.
_READ_ONCE = (
    BackgroundCognitionKind.MEMORY_EXTRACTION,
    BackgroundCognitionKind.SELF_MEMORY_EXTRACTION,
)
# A conflict is checked against the memories as they are when it is
# committed: a turn in between does not make it wrong.
_KEPT_HOWEVER_LATE = (MEMORY_TARGET, SELF_MEMORY_TARGET, DIARY_TARGET, MEMORY_CONFLICT_TARGET)
# Of use only before the next turn: a note about a reply is never moved onto
# a later one.
_THIS_TURN_ONLY = (REPLY_NOTE_TARGET,)
# How a slip in her last reply is pointed out to her, one line each.
REPLY_NOTE_LINE = "About your last reply: "
# How she is told to ask about two facts of the user that cannot both be
# true, in the writing of the conversation; latin for any other.
CONFLICT_NOTES = {
    "zh": "關於使用者：之前記得「{earlier}」，現在聽到「{newer}」——找個自然的時機，輕輕問一句哪個才對。",
    "zh-hans": "关于用户：之前记得「{earlier}」，现在听到「{newer}」——找个自然的时机，轻轻问一句哪个才对。",
    "ja": "ユーザーについて：前は「{earlier}」、今は「{newer}」と聞いた。自然なときに、どちらが正しいか軽く聞いてみて。",
    "ko": "사용자에 대해: 전에는 「{earlier}」, 지금은 「{newer}」라고 들었어. 자연스러울 때 어느 쪽이 맞는지 가볍게 물어봐.",
    "latin": 'About the user: earlier they said "{earlier}", now "{newer}" — ask which is right, lightly, when it fits.',
}
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
    # Her mood as it stands now, faded by time; see CompanionSettings.
    emotion: str
    trust: float
    favorability: float
    relationship_stage: str
    goals: tuple[str, ...] = ()
    thoughts: tuple[str, ...] = ()
    # The intensity of her mood as it was set, and when (seconds since the
    # epoch): a host fades it itself between snapshots with the half-life.
    # 0 once her mood has faded to neutral.
    mood_intensity: float = 0.0
    mood_updated_at: float | None = None
    mood_half_life_seconds: float = DEFAULT_MOOD_HALF_LIFE_SECONDS
    # Below this intensity, faded, she is neutral again: a host that fades
    # the face itself stops where the engine does.
    mood_floor: float = DEFAULT_MOOD_FLOOR
    # How the user has been lately, while it is not too old; see
    # CompanionSettings.user_state_ttl_hours.
    user_state: UserState | None = None


@dataclass(frozen=True, slots=True)
class UserState:
    """How the user has been lately, from the user's own words: energy (low,
    normal, high), how their mood moved (down, flat, up), at most three
    things weighing on them, and the user's sentences that show it. Read by
    the user state worker; ``updated_at`` is when, by the companion's clock."""

    energy: str
    mood_trend: str
    concerns: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class DiaryEntry:
    """Her day, in her words. ``date`` is the day (YYYY-MM-DD, local time),
    ``evidence`` the lines of what she remembered of it that the entry rests
    on, ``conversation_ids`` the conversations of that day, and ``until`` the
    end of the time it covers (seconds since the epoch, her clock)."""

    date: str
    text: str
    evidence: tuple[str, ...] = ()
    conversation_ids: tuple[str | None, ...] = ()
    until: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """Two facts of the user she holds that cannot both be true: the newer
    one, the earlier one, and why, as the model put it. Settled with
    CharacterCompanion.resolve_conflict()."""

    id: str
    summary: str
    earlier_id: str
    earlier_summary: str
    reason: str = ""
    # "contradicts" for two facts not settled (memory_conflicts()),
    # "supersedes" for a fact the newer one replaced (replaced_memories()).
    relation: str = "contradicts"


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
    # Her reply read back for slips, for a note on her next reply: it has to
    # be in before she speaks again. After the user's emotion, before her mood.
    _Worker(
        "reply_check",
        BackgroundCognitionKind.REPLY_CHECK,
        CognitiveRole.REPLY_CHECK,
        TaskPriority.HIGH,
        REPLY_NOTE_TARGET,
    ),
    # Her own mood, from both sides of the conversation: the face she shows
    # once she stops talking. Right after the user's emotion; not in its lane.
    _Worker(
        "mood",
        BackgroundCognitionKind.CHARACTER_MOOD,
        CognitiveRole.MOOD,
        TaskPriority.HIGH,
        MOOD_TARGET,
    ),
    _Worker(
        "memory",
        BackgroundCognitionKind.MEMORY_EXTRACTION,
        CognitiveRole.MEMORY,
        TaskPriority.HIGH,
        MEMORY_TARGET,
    ),
    # A memory of the user just written, held against the earlier ones on its
    # topic. Asked when one is written, not after a turn; right after memory.
    _Worker(
        "memory_conflict",
        BackgroundCognitionKind.MEMORY_CONFLICT,
        CognitiveRole.MEMORY_CONFLICT,
        TaskPriority.HIGH,
        MEMORY_CONFLICT_TARGET,
    ),
    # What she said about herself. Before goals and thoughts: left for later,
    # she contradicts it within the next few turns. After memory of the user,
    # which the user is more likely to ask about.
    _Worker(
        "self_memory",
        BackgroundCognitionKind.SELF_MEMORY_EXTRACTION,
        CognitiveRole.SELF_MEMORY,
        TaskPriority.NORMAL,
        SELF_MEMORY_TARGET,
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
    # How the user has been lately: for when she speaks up or asks after them,
    # never urgent.
    _Worker(
        "user_state",
        BackgroundCognitionKind.USER_STATE,
        CognitiveRole.USER_STATE,
        TaskPriority.LOW,
        USER_STATE_TARGET,
    ),
    # Her day, in her words: once a day, last of all.
    _Worker(
        "diary",
        BackgroundCognitionKind.DIARY,
        CognitiveRole.DIARY,
        TaskPriority.LOW,
        DIARY_TARGET,
    ),
)


_TARGET_OF = {worker.kind: worker.target for worker in _WORKERS}


class _NoModel:
    """The client of a worker the host gave none: the reply check without a
    model still reads her opening. Called, it fails at once."""

    async def generate(self, messages, *, tools=None):
        raise RuntimeError("no model for this worker")


_NO_MODEL = _NoModel()


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
# How she is asked to speak up; not kept in the conversation. A host may pass
# its own, in the language she speaks.
SPEAK_UP_INSTRUCTION = """Nobody has said anything for a while. Speak up on your own, in character.

You are not replying to anyone. You answered the user's last message long ago: do not answer it again, and do not open as if responding ("as you said", "since you put it that way").

Three choices are equally fine; take whichever is most natural right now, not always the first:
1. Carry on with what was just being talked about, if something about it is still open: add a new point or ask about it. Do not restate what was already said.
2. Bring up something new: something you want to do or have been thinking about, or something the host suggests below. Start with it directly; do not announce a change of topic.
3. Simply turn to the user: say their name, what is on your mind right now, or ask what they are doing.

Do not mix the choices: ending one thing and tacking on an unrelated question sounds like two people talking.

- One thing at a time, at most one question. If you asked a question the last time you spoke up, do not ask one now.
- Do not sound like you are reading from notes or a news feed, and do not ask empty questions ("anything new?", "what do you want to talk about?").
- Do not say that you are speaking up, and never mention topic lists, news feeds or any other mechanism. Just say it."""
# Asked again when what she was about to say repeats what she said. Her words
# are not quoted: a small model echoes a quoted line back.
ALREADY_SAID = (
    "What you were about to say repeats something you already said in this "
    "conversation. Say something else, or turn to the user in a new way."
)
# Asked again for the other reasons a sentence is left out; none of them
# quotes her words either.
NOT_AN_ASSISTANT = (
    "What you were about to say sounds like an assistant offering help, not like "
    "you. Stay in character."
)
NOTHING_TO_ACKNOWLEDGE = (
    "What you were about to say only acknowledges something, and nobody said "
    "anything to acknowledge. Say something of your own."
)
NO_QUESTION = "Do not ask a question this time; say something without one."
# What the conversation keeps of a reply of which nothing was passed on.
SILENCE = "……"
# How often she is asked again before she stays quiet instead.
REMARK_ATTEMPTS = 3
# How many of her latest lines a remark is checked against.
LINES_CHECKED = 8
# What stays in the conversation before a remark she made on her own.
REMARK_EVENT = "The user had been quiet for a while; you spoke up on your own."
# After every worker: the workers are ranked by their place in _WORKERS.
_HOST_RANK = 1_000
# How the user seemed on the turns of a conversation since its user state was
# last read: at most this many.
_EMOTIONS_KEPT = 32
# A diary entry that could not be written is tried again no sooner than this,
# not on every turn.
_DIARY_RETRY_SECONDS = 3600.0
# What her days were made of is kept this long after her last entry, for an
# entry the host asks for about an earlier day.
_DAYS_KEPT_SECONDS = 8 * 86400.0


def _plain(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _the_same_fact(one: str, other: str) -> bool:
    """Two summaries of one fact: the same letters and digits, whatever the
    spacing and punctuation. Any other difference may be the whole point:
    "likes cats" and "does not like cats", "blue" and "red". Nearly the same
    words were taken for the same fact once, and what she said last was lost."""
    return _plain(one) == _plain(other)


def _repeats(text: str, lines: Sequence[str]) -> bool:
    """Whether ``text`` says again what one of ``lines`` said: nearly the same
    line, or most of it taken over word for word."""
    new = _plain(text)
    if not new:
        return True
    for line in lines:
        old = _plain(line)
        if not old:
            continue
        matcher = SequenceMatcher(None, new, old, autojunk=False)
        if matcher.ratio() >= 0.8:
            return True
        longest = matcher.find_longest_match(0, len(new), 0, len(old)).size
        if longest >= 12 and longest >= 0.6 * min(len(new), len(old)):
            return True
    return False


_SENTENCE_ENDS = "。！？!?…\n"
# Closing marks belong to the sentence before them: "*輕輕點頭。*" is one
# sentence, not "*輕輕點頭。" and a stray "*". An asterisk closes only an
# action left open; otherwise it opens the next one ("好啊！*笑著點頭*").
_CLOSERS = "」』）)】》〉\"'”’*＊"
_ASTERISKS = "*＊"


def _closes(text: str, start: int, index: int) -> bool:
    """Whether the mark at ``index`` closes the sentence that began at ``start``."""
    char = text[index]
    if char in _ASTERISKS:
        return sum(text[start:index].count(mark) for mark in _ASTERISKS) % 2 == 1
    return char in _CLOSERS or char in _SENTENCE_ENDS
# Sentences shorter than this ("Hmm.", "嗯。") are not checked: saying them
# again is not repeating oneself.
_SHORTEST_CHECKED = 8


# Support closings that break character. Small models fall into them at the end
# of a reply, whatever the persona says; compared casefolded, outside quotes,
# in the last _CLOSING_SENTENCES sentences only. Whole phrases, as a closing
# says them: shorter ones ("請隨時", "is there anything else", "let me know
# if you have any") end sentences she says in character too.
ASSISTANT_SPEAK = (
    "let me know if there's anything else",
    "let me know if there is anything else",
    "let me know if you need anything else",
    "let me know if you need any help",
    "let me know if you have any questions",
    "let me know if you have any other questions",
    "feel free to ask me anything",
    "feel free to ask any questions",
    "feel free to ask if you have",
    "feel free to reach out if",
    "how can i help you",
    "how can i assist",
    "is there anything else i can help",
    "is there anything else i can do for you",
    "i'm here to help",
    "i am here to help",
    "i hope this helps",
    "有什麼可以幫你",
    "有什麼可以幫您",
    "有什麼我可以幫",
    "如果還有其他問題",
    "如果需要進一步的幫助",
    "如果需要進一步的資訊",
    "如果你需要任何幫助",
    "我很樂意聆聽並提供幫助",
    "提供幫助或討論其他話題",
    "希望我們的交流能",
    "請告訴我你現在最關心",
    "我可以盡力回答",
    "祝你程式編寫一切順利",
    "有什么可以帮你",
    "有什么可以帮您",
    "有什么我可以帮",
    "如果还有其他问题",
    "如果需要进一步的帮助",
    "如果需要进一步的信息",
    "如果你需要任何帮助",
    "我很乐意聆听并提供帮助",
    "提供帮助或讨论其他话题",
    "希望我们的交流能",
    "请告诉我你现在最关心",
    "我可以尽力回答",
    "祝你程序编写一切顺利",
)
# Only so many of the last sentences are taken for a closing: earlier on, the
# same words are her own ("let me know if..., then we leave at six").
_CLOSING_SENTENCES = 2
# A remark that is nothing but one of these acknowledges what nobody said.
# Compared as _plain() of each part between punctuation.
ACKNOWLEDGEMENTS = frozenset(
    {
        "嗯", "嗯嗯", "嗯哼", "恩", "好", "好吧", "好的", "好啊", "好喔", "好哦", "行",
        "是", "是的", "是啊", "對", "對啊", "對呀", "对", "对啊", "对呀", "沒錯", "没错",
        "喔", "哦", "噢", "了解", "知道了", "明白", "確實", "确实",
        "ok", "okay", "yeah", "yes", "yep", "yup", "sure", "right", "alright", "fine",
        "mm", "mmm", "hmm", "mmhmm", "uhhuh", "isee", "gotit", "noted",
        "うん", "はい", "ええ", "そう", "そうだね", "なるほど",
    }
)
# Quoted words: 「」, 『』, “” or "".
_QUOTED = re.compile(r"「([^」]*)」|『([^』]*)』|“([^”]*)”|\"([^\"]*)\"")
# Quoted words shorter than this are not taken for her own.
_SHORTEST_QUOTE = 4


def _outside_quotes(sentence: str) -> str:
    """What the sentence says in its own voice. A shop assistant quoted in a
    story is not her talking like one."""
    said = _QUOTED.sub(" ", sentence)
    said = re.split(r"[「『“]", said)[0]  # quoted to the end of the sentence
    return re.split(r"[」』”]", said)[-1]  # quoted from before it


def _sounds_like_an_assistant(sentence: str) -> bool:
    said = " ".join(_outside_quotes(sentence).casefold().replace("’", "'").split())
    return any(phrase in said for phrase in ASSISTANT_SPEAK)


def _only_acknowledges(text: str) -> bool:
    parts = [_plain(part) for part in re.split(r"[^\w\s-]+", text)]
    parts = [part for part in parts if part]
    return bool(parts) and all(part in ACKNOWLEDGEMENTS for part in parts)


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().rstrip(_CLOSERS).rstrip().endswith(("?", "？"))


def _quoting_herself(text: str, lines: Sequence[str]) -> list[tuple[int, int]]:
    """Where ``text`` quotes words of hers from ``lines``: a model with nobody
    to answer quotes its own words back and reacts to them. Only her remarks
    are given: a title she named in a reply ("「進擊的巨人」") is hers to name
    again."""
    mine = [_plain(line) for line in lines]
    spans = []
    for match in _QUOTED.finditer(text):
        quoted = _plain(next(group for group in match.groups() if group is not None))
        if len(quoted) >= _SHORTEST_QUOTE and any(quoted in line for line in mine):
            spans.append(match.span())
    return spans


def _why_asked_again(reasons: set[str]) -> tuple[str, ...]:
    """Notes for the next attempt only. Not added to what she is answering:
    the conversation keeps that, and memory reads it as the user's words."""
    notes = {
        "repeat": ALREADY_SAID,
        "assistant": NOT_AN_ASSISTANT,
        "acknowledgement": NOTHING_TO_ACKNOWLEDGE,
        "question": NO_QUESTION,
    }
    return tuple(notes[reason] for reason in notes if reason in reasons)


def _sentences(text: str) -> tuple[list[str], str]:
    """The complete sentences at the start of ``text``, and what is left."""
    done, start, index = [], 0, 0
    while index < len(text):
        char = text[index]
        end = char in _SENTENCE_ENDS or (
            char == "." and (index + 1 == len(text) or text[index + 1].isspace())
        )
        if end and (char != "." or index + 1 < len(text)):
            index += 1
            while index < len(text) and _closes(text, start, index):
                index += 1
            while index < len(text) and text[index] == " ":
                index += 1
            done.append(text[start:index])
            start = index
            continue
        index += 1
    return done, text[start:]


def _repeats_a_line(sentence: str, lines: Sequence[str]) -> bool:
    new = _plain(sentence)
    if len(new) < _SHORTEST_CHECKED:
        return False
    for line in lines:
        old_line = _plain(line)
        if new in old_line:
            return True
        # Most of the sentence taken over word for word, with a new opening.
        longest = SequenceMatcher(None, new, old_line, autojunk=False).find_longest_match(
            0, len(new), 0, len(old_line)
        ).size
        if longest >= max(12, 0.6 * len(new)):
            return True
        said, rest = _sentences(line)
        for old in [*said, rest]:
            old = _plain(old)
            if not old:
                continue
            matcher = SequenceMatcher(None, new, old, autojunk=False)
            if matcher.ratio() >= 0.85:
                return True
            # Retold with a few words swapped along the way: most of it still
            # comes from the old sentence, in order, in pieces.
            kept = sum(block.size for block in matcher.get_matching_blocks() if block.size >= 4)
            if kept >= 0.8 * len(new):
                return True
    return False


def _only_punctuation(sentence: str) -> bool:
    """Punctuation, spaces and line breaks only. An emoji or another symbol
    says something: "好啊！😊" ends in one."""
    return all(unicodedata.category(char)[0] in "PZC" for char in sentence)


def _left_out(sentence: str, lines: Sequence[str]) -> str | None:
    """Why a sentence is left out wherever it stands: "" when it is nothing
    but punctuation, which a host shows as an empty subtitle; None when it is
    passed on. Assistant talk depends on where it stands; see ASSISTANT_SPEAK."""
    if _only_punctuation(sentence):
        return ""
    if _repeats_a_line(sentence, lines):
        return "repeat"
    return None


class _SentenceGate:
    """Passes a reply on sentence by sentence, leaving out any sentence that
    repeats one of her latest lines or is nothing but punctuation, and a
    closing that talks like an assistant. A sentence that may be such a
    closing waits until two more have come, or the reply has ended."""

    def __init__(self, lines: Sequence[str], forward) -> None:
        self._lines = lines
        self._forward = forward
        self._pending = ""
        self.passed = ""
        self.left_out = 0
        # Why sentences with words in them were left out.
        self.reasons: set[str] = set()
        # From the first sentence that may be a closing on.
        self._held: list[str] = []

    async def __call__(self, delta: str) -> None:
        self._pending += delta
        done, self._pending = _sentences(self._pending)
        for sentence in done:
            await self._offer(sentence)

    async def finish(self) -> None:
        rest, self._pending = self._pending, ""
        if rest:
            await self._offer(rest)
        # What is held now is among the last sentences.
        held, self._held = self._held, []
        for sentence in held:
            if _sounds_like_an_assistant(sentence):
                self._leave_out("assistant")
            else:
                await self._pass(sentence)

    async def _offer(self, sentence: str) -> None:
        if _only_punctuation(sentence):
            self._leave_out("")
            return
        if not self._held and not _sounds_like_an_assistant(sentence):
            await self._pass(sentence)
            return
        self._held.append(sentence)
        # A possible closing with enough sentences after it is none.
        while self._held and (
            len(self._held) > _CLOSING_SENTENCES
            or not _sounds_like_an_assistant(self._held[0])
        ):
            await self._pass(self._held.pop(0))

    def _leave_out(self, reason: str) -> None:
        self.left_out += 1
        if reason:
            self.reasons.add(reason)

    async def _pass(self, sentence: str) -> None:
        reason = _left_out(sentence, self._lines)
        if reason is not None:
            self._leave_out(reason)
            return
        self.passed += sentence
        if self._forward is not None:
            delivered = self._forward(sentence)
            if inspect.isawaitable(delivered):
                await delivered


def _what_a_remark_keeps(
    text: str, lines: Sequence[str], statement_only: bool, remarks: Sequence[str] = ()
) -> tuple[str, set[str]]:
    """The sentences of a remark that may be passed on, and why the others
    with words in them were left out. ``remarks`` are her latest remarks of
    her own, the words she must not quote back."""
    reasons: set[str] = set()
    quoted = _quoting_herself(text, remarks)
    kept, at = [], 0
    done, rest = _sentences(text)
    sentences = [*done, rest]
    with_words = [
        index for index, sentence in enumerate(sentences) if not _only_punctuation(sentence)
    ]
    closing = set(with_words[-_CLOSING_SENTENCES:])
    for index, sentence in enumerate(sentences):
        start, at = at, at + len(sentence)
        reason = _left_out(sentence, lines)
        if reason is None and index in closing and _sounds_like_an_assistant(sentence):
            reason = "assistant"
        if reason is None and any(start < end and begin < at for begin, end in quoted):
            reason = "repeat"
        if reason is None and statement_only and _is_question(sentence):
            reason = "question"
        if reason is None:
            kept.append(sentence)
        elif reason:
            reasons.add(reason)
    return "".join(kept).strip(), reasons


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


def _edited_lines(shown: Sequence[str], wanted: Sequence[str]) -> dict[str, str]:
    """Each line of ``wanted`` that took the place of a line of ``shown``,
    with the line it replaced: where the edited text has other lines than
    the shown one, they are paired in order."""
    before = [one_line(line) for line in shown if line.strip()]
    pairs: dict[str, str] = {}
    for op, i1, i2, j1, j2 in SequenceMatcher(a=before, b=list(wanted), autojunk=False).get_opcodes():
        if op == "replace":
            pairs.update(zip(wanted[j1:j2], before[i1:i2]))
    return pairs


def _came_from(metadata: Mapping[str, Any], conversation_id: str | None) -> bool:
    """Whether a record came from that conversation. One written before 1.2.0
    names none: it came from another."""
    return CONVERSATION_KEY in metadata and metadata[CONVERSATION_KEY] == conversation_id


class _GoalsInMind(GoalManager):
    """The goals she still has: none left untouched for too long. The store
    keeps them all; the goal worker reads the store. How many of them she
    keeps in mind is the context builder's ``goals_shown``: a goal pushed out
    by a more pressing one is not given up, only not in mind.

    A short-term goal is something to do in the conversation it came from:
    with ``conversation`` (the conversation at hand) it is in mind there only,
    and it leaves her mind sooner, after ``short_term_max_age_hours``."""

    def __init__(
        self,
        *,
        store,
        max_age_days: int,
        short_term_max_age_hours: float = 0.0,
        conversation: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(store=store)
        self._max_age = timedelta(days=max_age_days)
        self._short_term_max_age = (
            timedelta(hours=short_term_max_age_hours) if short_term_max_age_hours > 0 else None
        )
        self._conversation = conversation

    def active_goals(self, *, character_id: str) -> tuple[GoalRecord, ...]:
        now = self._now()
        cutoff = now - self._max_age
        short_cutoff = None if self._short_term_max_age is None else now - self._short_term_max_age
        here = _ANY if self._conversation is None else self._conversation()

        def in_mind(goal: GoalRecord) -> bool:
            if goal.updated_at < cutoff:
                return False
            if goal.horizon is not GoalHorizon.SHORT_TERM:
                return True
            if short_cutoff is not None and goal.updated_at < short_cutoff:
                return False
            return here is _ANY or _came_from(goal.metadata, here)

        return tuple(
            goal for goal in super().active_goals(character_id=character_id) if in_mind(goal)
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


def _diary_line(entry: DiaryEntry) -> dict[str, Any]:
    return {
        "date": entry.date,
        "text": entry.text,
        "evidence": list(entry.evidence),
        "conversation_ids": list(entry.conversation_ids),
        "until": entry.until,
    }


def _diary_from_text(text: str) -> list[DiaryEntry]:
    entries: list[DiaryEntry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            until = raw.get("until")
            entries.append(
                DiaryEntry(
                    date=str(raw["date"]),
                    text=str(raw["text"]),
                    evidence=tuple(str(item) for item in raw.get("evidence") or ()),
                    conversation_ids=tuple(raw.get("conversation_ids") or ()),
                    until=float(until) if isinstance(until, (int, float)) else None,
                )
            )
        except Exception as exc:
            # A damaged line must not cost her the rest of her diary.
            logger.warning("diary entry unreadable, skipped: %s", exc)
    return entries


def _day_log_from_text(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if isinstance(raw, dict) and isinstance(raw.get("at"), (int, float)):
            lines.append(raw)
    return lines


def _jsonl(lines) -> str:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)


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
        clock: Callable[[], float] | None = None,
        meta_dir: str | Path | None = None,
    ) -> None:
        """``background_llm`` is the model for background cognition: one client
        for every worker, or a mapping from worker name (emotion, reply_check,
        mood, memory, memory_conflict, self_memory, goal, reflection, summary)
        to a client; a worker without a client does not run. It defaults to ``llm``. Background workers must return JSON, so a
        client with a low temperature serves them better than the one tuned for
        conversation.

        Without ``storage_dir`` everything is kept in memory only.

        ``clock`` returns the time in seconds since the epoch; her mood is
        dated and fades by it, and what she writes down is dated and ages by
        it. The system clock when not given.

        ``meta_dir`` holds what she remembers across runs of a game's world
        (remember_across_runs): apart from ``storage_dir``, not in a save.
        Without it there is no such memory.
        """
        self.settings = settings or CompanionSettings()
        self._clock: Callable[[], float] = clock or time.time
        self._character = character
        self._notes: tuple[str, ...] = ()
        # A slip in her newest reply: (conversation, the reply as read, the
        # fixes), until her next turn; and the fixes that turn is given.
        self._reply_note: tuple[str | None, str, tuple[str, ...]] | None = None
        self._reply_fixes: tuple[str, ...] = ()
        # Two facts of the user that cannot both be true, for her to ask
        # about on this turn.
        self._conflict_notes: tuple[str, ...] = ()
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
        self._unfinished: tuple[str | None, str, str | None] | None = None
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
        # The turns (revisions) after which her mood was given to the mood
        # worker to read: on those the rules leave her mood to its reading.
        self._mood_read_at: deque[int] = deque(maxlen=self.settings.records_kept)
        self._live_by_kind: dict[BackgroundCognitionKind, list] = {}
        self._conversation_by_task: dict[str, str | None] = {}
        # Per task: the first message its conversation held when the job was
        # scheduled. Gone by the time of its result, the conversation was
        # trimmed since, unless the host took it back.
        self._first_by_task: dict[str, Message | None] = {}
        self._taken_back: deque[Message] = deque(maxlen=self.settings.records_kept)
        # Called with snapshot() when a background result changed her mood,
        # for a host that shows her face between replies.
        self.on_mood_change: Callable[[CompanionSnapshot], None] | None = None
        # Async listeners under way: held so that they are not collected.
        self._listening: set[asyncio.Future] = set()
        # Per conversation: how the user seemed on each turn since the user
        # state was last read.
        self._emotions: dict[str | None, deque] = {}
        # Her diary, oldest first, and what her days were made of: when she
        # talked in which conversation, and when her mood changed.
        self._diary_entries: list[DiaryEntry] = self._load_diary()
        self._day_log: list[dict[str, Any]] = self._load_day_log()
        # Also without a diary: a host runs for months.
        self._forget_old_days(self._clock())
        self._turn_started_at: float = self._clock()
        self._diary_tasks: set[str] = set()
        self._diary_by_hand: set[str] = set()
        self._diary_written: dict[str, DiaryEntry] = {}
        self._diary_tried_at: float | None = None

        builder = context_builder or ContextBuilder()
        builder.goals_shown = self.settings.goals_shown
        builder.mood_half_life_seconds = self.settings.mood_half_life_seconds
        builder.mood_floor = self.settings.mood_floor
        builder.clock = self._clock
        builder.diary = self._diary_in_context()
        self._across_runs = AcrossRuns(meta_dir) if meta_dir is not None else None
        builder.across_runs = self._across_runs_in_context()
        notes_of_the_host = builder.turn_notes
        builder.turn_notes = lambda: [
            *(notes_of_the_host() if notes_of_the_host is not None else ()),
            *self._thoughts_for_context(),
            *self._self_memories_for_context(),
            # Notes stay in the conversation; an instruction for one reply
            # must not read as a standing one. What the host knows is no
            # instruction.
            *(
                note if note.startswith("- ") else f"For the next reply only: {note}"
                for note in self._notes
            ),
            "\n".join(f"{REPLY_NOTE_LINE}{fix}" for fix in self._reply_fixes),
            *(f"For the next reply only: {note}" for note in self._conflict_notes),
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
            state_policy=self._policy(state_policy),
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
                short_term_max_age_hours=self.settings.short_term_goal_max_age_hours,
                conversation=(
                    (lambda: self._active)
                    if self.settings.plans_stay_in_conversation
                    else None
                ),
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
        # What she writes down is dated by her clock, and ages by it.
        self.runtime.memory_manager.clock = self._clock
        self.runtime.goal_manager.clock = self._clock
        self._forget_how_the_user_was()
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
            self._background.mood_half_life_seconds = self.settings.mood_half_life_seconds
            self._background.mood_floor = self.settings.mood_floor
            self._background.clock = self._clock
            self._background.job_extras[BackgroundCognitionKind.DIARY] = self._day_if_due
            self._background.job_extras[BackgroundCognitionKind.USER_STATE] = (
                self._how_the_user_seemed
            )
        self._commits = CognitiveCommitCoordinator(
            self._tasks, event_history=self.settings.records_kept
        )
        self._commits.clock = self._clock
        self._commits.mood_half_life_seconds = self.settings.mood_half_life_seconds
        self._commits.mood_floor = self.settings.mood_floor
        for target, policy in tuple(self._commits.policies.items()):
            if target in _THIS_TURN_ONLY:
                continue
            # The defaults ask for a rerun whenever a turn happened in between.
            # A local model needs 30 s or more for the background work of one
            # turn, so nearly nothing would ever be committed.
            self._commits.policies[target] = replace(
                policy, stale_policy=StalePolicy.ALLOW_MANUAL_REBASE
            )

    # --- assembly --------------------------------------------------------------

    def _store(self, persistent, in_memory, name: str):
        return persistent(self._dir / name) if self._dir is not None else in_memory()

    def _policy(self, state_policy: CharacterStatePolicy | None) -> CharacterStatePolicy:
        policy = state_policy or RelationshipStatePolicy()
        if isinstance(policy, RelationshipStatePolicy):
            # A copy: the host's instance may serve another companion, whose
            # clock and mood readings these would otherwise overwrite.
            policy = copy.copy(policy)
            policy.clock = self._clock
            policy.mood_half_life_seconds = self.settings.mood_half_life_seconds
            policy.mood_floor = self.settings.mood_floor
            policy.mood_read_for = lambda revision: revision in self._mood_read_at
        return policy

    def _now(self) -> datetime:
        """Her clock, as a date."""
        return datetime.fromtimestamp(self._clock(), UTC)

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
        self._date_mood_by_my_clock(state)
        return state

    def _date_mood_by_my_clock(self, state: CharacterState) -> None:
        now = self._clock()
        if state.mood_updated_at is None:
            # A state saved before 1.1.0 does not say when her mood was set:
            # it is as of now.
            state.mood_updated_at = now
        elif state.mood_updated_at > now:
            # Set by another machine's clock, milliseconds read as seconds, or
            # a save loaded after the game turned its clock back: as of now,
            # or her mood would not fade until then.
            state.mood_updated_at = now

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
            if self._every(worker) <= 0:
                continue
            client = source.get(worker.name) if isinstance(source, Mapping) else source
            if client is None and worker.name == "reply_check" and source:
                # Her opening said again needs no model: a host that names
                # other workers but no reply_check client still has it read.
                # An empty mapping turns all background work off, as in 1.1.
                client = _NO_MODEL
            if client is not None:
                clients.append((worker, client))
        return clients

    def _every(self, worker: _Worker) -> int:
        if worker.name == "diary":
            # Asked after every turn whether her day is due (_day_if_due), and
            # written whenever the host asks (write_diary).
            return 1
        if worker.kind is BackgroundCognitionKind.MEMORY_CONFLICT:
            # Asked when a memory of the user is written (_judge_conflicts).
            return 1 if self.settings.memory_conflicts else 0
        return int(getattr(self.settings, f"{worker.name}_every"))

    def _build_background(self, clients) -> BackgroundCognitionRuntime | None:
        if not clients:
            return None
        specs, endpoints, policies = [], [], {}
        for rank, (worker, client) in enumerate(clients):
            every = self._every(worker)
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
                    client=client
                    if client is _NO_MODEL
                    else PoliteClient(
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

    def _self_memories_for_context(self) -> list[str]:
        shown = self.settings.self_memories_shown
        said = self.self_memories(in_conversation=self._active)[-shown:] if shown else []
        return ["\n".join(f"{SELF_MEMORY_LINE}{line}" for line in said)] * bool(said)

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

    def _self_scope(self) -> str:
        """What she said about herself belongs to her, not to a conversation,
        and is kept apart from what she remembers of the user."""
        return f"{self.character.id}#self"

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
                    self._emotions.pop(forgotten, None)
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
        return await self._run(
            text,
            conversation_id=conversation_id,
            frames=frames,
            on_text_delta=on_text_delta,
            skip_memory=skip_memory,
            proactive=proactive,
            notes=notes,
            before_turn=before_turn,
            remember_as=remember_as,
            turn_id=turn_id,
            remark=None,
        )

    async def speak_up(
        self,
        conversation_id: str | None = None,
        *,
        frames: tuple[VisionFrame, ...] = (),
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
        notes: Sequence[str] = (),
        before_turn: Callable[[], None] | None = None,
        remember_as: Callable[[str], str] | None = None,
        turn_id: Any = None,
        keep: bool = True,
        instruction: str | None = None,
        statement_only: bool = False,
    ) -> CharacterRunResult:
        """She speaks up on her own; the host decides when (the user has been
        quiet, a timer, an event).

        What she says comes from her: what is still open in the conversation,
        what she wants and what she has been thinking about, or simply turning
        to the user. ``notes`` are what the host suggests for this remark only
        (topics the user likes, news), in the form reply() takes them.

        The instruction is not kept. What she said is, after a short event
        saying she spoke up on her own, as the conversation's newest reply; cut
        short, what was heard of it. ``keep=False`` keeps nothing, for a host
        that filters what she says and keeps what was spoken itself with
        remember_remark(). ``instruction`` replaces the engine's own, for a
        host that asks in the language she speaks; what she said the last time
        she spoke up is added to either.

        What she says is checked before any of it is passed on. A sentence
        that repeats one of her latest lines, quotes her own words back, talks
        like an assistant offering help, or is nothing but punctuation is left
        out; so is a question with ``statement_only``, for a host whose user
        has stayed quiet through her last questions. What is left must say
        something: a remark that only acknowledges ("OK.", "嗯。") is none.
        Otherwise she is asked again, up to REMARK_ATTEMPTS times, and then
        stays quiet.
        """
        return await self._run(
            instruction or SPEAK_UP_INSTRUCTION,
            conversation_id=conversation_id,
            frames=frames,
            on_text_delta=on_text_delta,
            skip_memory=True,
            proactive=True,
            notes=notes,
            before_turn=before_turn,
            remember_as=remember_as,
            turn_id=turn_id,
            remark="keep" if keep else "drop",
            statement_only=statement_only,
        )

    async def _run(
        self,
        text: str,
        *,
        conversation_id: str | None,
        frames: tuple[VisionFrame, ...],
        on_text_delta: Callable[[str], Awaitable[None] | None] | None,
        skip_memory: bool,
        proactive: bool,
        notes: Sequence[str],
        before_turn: Callable[[], None] | None,
        remember_as: Callable[[str], str] | None,
        turn_id: Any,
        remark: str | None,
        statement_only: bool = False,
    ) -> CharacterRunResult:
        """One turn. ``remark`` is None for a reply, "keep" or "drop" for a
        remark she made on her own (see speak_up)."""
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
                    self._record_interrupted(conversation_id, text, turn.heard, remark)
                    raise TurnInterrupted("The reply was interrupted.")
                self._unfinished = None
                self._turn_started_at = self._clock()
                if before_turn is not None:
                    before_turn()
                serial = next(self._serial)
                self._last_turn[conversation_id] = serial
                self._take_back_what_the_host_no_longer_knows(conversation_id, notes)
                self._take_back_what_she_no_longer_holds()
                self._forget_how_the_user_was()
                self._notes = tuple(note for note in notes if note.strip())
                self._reply_fixes = self._take_reply_note(conversation_id)
                self._conflict_notes = self._take_conflict_notes(conversation_id)
                self._access.foreground_started()
                try:
                    turn.generating = True
                    try:
                        if remark is None:
                            result = await self._reply_without_repeating(
                                text, frames, skip_memory, proactive, on_text_delta
                            )
                        else:
                            result = await self._something_new(
                                text, frames, on_text_delta, statement_only
                            )
                    except asyncio.CancelledError:
                        if turn.heard is None:
                            # The host cancelled its own task. It may tell us
                            # later how much was heard; see interrupt().
                            self._unfinished = (conversation_id, text, remark)
                            raise
                        self._record_interrupted(conversation_id, text, turn.heard, remark)
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
                    if remark == "keep" and result.text:
                        said = remember_as(result.text) if remember_as else result.text
                        self._keep_remark(conversation_id, said, turn_id)
                    handles: tuple = ()
                    if self._background is not None and not skip_memory and not self._closed:
                        # Never None: None would mean one conversation only.
                        self._background.conversation = ("conversation", conversation_id)
                        handles = await self._background.schedule_after_foreground(result)
                    self._take_on(handles)
                    if not skip_memory or (remark == "keep" and result.text):
                        # A day she talked: what her diary is about.
                        self._log_day(
                            {
                                "kind": "turn",
                                "at": self._turn_started_at,
                                "conversation": conversation_id,
                            }
                        )
                finally:
                    self._notes = ()
                    self._reply_fixes = ()
                    self._conflict_notes = ()
                    # Not before the new jobs are taken on: work that waited for
                    # the reply to end wakes up here and must find the hold for
                    # this turn's mood in place.
                    self._access.foreground_finished()
                self._save_state()
        finally:
            self._turns.remove(turn)
        for handle in handles:
            self._conversation_by_task[handle.task_id] = conversation_id
            self._first_by_task[handle.task_id] = (
                self.runtime.history[0] if self.runtime.history else None
            )
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

    def _take_back_what_she_no_longer_holds(self) -> None:
        """What she said about herself and has since forgotten: edited away by
        the host's user, or pushed out by newer ones. It was written into this
        conversation while she held it. What she said in another conversation
        about what she is doing there is not held here."""
        held = set(self.self_memories(in_conversation=self._active))
        self.runtime.withdraw_from_notes(
            lambda line: line.startswith(SELF_MEMORY_LINE)
            and line.removeprefix(SELF_MEMORY_LINE) not in held
        )

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
        if not remark.strip():
            return
        async with self._turn_lock:
            if self._closed:
                raise CompanionClosed("companion is closed")
            self._switch_to(conversation_id)
            self._keep_remark(conversation_id, remark)
            self._save_state()

    async def _reply_without_repeating(
        self, text, frames, skip_memory, proactive, on_text_delta
    ) -> CharacterRunResult:
        """A reply passed on sentence by sentence, without the sentences that
        repeat one of her latest lines, talk like an assistant or are nothing
        but punctuation; what is kept is what was passed on. A reply that was
        nothing but repetition or assistant talk, and used no tool, is asked
        again once. One of nothing but punctuation is her silence: it is not."""
        mine = [
            message.content
            for message in self.runtime.history
            if message.role == "assistant" and message.content
        ][-LINES_CHECKED:]
        notes = self._notes
        for attempt in range(2):
            before = self.runtime.history[-1] if self.runtime.history else None
            gate = _SentenceGate(mine, on_text_delta)
            result = await self._tasks.run_foreground_turn(
                lambda: self._bridge.process(
                    text,
                    frames=frames,
                    skip_memory=skip_memory,
                    proactive=proactive,
                    on_text_delta=gate,
                )
            )
            await gate.finish()
            if not gate.left_out:
                return result
            said = gate.passed.strip()
            if said or result.tool_results or attempt == 1 or not gate.reasons:
                if not skip_memory and said:
                    self._bridge.replace_reply(said)
                elif not skip_memory and gate.reasons:
                    # Never an empty line in the conversation: nothing of it
                    # was said.
                    self._bridge.replace_reply(SILENCE)
                # Nothing but punctuation stays as she gave it: her silence.
                return replace(result, response=replace(result.response, text=said))
            if not self._undo_turn(before):
                return replace(result, response=replace(result.response, text=""))
            self._notes = (*notes, *_why_asked_again(gate.reasons))
        raise AssertionError("unreachable")

    def _undo_turn(self, before: Message | None) -> bool:
        """Take out what the turn just made, back to ``before``; False when
        that message is no longer in the history."""
        history = self.runtime.history
        if before is None:
            gone, kept = list(history), []
        else:
            at = next((i for i, m in enumerate(history) if m is before), None)
            if at is None:
                return False
            gone, kept = history[at + 1 :], history[: at + 1]
        self.runtime.history[:] = kept
        self.runtime.context_notes = [
            (anchor, note)
            for anchor, note in self.runtime.context_notes
            if not any(anchor is m for m in gone)
        ]
        return True

    async def _something_new(
        self, text, frames, on_text_delta, statement_only=False
    ) -> CharacterRunResult:
        """A remark she has not made before. Generated whole before any of it is
        passed on; asked again when it repeats one of her latest lines or says
        nothing (see speak_up); nothing at all rather than a repetition."""
        history = self.runtime.history
        mine = [
            message.content
            for message in history
            if message.role == "assistant" and message.content
        ][-LINES_CHECKED:]
        remarks = [
            message.content
            for before, message in zip(history, history[1:])
            if message.role == "assistant"
            and before.role == "event"
            and before.content.startswith("type: proactive_remark")
        ][-LINES_CHECKED:]
        notes = self._notes
        result = None
        for _ in range(REMARK_ATTEMPTS):
            said: list[str] = []
            result = await self._tasks.run_foreground_turn(
                lambda: self._bridge.process(
                    text,
                    frames=frames,
                    skip_memory=True,
                    proactive=True,
                    on_text_delta=said.append,
                )
            )
            # Sentences she already said, and the others a reply leaves out,
            # are left out; what is left must not repeat her either.
            kept, reasons = _what_a_remark_keeps(result.text, mine, statement_only, remarks)
            # A word or two left over ("Hello.") once the rest was left out is
            # not a remark of its own; a short remark as it came is.
            enough = not reasons or len(_plain(kept)) >= _SHORTEST_CHECKED
            if kept and enough and _repeats(kept, mine):
                reasons.add("repeat")
            elif kept and enough and _only_acknowledges(kept):
                reasons.add("acknowledgement")
            elif kept and enough:
                if on_text_delta is not None:
                    delivered = on_text_delta(kept)
                    if inspect.isawaitable(delivered):
                        await delivered
                return replace(result, response=replace(result.response, text=kept))
            self._notes = (*notes, *_why_asked_again(reasons))
        # Nothing new to say: better quiet than the same line again.
        return replace(result, response=replace(result.response, text=""))

    def _keep_remark(self, conversation_id: str | None, remark: str, turn_id: Any = None) -> None:
        """Call with the turn lock held and the conversation at hand."""
        remark = remark.strip()
        if not remark:
            return
        event = self.runtime.context_builder.event_to_message(
            CharacterEvent(type="proactive_remark", source="host", content=REMARK_EVENT)
        )
        reply = Message("assistant", remark)
        self.runtime.history.extend([event, reply])
        serial = next(self._serial)
        self._last_turn[conversation_id] = serial
        self._newest_turn[conversation_id] = serial
        self._newest_reply[conversation_id] = (turn_id, reply)

    def _record_interrupted(
        self, conversation_id: str | None, text: str, heard: str, remark: str | None
    ) -> None:
        """What was heard of a turn cut short. A remark keeps what was heard of
        it as hers; its instruction is never recorded as the user's words."""
        if remark is None:
            self._bridge.record_interrupted_turn(text, heard)
        elif remark == "keep" and heard.strip():
            self._keep_remark(conversation_id, f"{heard.strip()} [Interrupted by user]")

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
        self._taken_back.extend(gone)
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
            conversation, text, remark = self._unfinished
            self._unfinished = None
            if conversation == self._active:
                self._record_interrupted(conversation, text, heard_response, remark)
        elif meant(self._active):
            newest = self._newest_reply.get(self._active)
            if turn_id is None or (newest and newest[0] == turn_id):
                self._bridge.interrupt(heard_response)
                # A remark she made on her own was kept by the companion, not by
                # a turn of the bridge: cut that very message.
                if newest and _cut(self.runtime.history, newest[1], heard_response):
                    self._newest_reply.pop(self._active, None)
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
        own, of the user and of her.

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
            if kind is BackgroundCognitionKind.CHARACTER_MOOD:
                self._mood_read_at.append(self._tasks.revision)
            if kind is BackgroundCognitionKind.DIARY:
                self._diary_tasks.add(handle.task_id)
            if kind is BackgroundCognitionKind.EMOTION_ANALYSIS:
                self._access.hold(handle.task_id, self.settings.call_timeout_seconds)
            live = self._live_by_kind.setdefault(kind, [])
            live[:] = [item for item in live if _under_way(item)]
            if kind not in _READ_ONCE:
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
        if proposal.target in _KEPT_HOWEVER_LATE:
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
            first = self._first_by_task.pop(handle.task_id, None)
            if result.output is None:
                logger.debug("%s %s: %s", result.task_type, result.status.value, result.error)
                return
            # The same lock as a turn: the conversation must not change under
            # a commit.
            async with self._turn_lock:
                if self._closed:
                    return
                mood_before = self._mood_as_stored()
                proposals = [
                    proposal
                    for proposal in result.output.proposals
                    if self._still_said(proposal, conversation_id, first)
                ]
                outcomes = [
                    await self._commit_what_she_said(proposal, conversation_id)
                    for proposal in proposals
                ]
                for proposal, outcome in zip(proposals, outcomes):
                    if outcome.committed and proposal.target == REPLY_NOTE_TARGET:
                        self._keep_reply_note(proposal, conversation_id)
                    if outcome.committed and proposal.target == MEMORY_TARGET:
                        await self._judge_conflicts(outcome, conversation_id, first)
                    if outcome.committed and proposal.target == DIARY_TARGET:
                        entry = self._keep_diary(proposal)
                        if handle.task_id in self._diary_by_hand:
                            self._diary_written[handle.task_id] = entry
                if any(o.committed and o.target == "state.emotion_candidate" for o in outcomes):
                    self._remember_how_the_user_seemed(conversation_id)
                    await self._react_to_observation()
                if any(o.committed and o.target == SELF_MEMORY_TARGET for o in outcomes):
                    self._keep_the_newest_self_memories()
                try:
                    if any(o.committed for o in outcomes):
                        self._save_state()
                finally:
                    # Her mood changed whether or not it reached the disk:
                    # the host still hears of it.
                    if self._mood_as_stored() != mood_before:
                        self._log_day(
                            {
                                "kind": "mood",
                                "at": self._clock(),
                                "mood": self._mood()[0],
                                "conversation": conversation_id,
                            }
                        )
                        self._tell_mood()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("background result dropped (%s: %s)", type(exc).__name__, exc)
        finally:
            self._conversation_by_task.pop(handle.task_id, None)
            self._first_by_task.pop(handle.task_id, None)
            self._diary_tasks.discard(handle.task_id)

    async def _judge_conflicts(self, outcome, conversation_id: str | None, first) -> None:
        """Call with the turn lock held. A memory of the user just written,
        held against the earlier ones on its topic that the commit named; no
        model is asked without any."""
        candidates = list(outcome.metadata.get("conflict_candidates") or ())
        if not candidates or self._background is None or self._closed:
            return
        records = {
            record.id: record
            for record in self._memory_store.list_for_character(self._scope(conversation_id))
        }
        new = records.get(outcome.applied_record_id or "")
        earlier = [records[old_id] for old_id in candidates if old_id in records]
        if new is None or not earlier:
            return
        handle = await self._background.schedule_memory_conflict(new, earlier)
        if handle is None:
            return
        live = self._live_by_kind.setdefault(BackgroundCognitionKind.MEMORY_CONFLICT, [])
        live[:] = [item for item in live if _under_way(item)]
        live.append(handle)
        self._conversation_by_task[handle.task_id] = conversation_id
        self._first_by_task[handle.task_id] = first
        task = asyncio.create_task(self._collect(handle))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _take_conflict_notes(self, conversation_id: str | None) -> tuple[str, ...]:
        """Call with the turn lock held. Two facts of the user that cannot
        both be true and that she has not been told to ask about: told now,
        once. Kept on the memories, so once also across a restart."""
        if not self.settings.memory_conflicts:
            return ()
        scope = self._scope(conversation_id)
        records, unasked = take_unasked_conflicts(self._memory_store.list_for_character(scope))
        if not unasked:
            return ()
        self._memory_store.replace_for_character(scope, records)
        notes = []
        for newer, earlier, _ in unasked:
            frame = CONFLICT_NOTES[self._note_script(newer.summary + earlier.summary)]
            notes.append(
                frame.format(earlier=one_line(earlier.summary), newer=one_line(newer.summary))
            )
        return tuple(notes)

    def _note_script(self, said: str) -> str:
        """The writing of the language the host names, else of the facts."""
        language = self.settings.language
        script = script_of_language(language) if language.strip() else script_of(said)
        if script == "zh":
            name = language.casefold()
            simplified = ("简", "simplified", "hans", "china", "singapore", "中国", "大陆", "新加坡")
            if any(word in name for word in simplified) or re.search(r"(?<![a-z])(?:cn|sg)(?![a-z])", name):
                return "zh-hans"
        return script if script in CONFLICT_NOTES else "latin"

    def _keep_reply_note(self, proposal, conversation_id: str | None) -> None:
        fixes = tuple(
            dict.fromkeys(
                str(issue.get("fix", "")).strip()
                for issue in proposal.payload.get("issues") or ()
                if isinstance(issue, Mapping) and str(issue.get("fix", "")).strip()
            )
        )
        if fixes:
            reply = str(proposal.provenance.get("reply", ""))
            self._reply_note = (conversation_id, reply, fixes)

    def _take_reply_note(self, conversation_id: str | None) -> tuple[str, ...]:
        """Call with the turn lock held and the conversation at hand. The
        fixes for this turn, once: only in the conversation of the reply they
        are about, and only while that reply is still her newest as it was
        read. Cut short or rewritten, it is not what the user heard."""
        note, self._reply_note = self._reply_note, None
        if note is None or note[0] != conversation_id:
            return ()
        newest = self._newest_reply.get(conversation_id)
        if (
            newest is None
            or not any(message is newest[1] for message in self.runtime.history)
            or newest[1].content.strip() != note[1].strip()
        ):
            return ()
        return note[2]

    def _mood_as_stored(self) -> tuple[str, float, float | None]:
        state = self.runtime.state
        return state.emotion, state.mood_intensity, state.mood_updated_at

    def _tell_mood(self) -> None:
        listener = self.on_mood_change
        if listener is None:
            return
        try:
            told = listener(self.snapshot())
        except Exception as exc:
            # The host's listener must not cost her the result.
            logger.warning("on_mood_change failed (%s: %s)", type(exc).__name__, exc)
            return
        if not inspect.isawaitable(told):
            return
        # An async listener: run it on the loop, not under the turn lock
        # this is called with.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if inspect.iscoroutine(told):
                told.close()
            logger.warning("on_mood_change is async but no event loop is running")
            return
        task = asyncio.ensure_future(told, loop=loop)
        self._listening.add(task)
        task.add_done_callback(self._listened)

    def _listened(self, task: asyncio.Future) -> None:
        self._listening.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("on_mood_change failed (%s: %s)", type(exc).__name__, exc)

    def _still_said(
        self, proposal, conversation_id: str | None, first: Message | None
    ) -> bool:
        """Whether what she said about herself is still in the conversation
        as it was heard. The job read her reply when the turn ended; cut short
        while it was played, or taken back by the host, the rest of it never
        reached the user. A conversation no longer held, or trimmed since the
        job was scheduled, cannot tell: only a quote missing from a turn still
        there is refused."""
        if proposal.target != SELF_MEMORY_TARGET:
            return True
        if self._started and conversation_id == self._active:
            history = self.runtime.history
        elif conversation_id in self._kept:
            history = self._kept[conversation_id][0]
        else:
            return True
        if (
            first is not None
            and not any(message is first for message in history)
            and not any(message is first for message in self._taken_back)
        ):
            return True
        quote = "".join(proposal.provenance.get("evidence") or ())
        return said_in(quote, [m.content for m in history if m.role == "assistant"])

    async def _commit_what_she_said(self, proposal, conversation_id: str | None):
        """Call with the turn lock held. What she says again about herself
        replaces what she held: she says the same about herself often, once is
        enough, and the newest is the one in mind."""
        if proposal.target != SELF_MEMORY_TARGET:
            return await self._commit(proposal, conversation_id)
        scope = self._self_scope()
        before = self._memory_store.list_for_character(scope)
        summary = str(proposal.payload.get("summary", ""))
        now = self._now()
        same = [r.id for r in before if r.is_active and _the_same_fact(r.summary, summary)]
        if same:
            self._memory_store.replace_for_character(
                scope,
                [
                    replace(record, status="forgotten", forgotten_at=now)
                    if record.id in same
                    else record
                    for record in before
                ],
            )
        outcome = await self._commit(proposal, conversation_id)
        if same and not outcome.committed:
            self._memory_store.replace_for_character(scope, before)
        return outcome

    def _keep_the_newest_self_memories(self) -> None:
        """The oldest beyond ``self_memories_kept`` are forgotten: counted
        apart for what is hers in every conversation and for what she holds
        in each conversation, so that one long conversation does not push out
        who she is."""
        scope = self._self_scope()
        records = self._memory_store.list_for_character(scope)
        active = sorted(
            (record for record in records if record.is_active),
            key=lambda record: record.created_at,
        )
        shelves: dict[tuple, list[MemoryRecord]] = {}
        for record in active:
            shelves.setdefault(self._shelf(record), []).append(record)
        kept = self.settings.self_memories_kept
        over = {record.id for shelf in shelves.values() for record in shelf[:-kept]}
        if not over:
            return
        now = self._now()
        self._memory_store.replace_for_character(
            scope,
            [
                replace(record, status="forgotten", forgotten_at=now)
                if record.id in over
                else record
                for record in records
            ],
        )

    def _shelf(self, record: MemoryRecord) -> tuple:
        """Which count of self memories a record is in; see
        _keep_the_newest_self_memories."""
        if not self.settings.plans_stay_in_conversation or not stays_in_conversation(
            record.kind
        ):
            return ("every conversation",)
        if CONVERSATION_KEY not in record.metadata:
            return ("before 1.2.0",)
        return ("conversation", record.metadata[CONVERSATION_KEY])

    def _in_mind(self, record: MemoryRecord, conversation_id: str | None) -> bool:
        """Whether what she said about herself is in her mind in that
        conversation: hers in every conversation, or said there."""
        if not self.settings.plans_stay_in_conversation or not stays_in_conversation(
            record.kind
        ):
            return True
        return _came_from(record.metadata, conversation_id)

    async def _react_to_observation(self) -> None:
        policy = self.runtime.state_policy
        if not isinstance(policy, RelationshipStatePolicy):
            return
        async with self._tasks.authority_guard():
            patch = policy.patch(self.runtime.state.snapshot(), count_turn=False)
            if patch is not None:
                self.runtime.state.apply(patch)

    async def _commit(self, proposal, conversation_id: str | None):
        """Call with the turn lock held.

        The coordinator writes memory into the scope that is current when it
        commits. A result that arrives after the user moved to another
        conversation would land there, so the scope is set back to the
        conversation the job came from for the duration of the commit.

        The proposal is marked with that conversation: what she holds only
        there (see plans_stay_in_conversation) is told apart by it.
        """
        proposal = replace(
            proposal, provenance={**proposal.provenance, CONVERSATION_KEY: conversation_id}
        )
        current_scope = self.runtime.memory_scope_id
        if proposal.target in CONVERSATION_SCOPED_TARGETS:
            self.runtime.memory_scope_id = self._scope(conversation_id)
        elif proposal.target == SELF_MEMORY_TARGET:
            self.runtime.memory_scope_id = self._self_scope()
        try:
            outcome = await self._commits.commit(proposal)
            turns_late = self._tasks.revision - proposal.base_revision
            if (
                outcome.status is CommitStatus.STALE
                and outcome.reason == "foreground_revision_changed"
                and turns_late > 0
                and proposal.target not in _THIS_TURN_ONLY
                and (
                    proposal.target in _KEPT_HOWEVER_LATE
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

    # --- how the user has been, and her diary ------------------------------------

    def _remember_how_the_user_seemed(self, conversation_id: str | None) -> None:
        """Call when the user's emotion was committed: one more reading of how
        the user seemed, for the next user state of that conversation."""
        observed = self.runtime.state.custom.get("observed_user_emotion")
        if not isinstance(observed, Mapping):
            return
        self._emotions.setdefault(conversation_id, deque(maxlen=_EMOTIONS_KEPT)).append(
            {key: observed[key] for key in ("emotion", "valence", "stance") if key in observed}
        )

    def _how_the_user_seemed(self) -> Mapping[str, Any]:
        """For the user state of the conversation at hand: how the user seemed
        on each turn since it was last read."""
        return {"user_emotions": list(self._emotions.pop(self._active, ()))}

    def _user_state_expired(self, stored: Mapping[str, Any]) -> bool:
        ttl = self.settings.user_state_ttl_hours
        if ttl <= 0:
            return False
        updated = stored.get("updated_at")
        if isinstance(updated, bool) or not isinstance(updated, (int, float)):
            return True
        return self._clock() - float(updated) > ttl * 3600

    def _user_state(self) -> UserState | None:
        stored = self.runtime.state.custom.get("user_state")
        if not isinstance(stored, Mapping) or self._user_state_expired(stored):
            return None
        updated = stored.get("updated_at")
        return UserState(
            energy=str(stored.get("energy") or ""),
            mood_trend=str(stored.get("mood_trend") or ""),
            concerns=tuple(str(item) for item in stored.get("concerns") or ()),
            evidence=tuple(str(item) for item in stored.get("evidence") or ()),
            updated_at=float(updated) if isinstance(updated, (int, float)) else None,
        )

    def _forget_how_the_user_was(self) -> None:
        """How the user was, read too long ago, is forgotten: out of her state
        and out of the notes of the conversation at hand."""
        stored = self.runtime.state.custom.get("user_state")
        if stored is None or (isinstance(stored, Mapping) and not self._user_state_expired(stored)):
            return
        del self.runtime.state.custom["user_state"]
        self.runtime.withdraw_from_notes(lambda line: line.startswith(USER_LATELY_LINE))

    def _diary_file(self) -> Path | None:
        return self._dir / "diary.jsonl" if self._dir is not None else None

    def _day_log_file(self) -> Path | None:
        return self._dir / "diary_log.jsonl" if self._dir is not None else None

    def _load_diary(self) -> list[DiaryEntry]:
        path = self._diary_file()
        if path is None or not path.is_file():
            return []
        return _diary_from_text(path.read_text(encoding="utf-8"))

    def _load_day_log(self) -> list[dict[str, Any]]:
        path = self._day_log_file()
        if path is None or not path.is_file():
            return []
        return _day_log_from_text(path.read_text(encoding="utf-8"))

    def _log_day(self, line: dict[str, Any]) -> None:
        if self._closed:
            return
        self._day_log.append(line)
        path = self._day_log_file()
        if path is not None:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _forget_old_days(self, until: float | None) -> None:
        """What her days were made of, kept a while after her last entry."""
        if until is None:
            return
        kept = [line for line in self._day_log if line["at"] >= until - _DAYS_KEPT_SECONDS]
        if len(kept) == len(self._day_log):
            return
        self._day_log = kept
        path = self._day_log_file()
        if path is not None:
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(_jsonl(kept), encoding="utf-8")
            os.replace(temporary, path)

    def _diary_in_context(self) -> tuple[str, str] | None:
        """The start of her last entry, for her system prompt: two sentences."""
        if not self.settings.diary_in_context or not self._diary_entries:
            return None
        entry = self._diary_entries[-1]
        done, rest = _sentences(entry.text)
        sentences = [sentence for sentence in (*done, rest) if sentence.strip()]
        return entry.date, "".join(sentences[:2]).strip()

    def _diary_since(self) -> float | None:
        """Where the day of her next entry begins: where her last one ended,
        or, before her first, when she first talked."""
        ends = [entry.until for entry in self._diary_entries if entry.until is not None]
        if ends:
            return max(ends)
        turns = [line["at"] for line in self._day_log if line.get("kind") == "turn"]
        return min(turns) if turns else None

    def _day_between(
        self, since: float, until: float, *, date: str | None = None
    ) -> dict[str, Any] | None:
        """What her day was made of, from ``since`` to ``until``: what she
        remembers of it, how she felt and what she set out to do. None when
        she talked to nobody."""
        turns = [
            line
            for line in self._day_log
            if line.get("kind") == "turn" and since <= line["at"] < until
        ]
        if not turns:
            return None
        conversations = list(dict.fromkeys(line.get("conversation") for line in turns))

        def within(moment: datetime) -> bool:
            return since <= moment.timestamp() < until

        summaries: list[str] = []
        told: list[str] = []
        for conversation in conversations:
            for record in self._memory_store.list_for_character(self._scope(conversation)):
                if record.is_active and within(record.created_at):
                    kept = summaries if record.kind == "conversation_summary" else told
                    kept.append(one_line(record.summary))
        hers: list[str] = []
        views: list[str] = []
        for record in self._memory_store.list_for_character(self._self_scope()):
            if record.is_active and within(record.created_at):
                kept = views if self_memory_kind(record.kind) == "view_of_user" else hers
                kept.append(one_line(record.summary))
        moods: list[str] = []
        felt = None
        for line in self._day_log:
            if line.get("kind") == "mood" and since <= line["at"] < until and line.get("mood") != felt:
                felt = line.get("mood")
                moods.append(f"{datetime.fromtimestamp(line['at']):%H:%M} {felt}")
        goals = [
            f"{'new' if within(goal.created_at) else goal.status.value}: {one_line(goal.objective)}"
            for goal in self.runtime.goal_manager.store.list_goals(self.character.id)
            if within(goal.updated_at)
        ]
        return {
            "date": date or datetime.fromtimestamp(turns[-1]["at"]).date().isoformat(),
            "until": until,
            "conversation_ids": conversations,
            "summaries": summaries,
            "user_told": told,
            "said_about_herself": hers,
            "views_of_user": views,
            "moods": moods,
            "goals": goals[:8],
        }

    def _day_if_due(self) -> Mapping[str, Any] | None:
        """Asked after every turn: her day, when diary_every_hours have passed
        since her last entry and she talked to someone since. The turn that
        asks belongs to the next day."""
        hours = self.settings.diary_every_hours
        if hours <= 0 or self._diary_tasks:
            return None
        since, until = self._diary_since(), self._turn_started_at
        if since is None or until - since < hours * 3600:
            return None
        if self._diary_tried_at is not None and until - self._diary_tried_at < _DIARY_RETRY_SECONDS:
            return None
        day = self._day_between(since, until)
        if day is None:
            return None
        self._diary_tried_at = until
        return {"diary": day}

    def _keep_diary(self, proposal) -> DiaryEntry:
        """Call with the turn lock held: an entry the coordinator accepted."""
        payload = proposal.payload
        until = payload.get("until")
        entry = DiaryEntry(
            date=str(payload.get("date") or ""),
            text=str(payload.get("text") or ""),
            evidence=tuple(str(item) for item in payload.get("evidence") or ()),
            conversation_ids=tuple(payload.get("conversation_ids") or ()),
            until=float(until) if isinstance(until, (int, float)) else None,
        )
        self._diary_entries.append(entry)
        path = self._diary_file()
        if path is not None:
            with path.open("a", encoding="utf-8") as file:
                file.write(_jsonl([_diary_line(entry)]))
        self.runtime.context_builder.diary = self._diary_in_context()
        self._forget_old_days(entry.until)
        return entry

    async def write_diary(self, date: Date | str | None = None) -> DiaryEntry | None:
        """Her diary entry, now, for a host that wants one (to read before a
        stream). Without ``date``, about the time since her last entry; with
        one (a date or "YYYY-MM-DD"), about that day, local time. None when
        she talked to nobody then, there is no diary worker, or the entry
        rested on nothing that happened."""
        if self._closed:
            raise CompanionClosed("companion is closed")
        background = self._background
        if background is None or background.config.spec_for(BackgroundCognitionKind.DIARY) is None:
            return None
        now = self._clock()
        if date is None:
            since = self._diary_since()
            day = None if since is None else self._day_between(since, now)
        else:
            day_of = date if isinstance(date, Date) else Date.fromisoformat(str(date))
            start = datetime.combine(day_of, datetime.min.time()).timestamp()
            day = self._day_between(start, min(start + 86400.0, now), date=day_of.isoformat())
        if day is None:
            return None
        handle = await background.submit_now(BackgroundCognitionKind.DIARY, {"diary": day})
        self._diary_tasks.add(handle.task_id)
        self._diary_by_hand.add(handle.task_id)
        try:
            await self._collect(handle)
        finally:
            self._diary_by_hand.discard(handle.task_id)
        return self._diary_written.pop(handle.task_id, None)

    def diary(self, limit: int = 7) -> tuple[DiaryEntry, ...]:
        """Her newest ``limit`` diary entries, oldest first."""
        if limit <= 0:
            return ()
        return tuple(self._diary_entries[-limit:])

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

    # --- across runs -----------------------------------------------------------

    def _across_runs_in_context(self) -> str | None:
        if self._across_runs is None:
            return None
        return across_runs_section(
            self._across_runs.memories,
            framing=self.settings.across_runs_framing,
            shown=self.settings.across_runs_in_context,
        )

    def remember_across_runs(
        self, text: str, *, tags: Sequence[str] = (), run: int | None = None
    ) -> str:
        """Something she keeps when the game's world begins again: written by
        the game, never by her background work. Her system prompt holds the
        newest ``across_runs_in_context``. Returns its id. Raises
        ``RuntimeError`` without ``meta_dir``."""
        if self._across_runs is None:
            raise RuntimeError("memories across runs need meta_dir")
        memory = self._across_runs.add(text, tags=tags, created_at=self._clock(), run=run)
        self.runtime.context_builder.across_runs = self._across_runs_in_context()
        return memory.id

    def across_runs(self) -> list[AcrossRunsMemory]:
        """What she remembers across runs, oldest first."""
        return [] if self._across_runs is None else list(self._across_runs.memories)

    def forget_across_runs(self, memory_id: str) -> bool:
        if self._across_runs is None or not self._across_runs.forget(memory_id):
            return False
        self.runtime.context_builder.across_runs = self._across_runs_in_context()
        return True

    # --- saves -----------------------------------------------------------------

    async def export_state(self, *, timeout: float = 120.0) -> bytes:
        """Everything she keeps, as bytes for a game's own save: her state,
        memories, goals, thoughts, diary and the conversations she holds.

        Waits for a reply under way and for her background work to settle,
        so that the save holds a whole turn. Raises ``StateBusy`` when that
        takes more than ``timeout`` seconds; nothing is saved then.
        """
        if self._closed:
            raise CompanionClosed("companion is closed")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        busy = StateBusy(f"her background work did not settle in {timeout} s")
        while True:
            # Waited for, never cancelled: her work goes on when the save
            # gives up.
            while self._pending:
                left = deadline - loop.time()
                if left <= 0:
                    raise busy
                await asyncio.wait(tuple(self._pending), timeout=left)
            try:
                await asyncio.wait_for(self._turn_lock.acquire(), max(0.0, deadline - loop.time()))
            except TimeoutError:
                raise busy from None
            try:
                # Nothing below awaits: no turn and no commit can change her
                # while she is packed.
                if not self._pending:
                    return self._pack()
            finally:
                self._turn_lock.release()

    def _pack(self) -> bytes:
        kept = [(conversation, history, notes) for conversation, (history, notes) in self._kept.items()]
        if self._started:
            kept.append((self._active, list(self.runtime.history), list(self.runtime.context_notes)))
        cognition = self.runtime.long_term_cognition.store
        files = {
            save_state.STATE: json.dumps(
                state_to_dict(self.runtime.state.snapshot()), ensure_ascii=False, indent=2
            ).encode("utf-8"),
            save_state.MEMORY: save_state.memory_lines(self._memory_store),
            save_state.GOALS: save_state.goal_lines(self.runtime.goal_manager.store),
            save_state.COGNITION: save_state.cognition_lines(cognition),
            save_state.DIARY: _jsonl(_diary_line(entry) for entry in self._diary_entries).encode("utf-8"),
            save_state.DAY_LOG: _jsonl(self._day_log).encode("utf-8"),
            save_state.CONVERSATIONS: save_state.conversations_to_bytes(
                active=self._active, started=self._started, kept=kept, told=self._told
            ),
        }
        return save_state.pack(files, character_id=self.character.id, clock=self._clock())

    def import_state(self, data: bytes, *, allow_other_character: bool = False) -> None:
        """Make her what ``export_state`` saved, before she first speaks.

        Everything she keeps is replaced, in ``storage_dir`` too; what she
        remembers across runs (``meta_dir``) is not. Raises ``RuntimeError``
        once she has spoken (to load a save, start a new companion),
        ``StateFormatError`` for a damaged save or one made by a newer engine,
        and ``ValueError`` for a save of another character unless
        ``allow_other_character``. Nothing changes when it raises.
        """
        if self._closed:
            raise CompanionClosed("companion is closed")
        if self._loop is not None or self._turns:
            raise RuntimeError("a save is loaded before she first speaks; start a new companion")
        manifest, files = save_state.unpack(data)
        saved_as = manifest.get("character_id")
        if saved_as != self.character.id and not allow_other_character:
            raise ValueError(
                f"a save of {saved_as!r}, not of {self.character.id!r}; "
                "pass allow_other_character=True to load it anyway"
            )
        # Everything is read before anything is replaced.
        try:
            state = state_from_dict(json.loads(files[save_state.STATE]))
            memory = save_state.read_memory(files[save_state.MEMORY])
            goals = save_state.read_goals(files[save_state.GOALS])
            reflections, beliefs = save_state.read_cognition(files[save_state.COGNITION])
            diary = _diary_from_text(files[save_state.DIARY].decode("utf-8"))
            day_log = _day_log_from_text(files[save_state.DAY_LOG].decode("utf-8"))
            active, started, kept, told = save_state.conversations_from_bytes(
                files[save_state.CONVERSATIONS]
            )
        except Exception as exc:
            raise StateFormatError(f"damaged save: {exc}") from exc
        if isinstance(saved_as, str) and saved_as != self.character.id:
            scope = save_state.as_another_character(saved_as, self.character.id)
            memory, goals = save_state.rescoped(memory, scope), save_state.rescoped(goals, scope)
            reflections = save_state.rescoped(reflections, scope)
            beliefs = save_state.rescoped(beliefs, scope)

        save_state.put_memory(self._memory_store, memory)
        save_state.put_goals(self.runtime.goal_manager.store, goals)
        save_state.put_cognition(self.runtime.long_term_cognition.store, reflections, beliefs)
        self.runtime.state.restore(state)
        self._date_mood_by_my_clock(self.runtime.state)
        self._mood_read_at.clear()
        self._save_state()
        self._diary_entries = diary
        self._day_log = day_log
        for path, lines in (
            (self._diary_file(), [_diary_line(entry) for entry in diary]),
            (self._day_log_file(), day_log),
        ):
            if path is not None:
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text(_jsonl(lines), encoding="utf-8")
                os.replace(temporary, path)
        self._diary_tried_at = None
        self._turn_started_at = self._clock()
        self.runtime.context_builder.diary = self._diary_in_context()

        for forget in (self._told, self._emotions, self._newest_reply, self._last_turn, self._newest_turn):
            forget.clear()
        self._told.update(told)
        held = OrderedDict((conversation, (history, notes)) for conversation, history, notes in kept)
        for conversation in held:
            # What the host loaded before is older than the save.
            self._loaded.pop(conversation, None)
        if started:
            history, notes = held.pop(active, ([], []))
            self._bridge.restore_history(history)
            self.runtime.context_notes = notes
            self.runtime.memory_scope_id = self._scope(active)
        else:
            self._bridge.restore_history([])
        self._kept = held
        self._active = active
        self._started = started
        self._forget_how_the_user_was()

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

    def _mood(self) -> tuple[str, float]:
        """Her mood as it stands now, and how strongly she still feels it."""
        state = self.runtime.state
        return effective_mood(
            state.emotion,
            state.mood_intensity,
            state.mood_updated_at,
            now=self._clock(),
            half_life_seconds=self.settings.mood_half_life_seconds,
            floor=self.settings.mood_floor,
        )

    def snapshot(self) -> CompanionSnapshot:
        """Goals and thoughts are those in mind in the conversation at hand."""
        state = self.runtime.state
        reflections = self.runtime.long_term_cognition.reflections(character_id=self.character.id)
        if self.settings.plans_stay_in_conversation:
            reflections = [
                record for record in reflections if _came_from(record.metadata, self._active)
            ]
        newest_first = sorted(reflections, key=lambda record: record.created_at, reverse=True)
        goals = [
            goal.objective
            for goal in self.runtime.goal_manager.active_goals(character_id=self.character.id)
        ][: self.settings.goals_shown]
        thoughts = [record.insight for record in newest_first[: self.settings.thoughts_shown]]
        mood, _ = self._mood()
        return CompanionSnapshot(
            emotion=mood,
            trust=state.trust,
            favorability=state.favorability,
            relationship_stage=state.relationship_stage,
            goals=tuple(goals),
            thoughts=tuple(thoughts),
            mood_intensity=0.0 if mood == NEUTRAL else state.mood_intensity,
            mood_updated_at=state.mood_updated_at,
            mood_half_life_seconds=self.settings.mood_half_life_seconds,
            mood_floor=self.settings.mood_floor,
            user_state=self._user_state(),
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
        self._rewrite(self._scope(conversation_id), summaries, edited_from, "asserted_fact")

    def memory_conflicts(self, conversation_id: str | None = None) -> list[MemoryConflict]:
        """Two facts of the user she holds in a conversation that cannot both
        be true and are not settled, oldest first: for a host's memory page."""
        records = self._memory_store.list_for_character(self._scope(conversation_id))
        return [
            MemoryConflict(
                id=newer.id,
                summary=one_line(newer.summary),
                earlier_id=earlier.id,
                earlier_summary=one_line(earlier.summary),
                reason=reason,
            )
            for newer, earlier, reason in unresolved_conflicts(records)
        ]

    def replaced_memories(self, conversation_id: str | None = None) -> list[MemoryConflict]:
        """Facts of the user that a newer one replaced, with the newer one
        still held, oldest first: a memory page shows what was replaced and
        why. Forgetting the newer one (rewrite_memories) brings the earlier
        one back."""
        records = self._memory_store.list_for_character(self._scope(conversation_id))
        by_id = {record.id: record for record in records}
        found = []
        for earlier in sorted(records, key=lambda record: record.created_at):
            newer = by_id.get(earlier.superseded_by or "")
            if earlier.status != "superseded" or newer is None or not newer.is_active:
                continue
            reason = next(
                (
                    str(entry.get("reason") or "")
                    for entry in newer.metadata.get(CONFLICTS_LOG) or ()
                    if isinstance(entry, Mapping) and entry.get("old_id") == earlier.id
                ),
                "",
            )
            found.append(
                MemoryConflict(
                    id=newer.id,
                    summary=one_line(newer.summary),
                    earlier_id=earlier.id,
                    earlier_summary=one_line(earlier.summary),
                    reason=reason,
                    relation="supersedes",
                )
            )
        return found

    def resolve_conflict(self, keep_id: str, *, conversation_id: str | None = None) -> bool:
        """Settle the contradictions a fact is in, as the user answered: the
        fact ``keep_id`` stays, the one against it leaves her mind. False
        when it is in none."""
        scope = self._scope(conversation_id)
        records, settled = resolve_conflict(self._memory_store.list_for_character(scope), keep_id)
        if settled:
            self._memory_store.replace_for_character(scope, records)
        return settled

    def memories(self, conversation_id: str | None = None) -> list[str]:
        records = self._memory_store.list_for_character(self._scope(conversation_id))
        return [one_line(record.summary) for record in records if record.is_active]

    def rewrite_self_memories(
        self,
        summaries: Sequence[str],
        *,
        edited_from: Sequence[str] | None = None,
        from_before: bool = False,
    ) -> None:
        """What she said about herself, as the host's user edited it, or as a
        host that kept such lines itself hands them over.

        The same as rewrite_memories(): lines that are still there stay, a line
        that was shown and is gone is forgotten, a new line is remembered as
        something she said. Beyond ``self_memories_kept`` the oldest are
        forgotten.

        ``from_before`` is for memories brought in from before the engine kept
        them: new lines are dated before every one she holds, in the order
        given, so they are in mind last and forgotten first. Dated now, they
        took the places of what she said most recently. ``edited_from=[]``
        only adds.
        """
        scope = self._self_scope()
        dated_before = None
        if from_before:
            dated_before = min(
                (record.created_at for record in self._memory_store.list_for_character(scope)),
                default=self._now(),
            )
        self._rewrite(
            scope, summaries, edited_from, "character_statement", dated_before, carry_over=True
        )
        self._keep_the_newest_self_memories()

    def self_memories(self, *, in_conversation: Any = _ANY) -> list[str]:
        """What she said about herself that she holds, oldest first.

        Without ``in_conversation``, all of it, wherever it was said: what a
        host's memory page shows. With it (None is the conversation of a host
        that names none), what is in her mind in that conversation: what is
        hers in every conversation, and what she said there about what she is
        doing, plans to do and thinks of the user."""
        records = self._memory_store.list_for_character(self._self_scope())
        active = sorted(
            (
                record
                for record in records
                if record.is_active
                and (in_conversation is _ANY or self._in_mind(record, in_conversation))
            ),
            key=lambda record: record.created_at,
        )
        return [one_line(record.summary) for record in active]

    def _shown(self, scope: str) -> list[str]:
        """The lines a memory page shows of ``scope``, oldest first."""
        records = self._memory_store.list_for_character(scope)
        active = sorted((r for r in records if r.is_active), key=lambda r: r.created_at)
        return [one_line(record.summary) for record in active]

    def _rewrite(
        self,
        scope: str,
        summaries: Sequence[str],
        edited_from: Sequence[str] | None,
        evidence_type: str,
        dated_before: datetime | None = None,
        *,
        carry_over: bool = False,
    ) -> None:
        """``carry_over``: a new line that took the place of a line that is
        gone keeps that line's kind and conversation."""
        store = self._memory_store
        wanted = list(dict.fromkeys(one_line(line) for line in summaries if line.strip()))
        removable = None if edited_from is None else {one_line(line) for line in edited_from}
        records = []
        forgotten: dict[str, MemoryRecord] = {}
        gone: set[str] = set()
        for record in store.list_for_character(scope):
            shown = one_line(record.summary)
            if record.is_active and shown not in wanted and (removable is None or shown in removable):
                forgotten.setdefault(shown, record)
                gone.add(record.id)
                record = replace(record, status="forgotten", forgotten_at=self._now())
            records.append(record)
        # A fact that replaced another, forgotten: the one it replaced is held
        # again. A wrong replacement is undone by removing the newer line.
        records = [
            replace(record, status="active", superseded_by=None)
            if record.status == "superseded" and record.superseded_by in gone
            else record
            for record in records
        ]
        remembered = {one_line(record.summary) for record in records if record.is_active}
        new = [summary for summary in wanted if summary not in remembered]
        edited = (
            _edited_lines(self._shown(scope) if edited_from is None else edited_from, wanted)
            if carry_over
            else {}
        )
        for number, summary in enumerate(new):
            kind = "fact"
            metadata: dict[str, Any] = {"evidence_type": evidence_type, "source": "host"}
            was = forgotten.get(edited.get(summary, ""))
            if was is not None:
                # An edited line is still the kind of thing it was and holds
                # where it did: an edited plan for one conversation written
                # back as a fact was in every conversation.
                kind = self_memory_kind(was.kind)
                if CONVERSATION_KEY in was.metadata:
                    metadata[CONVERSATION_KEY] = was.metadata[CONVERSATION_KEY]
            record = MemoryRecord(
                character_id=scope,
                summary=summary,
                kind=kind,
                importance=0.7,
                metadata=metadata,
                created_at=self._now(),
            )
            if dated_before is not None:
                earlier = timedelta(seconds=len(new) - number)
                record = replace(record, created_at=dated_before - earlier)
            records.append(record)
        store.replace_for_character(scope, records)
