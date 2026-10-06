from __future__ import annotations

import asyncio
import difflib
import json
import math
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping, Sequence

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals.models import MOTIVATION_SOURCE_TYPES, MotivationKind
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.context.builder import is_turn_context
from ai_character_engine.memory.evidence import classify_user_text
from ai_character_engine.memory.self_kinds import SELF_MEMORY_KINDS, kind_of_what_she_said
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.state.mood import (
    CHARACTER_MOODS,
    DEFAULT_MOOD_FLOOR,
    DEFAULT_MOOD_HALF_LIFE_SECONDS,
    MOOD_SYNONYMS,
    effective_mood,
    mood_intensity,
)
from ai_character_engine.tasks.errors import (
    TaskQueueFullError,
    TaskRuntimeClosedError,
    UnknownTaskError,
)
from ai_character_engine.tasks.models import (
    TaskContext,
    TaskOutput,
    TaskPriority,
    TaskProposal,
    TaskStatus,
)
from ai_character_engine.tasks.runtime import MultiTaskRuntime, TaskHandle

from .models import CognitiveRole, CognitiveRouteRequirements
from .runtime import CognitiveModelRuntime


class BackgroundCognitionKind(str, Enum):
    MEMORY_EXTRACTION = "memory_extraction"
    EMOTION_ANALYSIS = "emotion_analysis"
    CONVERSATION_SUMMARY = "conversation_summary"
    REFLECTION = "reflection"
    GOAL_MOTIVATION = "goal_motivation"
    VISION_INTERPRETATION = "vision_interpretation"
    SELF_MEMORY_EXTRACTION = "self_memory_extraction"
    CHARACTER_MOOD = "character_mood"
    REPLY_CHECK = "reply_check"
    USER_STATE = "user_state"
    DIARY = "diary"


_ROLE_BY_KIND: dict[BackgroundCognitionKind, CognitiveRole] = {
    BackgroundCognitionKind.MEMORY_EXTRACTION: CognitiveRole.MEMORY,
    BackgroundCognitionKind.EMOTION_ANALYSIS: CognitiveRole.EMOTION,
    BackgroundCognitionKind.CONVERSATION_SUMMARY: CognitiveRole.SUMMARY,
    BackgroundCognitionKind.REFLECTION: CognitiveRole.REFLECTION,
    BackgroundCognitionKind.GOAL_MOTIVATION: CognitiveRole.GOAL,
    BackgroundCognitionKind.VISION_INTERPRETATION: CognitiveRole.VISION,
    BackgroundCognitionKind.SELF_MEMORY_EXTRACTION: CognitiveRole.SELF_MEMORY,
    BackgroundCognitionKind.CHARACTER_MOOD: CognitiveRole.MOOD,
    BackgroundCognitionKind.REPLY_CHECK: CognitiveRole.REPLY_CHECK,
    BackgroundCognitionKind.USER_STATE: CognitiveRole.USER_STATE,
    BackgroundCognitionKind.DIARY: CognitiveRole.DIARY,
}

SELF_MEMORY_TARGET = "memory.self_candidate"
MOOD_TARGET = "state.mood_candidate"
# A note for her next reply about a slip in this one. Nothing is written for
# it: CharacterCompanion keeps the note for that one reply.
REPLY_NOTE_TARGET = "context.reply_note_candidate"
# What the reply check looks for, and nothing else.
REPLY_CHECK_KINDS: tuple[str, ...] = (
    "broke_character",
    "leaked_markup",
    "off_persona",
    "repeated",
    "wrong_language",
)
# A fix is one short sentence; longer is cut.
REPLY_FIX_CHARS = 40
_REPLY_ISSUES_KEPT = 2
# How the user has been lately: written over what was there before.
USER_STATE_TARGET = "state.user_state_candidate"
USER_ENERGY_LEVELS: tuple[str, ...] = ("low", "normal", "high")
USER_MOOD_TRENDS: tuple[str, ...] = ("down", "flat", "up")
# A concern is a few words; at most this many, each with the user's own words.
USER_CONCERN_CHARS = 30
USER_CONCERNS_KEPT = 3
# Her day, in her words. Nothing is written for it by the coordinator:
# CharacterCompanion keeps her diary.
DIARY_TARGET = "memory.diary_candidate"
DIARY_SENTENCES = 6
_EVIDENCE_KEPT = 3
# A stage direction between asterisks describes what she does; it is not
# something she said about herself.
_STAGE_DIRECTION = re.compile(r"\*[^*\n]*\*")
# What may follow the question mark that ends a quoted question.
_CLOSING_MARKS = "」』）)】\"'”’ "


@dataclass(frozen=True, slots=True)
class BackgroundWorkerSpec:
    kind: BackgroundCognitionKind
    enabled: bool = True
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_s: float = 20.0
    every_n_revisions: int = 1
    max_concurrency: int = 1
    requirements: CognitiveRouteRequirements | None = None

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("background cognition timeout_s must be > 0")
        if self.every_n_revisions < 1:
            raise ValueError("every_n_revisions must be >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


@dataclass(frozen=True, slots=True)
class BackgroundCognitionConfig:
    worker_specs: tuple[BackgroundWorkerSpec, ...] = field(
        default_factory=lambda: (
            BackgroundWorkerSpec(
                BackgroundCognitionKind.MEMORY_EXTRACTION,
                priority=TaskPriority.HIGH,
                timeout_s=20.0,
            ),
            BackgroundWorkerSpec(
                BackgroundCognitionKind.EMOTION_ANALYSIS,
                priority=TaskPriority.HIGH,
                timeout_s=12.0,
            ),
            BackgroundWorkerSpec(
                BackgroundCognitionKind.CONVERSATION_SUMMARY,
                priority=TaskPriority.NORMAL,
                timeout_s=20.0,
                every_n_revisions=3,
            ),
            BackgroundWorkerSpec(
                BackgroundCognitionKind.REFLECTION,
                priority=TaskPriority.LOW,
                timeout_s=30.0,
                every_n_revisions=5,
            ),
            BackgroundWorkerSpec(
                BackgroundCognitionKind.GOAL_MOTIVATION,
                priority=TaskPriority.NORMAL,
                timeout_s=25.0,
                every_n_revisions=1,
            ),
            BackgroundWorkerSpec(
                BackgroundCognitionKind.VISION_INTERPRETATION,
                priority=TaskPriority.NORMAL,
                timeout_s=20.0,
            ),
        )
    )
    history_messages: int = 12
    event_history: int = 512

    def __post_init__(self) -> None:
        if self.history_messages < 1:
            raise ValueError("history_messages must be >= 1")
        if self.event_history < 1:
            raise ValueError("event_history must be >= 1")
        kinds = [spec.kind for spec in self.worker_specs]
        if len(set(kinds)) != len(kinds):
            raise ValueError("background cognition worker kinds must be unique")

    def spec_for(self, kind: BackgroundCognitionKind) -> BackgroundWorkerSpec | None:
        for spec in self.worker_specs:
            if spec.kind is kind:
                return spec
        return None


@dataclass(frozen=True, slots=True)
class BackgroundCognitionEvent:
    kind: BackgroundCognitionKind
    action: str
    revision: int
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    foreground_event_id: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class _Reading:
    """How far the workers have read one conversation."""

    # The opening of the last turn whose user lines memory extraction has been
    # given, and of the last whose replies self-memory extraction has been
    # given. None: nothing of the conversation that begins with first_line.
    read_up_to: Message | None = None
    said_up_to: Message | None = None
    # The same for the user state: how the user has been since its last run.
    state_up_to: Message | None = None
    first_line: Message | None = None
    known: bool = False
    turns: int = 0


@dataclass(frozen=True, slots=True)
class BackgroundCognitionResult:
    kind: BackgroundCognitionKind
    value: Any
    confidence: float | None
    evidence: tuple[str, ...]
    proposal_targets: tuple[str, ...]
    model: str | None
    base_revision: int


class StructuredBackgroundWorker:
    """One standardized background cognition worker.

    The worker receives only a TaskSnapshot and provider-neutral text payload.
    It returns TaskOutput/TaskProposal values and has no CharacterRuntime,
    MemoryManager, CharacterState, or renderer reference to mutate.
    """

    def __init__(
        self,
        *,
        models: CognitiveModelRuntime,
        spec: BackgroundWorkerSpec,
        history_messages: int,
    ) -> None:
        self.models = models
        self.spec = spec
        self.history_messages = history_messages
        self._semaphore = asyncio.Semaphore(spec.max_concurrency)

    async def __call__(self, context: TaskContext) -> TaskOutput:
        async with self._semaphore:
            messages = self._messages(context)
            try:
                response = await self.models.generate(
                    _ROLE_BY_KIND[self.spec.kind],
                    messages,
                    requirements=self.spec.requirements,
                )
                data = _parse_json_object(response.text)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.spec.kind is not BackgroundCognitionKind.REPLY_CHECK:
                    raise
                # Her opening said again is read off her words: it needs no
                # model, and is pointed out when the model fails or is missing.
                response, data = LLMResponse(text="", model=None), {}
            if self.spec.kind is BackgroundCognitionKind.DIARY:
                data = await self._diary_again(context, messages, response.text, data)
            value, confidence, evidence, proposals = self._normalize(context, data)
            if self.spec.kind is BackgroundCognitionKind.REPLY_CHECK and proposals:
                value, evidence, proposals = await self._confirmed_slips(
                    context, data, proposals[0]
                )
            return TaskOutput(
                value=BackgroundCognitionResult(
                    kind=self.spec.kind,
                    value=value,
                    confidence=confidence,
                    evidence=evidence,
                    proposal_targets=tuple(proposal.target for proposal in proposals),
                    model=response.model,
                    base_revision=context.snapshot.revision,
                ),
                proposals=tuple(proposals),
                metadata={
                    "background_cognition": {
                        "kind": self.spec.kind.value,
                        "role": _ROLE_BY_KIND[self.spec.kind].value,
                        "base_revision": context.snapshot.revision,
                        "confidence": confidence,
                        "evidence": list(evidence),
                    },
                    "model": response.model,
                    "cognitive": dict(response.metadata.get("cognitive", {})),
                },
            )

    def _messages(self, context: TaskContext) -> list[Message]:
        if self.spec.kind is BackgroundCognitionKind.CHARACTER_MOOD:
            return _mood_messages(context, self.history_messages)
        if self.spec.kind is BackgroundCognitionKind.REPLY_CHECK:
            return _reply_check_messages(context, self.history_messages)
        if self.spec.kind is BackgroundCognitionKind.USER_STATE:
            return _user_state_messages(context, self.spec.every_n_revisions)
        if self.spec.kind is BackgroundCognitionKind.DIARY:
            return _diary_messages(context)
        history = list(context.snapshot.history[-self.history_messages :])
        payload = context.request.payload
        latest = str(payload.get("latest_user_or_event", "")).strip()
        assistant = str(payload.get("assistant_response", "")).strip()
        transcript_title = "Recent transcript"
        if self.spec.kind is BackgroundCognitionKind.EMOTION_ANALYSIS:
            # Nothing the character said is evidence of how the user feels or
            # treats her, and a small model cannot keep the two apart: with a
            # character that insults the user it rated his plain questions as
            # hostile. It is given his side of the conversation only. The
            # latest line stands apart from the earlier ones: given together,
            # a 9B model read a plain question after an angry turn as angry.
            history = _earlier_user_lines(history)
            assistant = ""
            transcript_title = _EARLIER_USER_LINES
        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in history if message.content.strip()
        )
        vision = str(payload.get("vision_observation", "")).strip()
        goal_sources = payload.get("goal_sources")
        system = _SYSTEM_PROMPTS[self.spec.kind]
        user = (
            f"Character: {context.snapshot.character_name}\n"
            f"Authoritative snapshot revision: {context.snapshot.revision}\n"
            f"{transcript_title}:\n{transcript or '(empty)'}\n\n"
            f"Latest event/user content:\n{latest or '(none)'}\n"
        )
        if self.spec.kind is not BackgroundCognitionKind.EMOTION_ANALYSIS:
            user += f"\nForeground assistant response:\n{assistant or '(none)'}\n"
        if self.spec.kind is BackgroundCognitionKind.MEMORY_EXTRACTION:
            # One run is responsible for every turn since the previous run.
            lines = _user_lines(context, turns=self.spec.every_n_revisions)
            user += "\nUser lines to extract from:\n" + "\n".join(
                f"- {line}" for line in lines
            ) + "\n"
        if self.spec.kind is BackgroundCognitionKind.SELF_MEMORY_EXTRACTION:
            lines = _assistant_lines(context, turns=self.spec.every_n_revisions)
            user += "\nCharacter lines to extract from:\n" + "\n".join(
                f"- {line}" for line in lines
            ) + "\n"
        if vision:
            user += f"\nVision observation:\n{vision}\n"
        if self.spec.kind is BackgroundCognitionKind.GOAL_MOTIVATION and isinstance(goal_sources, dict):
            user += (
                "\nAvailable authoritative goal sources (use only these ids; raw reflections are intentionally absent):\n"
                + json.dumps(goal_sources, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        language = str(payload.get("output_language") or "").strip()
        user += "\nReturn only the requested JSON object." + (
            f" Text values must be written in {language}." if language else _OUTPUT_LANGUAGE_RULE
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    async def _diary_again(
        self,
        context: TaskContext,
        messages: list[Message],
        answered: str,
        data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Her entry asked for once more when sentences of it tell what is not
        in what happened, or it speaks to the user: a 9B model wrote one entry
        in three from her persona or as a reply to the user. The first answer
        stands when the second is none."""
        day = context.request.payload.get("diary")
        happened = [
            line
            for _, lines in _diary_sections(day if isinstance(day, Mapping) else {})
            for line in lines
        ]
        text = _diary_text(str(data.get("text") or ""))
        made_up = _not_of_the_day(text, happened)
        speaks_to_the_user = _SPEAKS_TO_THE_USER.search(_QUOTED_WORDS.sub(" ", text)) is not None
        if not text or (not made_up and not speaks_to_the_user):
            return data
        ask = []
        if made_up:
            ask.append(
                "Some sentences of that entry tell what is not in what happened:\n"
                + "\n".join(f"- {sentence}" for sentence in made_up)
            )
        ask.append(
            "Write the entry again: tell only what is in what happened, in the "
            "character's own voice. The diary is for the character alone: write of the "
            'user as he, she or by name, never as "you" (你, 您, あなた). Return only the '
            "JSON object; evidence is copied exactly from what happened."
        )
        try:
            response = await self.models.generate(
                _ROLE_BY_KIND[self.spec.kind],
                [
                    *messages,
                    Message(role="assistant", content=answered),
                    Message(role="user", content="\n\n".join(ask)),
                ],
                requirements=self.spec.requirements,
            )
            again = _parse_json_object(response.text)
        except asyncio.CancelledError:
            raise
        except Exception:
            return data
        return again if str(again.get("text") or "").strip() else data

    async def _confirmed_slips(
        self, context: TaskContext, data: Mapping[str, Any], proposal: TaskProposal
    ) -> tuple[tuple[dict[str, str], ...], tuple[str, ...], list[TaskProposal]]:
        """The slips the words cannot show, asked about once more one by one:
        a 9B model quoted a line of her persona against nearly every reply,
        and asked about one sentence and one fact it answered no."""
        against_of = {
            (str(raw.get("kind") or "").strip().casefold(), str(raw.get("evidence") or "").strip()):
            str(raw.get("against") or "").strip()
            for raw in data.get("issues") or ()
            if isinstance(raw, dict)
        }
        persona = str(context.request.payload.get("character_persona") or "")
        confirmed = []
        for item in proposal.payload["issues"]:
            if item["kind"] in _ASKED_AGAIN:
                question = _slip_question(
                    item, against_of.get((item["kind"], item["evidence"]), ""), persona
                )
                against = against_of.get((item["kind"], item["evidence"]), "")
                try:
                    response = await self.models.generate(
                        _ROLE_BY_KIND[self.spec.kind],
                        question,
                        requirements=self.spec.requirements,
                    )
                    answer = _parse_json_object(response.text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # No clear answer is no slip.
                    continue
                if not _slip_confirmed(item, against, answer):
                    continue
            confirmed.append(item)
        evidence = tuple(item["evidence"] for item in confirmed)
        if not confirmed:
            return (), evidence, []
        return (
            tuple(confirmed),
            evidence,
            [
                replace(
                    proposal,
                    payload={"issues": confirmed},
                    provenance={**proposal.provenance, "evidence": list(evidence)},
                )
            ],
        )

    def _normalize(
        self,
        context: TaskContext,
        data: Mapping[str, Any],
    ) -> tuple[Any, float | None, tuple[str, ...], list[TaskProposal]]:
        kind = self.spec.kind
        confidence = _confidence(data.get("confidence"))
        evidence = _evidence(data.get("evidence"))
        provenance = {
            "worker_kind": kind.value,
            "foreground_event_id": context.request.payload.get("foreground_event_id"),
            "foreground_event_type": context.request.payload.get("foreground_event_type"),
            "foreground_event_source": context.request.payload.get("foreground_event_source"),
            "foreground_event_content": context.request.payload.get("latest_user_or_event"),
            "evidence_type": context.request.payload.get("memory_evidence_type"),
            "evidence": list(evidence),
        }
        proposals: list[TaskProposal] = []

        if kind is BackgroundCognitionKind.MEMORY_EXTRACTION:
            items = data.get("items", [])
            if not isinstance(items, list):
                raise ValueError("memory extraction response.items must be an array")
            normalized: list[dict[str, Any]] = []
            user_lines = _user_lines(context, turns=self.spec.every_n_revisions)
            her_lines = _her_lines(context, self.history_messages)
            for raw in items[:8]:
                if not isinstance(raw, dict):
                    continue
                summary = str(raw.get("summary", "")).strip()
                if not summary:
                    continue
                # A quote the user never said means the fact came from the
                # assistant or from nowhere; it must not become a memory. No
                # quote at all says the same: the prompt asks for one.
                quote = str(raw.get("evidence") or "")
                source = _line_quoted(quote, user_lines)
                # Judge the line the fact came from, not the latest turn: a
                # question after "my name is Dawn" must not veto the name, and
                # a fact after a quotation must not launder the quotation.
                if source is None or classify_user_text(source) != "asserted_fact":
                    continue
                if _echoes(quote, her_lines):
                    # Words she said first, said after her: a sentence she is
                    # teaching, practised. "私はカラフルが好きです。" said back
                    # to her is not the user liking colourful things.
                    continue
                item_provenance = {
                    **provenance,
                    "evidence": [quote],
                    "evidence_type": "asserted_fact",
                }
                item_conf = _confidence(raw.get("confidence"))
                if item_conf is None:
                    item_conf = confidence
                item = {
                    "summary": summary,
                    "kind": str(raw.get("kind", "event")).strip() or "event",
                    "importance": _unit_float(raw.get("importance"), default=0.5),
                }
                normalized.append(item)
                proposals.append(
                    context.proposal(
                        "memory.append_candidate",
                        item,
                        confidence=item_conf,
                        provenance=item_provenance,
                    )
                )
            return tuple(normalized), confidence, evidence, proposals

        if kind is BackgroundCognitionKind.SELF_MEMORY_EXTRACTION:
            items = data.get("items", [])
            if not isinstance(items, list):
                raise ValueError("self memory extraction response.items must be an array")
            normalized = []
            said = _assistant_lines(context, turns=self.spec.every_n_revisions)
            for raw in items[:8]:
                if not isinstance(raw, dict):
                    continue
                summary = str(raw.get("summary", "")).strip()
                quote = str(raw.get("evidence") or "").strip()
                # Checked like a memory of the user: a quote she never said
                # means the fact came from the user or from nowhere. A question
                # states nothing about her.
                if not summary or not said_in(quote, said):
                    continue
                if quote.rstrip(_CLOSING_MARKS).endswith(("?", "？")):
                    continue
                item_conf = _confidence(raw.get("confidence"))
                if item_conf is None:
                    item_conf = confidence
                item = {
                    "summary": summary,
                    # One of SELF_MEMORY_KINDS when the model meant one of
                    # them; what she is doing in this conversation, and what
                    # she thinks of the user, are told apart by it.
                    "kind": kind_of_what_she_said(raw.get("kind"), summary),
                    "importance": _unit_float(raw.get("importance"), default=0.5),
                }
                normalized.append(item)
                proposals.append(
                    context.proposal(
                        SELF_MEMORY_TARGET,
                        item,
                        confidence=item_conf,
                        provenance={
                            **provenance,
                            "evidence": [quote],
                            "evidence_type": "character_statement",
                        },
                    )
                )
            return tuple(normalized), confidence, evidence, proposals

        if kind is BackgroundCognitionKind.CHARACTER_MOOD:
            # One answer repeated the same quote in the evidence list until
            # the token limit; items past the first 3 are ignored, not a
            # reason to reject the reading.
            evidence = evidence[:3]
            mood = str(data.get("mood") or "").strip().lower()
            # A near word ("relieved", "annoyed") stands for the mood it means.
            mood = MOOD_SYNONYMS.get(mood, mood)
            raw = _number(data.get("intensity"))
            if mood not in CHARACTER_MOODS or raw is None or confidence is None:
                # Any other word ("開心", "nostalgic") or a missing field is no
                # reading of her mood; her mood stays what it was.
                return None, confidence, evidence, []
            value = {"mood": mood, "intensity": mood_intensity(mood, raw)}
            proposals.append(
                context.proposal(
                    MOOD_TARGET,
                    value,
                    confidence=confidence,
                    provenance={
                        **provenance,
                        "evidence": list(evidence),
                        "turn_ended_at": context.snapshot.captured_at.timestamp(),
                    },
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.REPLY_CHECK:
            raw_issues = data.get("issues")
            previous, reply, latest = _replies_around(context, self.history_messages)
            payload = context.request.payload
            persona = str(payload.get("character_persona") or "")
            said_by_user = latest if payload.get("foreground_event_type") == "user_message" else ""
            fix_script = _fix_script(str(payload.get("output_language") or ""), said_by_user)
            issues: list[dict[str, str]] = []
            # Her opening said again is read off the words; the model never
            # reported it. It comes first, and stands for the model's own.
            opening = _same_opening(reply, previous)
            if opening:
                fix = _opening_fix(fix_script, payload)
                issues.append({"kind": "repeated", "evidence": opening, "fix": fix})
            for raw in raw_issues if isinstance(raw_issues, list) else ():
                if not isinstance(raw, dict):
                    continue
                issue_kind = str(raw.get("kind") or "").strip().casefold()
                quote = str(raw.get("evidence") or "").strip()
                fix = str(raw.get("fix") or "").strip()[:REPLY_FIX_CHARS].strip()
                # A kind off the list is something this check does not judge;
                # a quote she did not say in this reply, or no fix, is no slip
                # she can do anything about.
                if issue_kind not in REPLY_CHECK_KINDS or not fix:
                    continue
                if opening and issue_kind == "repeated":
                    continue
                # A fix she cannot read in the language of the conversation is
                # no help to her.
                if fix_script and not _written_in(fix, fix_script):
                    continue
                if _line_quoted(quote, [reply]) is None:
                    continue
                against = str(raw.get("against") or "").strip()
                if not _slip_holds(
                    issue_kind,
                    quote,
                    against,
                    reply=reply,
                    previous=previous,
                    persona=persona,
                    said_by_user=said_by_user,
                ):
                    continue
                issues.append({"kind": issue_kind, "evidence": quote, "fix": fix})
            issues = issues[:_REPLY_ISSUES_KEPT]
            evidence = tuple(item["evidence"] for item in issues)
            if not issues:
                return (), confidence, evidence, []
            proposals.append(
                context.proposal(
                    REPLY_NOTE_TARGET,
                    {"issues": issues},
                    # Every slip is quoted from her reply, checked above.
                    confidence=1.0,
                    provenance={**provenance, "evidence": list(evidence), "reply": reply},
                )
            )
            return tuple(issues), confidence, evidence, proposals

        if kind is BackgroundCognitionKind.USER_STATE:
            energy = str(data.get("energy") or "").strip().casefold()
            trend = str(data.get("mood_trend") or "").strip().casefold()
            if energy not in USER_ENERGY_LEVELS or trend not in USER_MOOD_TRENDS:
                # Any other word is no reading; what was there stays.
                return None, confidence, (), []
            said = _state_lines(context, turns=self.spec.every_n_revisions)
            hers = _her_lines(context, self.history_messages)
            script = _fix_script(
                str(context.request.payload.get("output_language") or ""), " ".join(said)
            )

            def the_users(quote: str) -> bool:
                # The user's own words, not a sentence of hers said after her.
                return _line_quoted(quote, said) is not None and not _echoes(quote, hers)

            concerns: list[str] = []
            quotes: list[str] = []
            raw_concerns = data.get("concerns")
            for raw in raw_concerns if isinstance(raw_concerns, list) else ():
                if not isinstance(raw, dict):
                    continue
                concern = " ".join(str(raw.get("concern") or "").split())
                concern = concern[:USER_CONCERN_CHARS].strip()
                quote = str(raw.get("evidence") or "").strip()
                # A concern the user did not voice, a diagnosis, or one she
                # cannot read in the language of the conversation is dropped.
                if not concern or not the_users(quote) or _DIAGNOSIS.search(concern):
                    continue
                if script and not _written_in(concern, script):
                    continue
                if concern in concerns or len(concerns) >= USER_CONCERNS_KEPT:
                    continue
                concerns.append(concern)
                quotes.append(quote)
            quotes += [quote for quote in evidence if the_users(quote)]
            evidence = tuple(dict.fromkeys(quotes))[:_EVIDENCE_KEPT]
            if not evidence:
                # Nothing the user said shows it: how the user has been is
                # not known, so it is ordinary.
                energy, trend = "normal", "flat"
            value = {
                "energy": energy,
                "mood_trend": trend,
                "concerns": concerns,
                "evidence": list(evidence),
            }
            proposals.append(
                context.proposal(
                    USER_STATE_TARGET,
                    value,
                    # Every part is held to the user's own words, checked above.
                    confidence=1.0,
                    provenance={**provenance, "evidence": list(evidence)},
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.DIARY:
            day = context.request.payload.get("diary")
            day = day if isinstance(day, Mapping) else {}
            happened = [line for _, lines in _diary_sections(day) for line in lines]
            text = _diary_text(str(data.get("text") or ""))
            script = _fix_script(
                str(context.request.payload.get("output_language") or ""),
                " ".join(line for title, lines in _diary_sections(day) if title != _MOODS
                         for line in lines),
            )
            evidence = tuple(
                dict.fromkeys(
                    quote
                    for quote in evidence
                    if len(_letters(quote)) >= _SHORTEST_ECHO
                    and _line_quoted(quote, happened) is not None
                )
            )[:_EVIDENCE_KEPT]
            # What is still made up after she was asked once more is left out.
            text = _told_of_the_day(text, happened)
            # A day told with nothing of the day under it is made up; one in a
            # language she does not speak there is no diary of hers.
            if not text or not evidence or (script and not _written_in(text, script)):
                return None, confidence, evidence, []
            value = {
                "date": str(day.get("date") or ""),
                "text": text,
                "evidence": list(evidence),
                "conversation_ids": list(day.get("conversation_ids") or ()),
                "until": day.get("until"),
            }
            proposals.append(
                context.proposal(
                    DIARY_TARGET,
                    value,
                    confidence=1.0,
                    provenance={**provenance, "evidence": list(evidence)},
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.EMOTION_ANALYSIS:
            emotion = str(data.get("emotion", "neutral")).strip() or "neutral"
            value = {
                "emotion": emotion,
                "intensity": _unit_float(data.get("intensity"), default=0.5),
            }
            # Optional signed scores give host state policies something to act
            # on without parsing a free-form, any-language emotion label.
            for key in ("valence", "stance"):
                score = _signed_unit_float(data.get(key))
                if score is not None:
                    value[key] = score
            proposals.append(
                context.proposal(
                    "state.emotion_candidate",
                    value,
                    confidence=confidence,
                    # Dated by its turn: a reading of her mood from the same
                    # turn stands against the rules (state.relationship).
                    provenance={
                        **provenance,
                        "turn_ended_at": context.snapshot.captured_at.timestamp(),
                    },
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.CONVERSATION_SUMMARY:
            summary = str(data.get("summary", "")).strip()
            if not summary:
                raise ValueError("conversation summary response.summary must not be empty")
            value = {"summary": summary}
            proposals.append(
                context.proposal(
                    "memory.conversation_summary_candidate",
                    value,
                    confidence=confidence,
                    provenance=provenance,
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.REFLECTION:
            insight = str(data.get("insight", "")).strip()
            if not insight:
                raise ValueError("reflection response.insight must not be empty")
            raw_claim = data.get("belief_candidate")
            claim: dict[str, str] | None
            if raw_claim is None:
                claim = None
            elif isinstance(raw_claim, dict):
                required = ("subject", "predicate", "object")
                values = {key: str(raw_claim.get(key, "")).strip() for key in required}
                if not all(values.values()):
                    raise ValueError(
                        "reflection belief_candidate must contain non-empty subject, predicate, and object"
                    )
                claim = values
            else:
                raise ValueError("reflection belief_candidate must be an object or null")
            value = {"insight": insight, "belief_candidate": claim}
            # Evidence about the user is what the user said, word for word.
            # Her own lines ("you drive me mad!") and retellings ("the user
            # said twice that they do not understand") are not; a thought
            # with none left is not proposed.
            said_by_user = _lines_the_user_said(context, self.history_messages)
            quoted = [
                (quote, line)
                for quote in evidence
                if (line := _line_quoted(quote, said_by_user)) is not None
            ]
            evidence = tuple(quote for quote, _ in quoted)
            if not quoted:
                return value, confidence, evidence, []
            proposals.append(
                context.proposal(
                    "cognition.reflection_candidate",
                    value,
                    confidence=confidence,
                    provenance={
                        **provenance,
                        "evidence": list(evidence),
                        # Each quote judged by the line it came from, not by
                        # the latest line of the user.
                        "evidence_types": [classify_user_text(line) for _, line in quoted],
                    },
                )
            )
            return value, confidence, evidence, proposals

        if kind is BackgroundCognitionKind.GOAL_MOTIVATION:
            raw_goals = data.get("goals", [])
            if not isinstance(raw_goals, list):
                raise ValueError("goal motivation response.goals must be an array")
            normalized: list[dict[str, Any]] = []
            supplied = _supplied_goal_sources(context.request.payload.get("goal_sources"))
            for raw in raw_goals[:6]:
                if not isinstance(raw, dict):
                    continue
                objective = str(raw.get("objective", "")).strip()
                if not objective:
                    continue
                horizon = str(raw.get("horizon", "short_term")).strip() or "short_term"
                if horizon not in {"short_term", "long_term"}:
                    raise ValueError("goal horizon must be short_term or long_term")
                urgency = _unit_float(raw.get("urgency"), default=0.5)
                conflict_key_raw = raw.get("conflict_key")
                conflict_key = None if conflict_key_raw is None else str(conflict_key_raw).strip() or None
                raw_signals = raw.get("motivation_signals", [])
                if not isinstance(raw_signals, list) or not raw_signals:
                    raise ValueError("goal motivation_signals must be a non-empty array")
                signals: list[dict[str, Any]] = []
                for signal in raw_signals[:8]:
                    if not isinstance(signal, dict):
                        continue
                    kind_value = str(signal.get("kind", "")).strip()
                    source_type = str(signal.get("source_type", "")).strip()
                    source_id = str(signal.get("source_id", "")).strip()
                    rationale = str(signal.get("rationale", "")).strip()
                    if not all((kind_value, source_type, source_id, rationale)):
                        raise ValueError("goal motivation signal requires kind/source_type/source_id/rationale")
                    if supplied is not None and not _citable(
                        kind_value, source_type, source_id, supplied
                    ):
                        # The commit check refuses a goal over one bad source.
                        # What the model made up is removed here, so a goal
                        # that also has a real source is not lost with it.
                        continue
                    signals.append({
                        "kind": kind_value,
                        "strength": _unit_float(signal.get("strength"), default=0.5),
                        "source_type": source_type,
                        "source_id": source_id,
                        "rationale": rationale,
                    })
                if not signals and supplied is not None:
                    continue
                if not signals:
                    raise ValueError("goal requires at least one valid motivation signal")
                item = {
                    "objective": objective,
                    "horizon": horizon,
                    "urgency": urgency,
                    "conflict_key": conflict_key,
                    "motivation_signals": signals,
                }
                normalized.append(item)
                item_conf = _confidence(raw.get("confidence"))
                if item_conf is None:
                    item_conf = confidence
                proposals.append(
                    context.proposal(
                        "cognition.goal_candidate",
                        item,
                        confidence=item_conf,
                        provenance=provenance,
                    )
                )
            return tuple(normalized), confidence, evidence, proposals

        if kind is BackgroundCognitionKind.VISION_INTERPRETATION:
            interpretation = str(data.get("interpretation", "")).strip()
            if not interpretation:
                raise ValueError("vision interpretation response.interpretation must not be empty")
            value = {
                "interpretation": interpretation,
                "tags": tuple(str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()),
            }
            proposals.append(
                context.proposal(
                    "context.vision_interpretation_candidate",
                    value,
                    confidence=confidence,
                    provenance=provenance,
                )
            )
            return value, confidence, evidence, proposals

        raise AssertionError(f"unsupported background cognition kind: {kind}")


class BackgroundCognitionRuntime:
    """Schedules standard cognition workers after a successful foreground turn.

    ``run_foreground`` waits only for the authoritative CharacterRuntime turn and
    queue admission.  It never awaits model inference performed by background
    workers.  Results stay non-authoritative as TaskProposal values until an explicit commit policy accepts them.
    """

    def __init__(
        self,
        tasks: MultiTaskRuntime,
        models: CognitiveModelRuntime,
        *,
        config: BackgroundCognitionConfig | None = None,
    ) -> None:
        self.tasks = tasks
        self.models = models
        self.config = config or BackgroundCognitionConfig()
        self._events: deque[BackgroundCognitionEvent] = deque(maxlen=self.config.event_history)
        self._scheduled_keys: dict[tuple[BackgroundCognitionKind, str], None] = {}
        # The language the workers write in. Empty: the language the user
        # writes in, which the model has to work out from the transcript.
        self.output_language = ""
        # How her mood fades before a goal source reads it; a CharacterCompanion
        # sets these from its settings and clock. The spec's own defaults
        # otherwise (CompanionSettings.mood_half_life_seconds/mood_floor).
        self.mood_half_life_seconds = DEFAULT_MOOD_HALF_LIFE_SECONDS
        self.mood_floor = DEFAULT_MOOD_FLOOR
        self.clock: Callable[[], float] = time.time
        # The conversation the next turn belongs to, for a host that switches
        # the runtime's history between conversations. Each conversation is
        # then read on its own, and cadence counts its own turns: counted
        # together, two conversations taken in turns with every_n_revisions=2
        # left one of them unread. None: one conversation, cadence counts the
        # revisions of the runtime.
        self.conversation: Hashable | None = None
        self._readings: dict[Hashable | None, _Reading] = {}
        # What a host adds to the job of a worker, asked when the job is due
        # by cadence: a mapping merged into its payload, or None when there is
        # no job this time. The diary runs only with one (its day comes from
        # the host); the user state reads how the user seemed from one.
        self.job_extras: dict[
            BackgroundCognitionKind, Callable[[], Mapping[str, Any] | None]
        ] = {}
        self._handles: dict[str, TaskHandle] = {}
        self._task_kind: dict[str, BackgroundCognitionKind] = {}
        self._install_workers()

    @property
    def revision(self) -> int:
        return self.tasks.revision

    def events(self) -> tuple[BackgroundCognitionEvent, ...]:
        return tuple(self._events)

    def handles(self) -> tuple[TaskHandle, ...]:
        return tuple(self._handles.values())

    def proposal_is_stale(self, proposal: TaskProposal) -> bool:
        return proposal.is_stale(self.tasks.revision)

    async def run_foreground(self, event: CharacterEvent) -> CharacterRunResult:
        result = await self.tasks.run_foreground(event)
        await self.schedule_after_foreground(result)
        return result

    async def run_turn(self, user_message: str) -> CharacterRunResult:
        return await self.run_foreground(CharacterEvent.user_message(user_message))

    async def schedule_after_foreground(self, result: CharacterRunResult) -> tuple[TaskHandle, ...]:
        event = result.event
        revision = self.tasks.revision
        payload = self._payload_for(result)
        openers = [
            message
            for message in self.tasks.runtime.history
            if message.role in {"user", "event"}
        ]
        reading = self._readings.pop(self.conversation, None) or _Reading()
        self._readings[self.conversation] = reading
        while len(self._readings) > self.config.event_history:
            del self._readings[next(iter(self._readings))]
        reading.turns += 1
        turn = revision if self.conversation is None else reading.turns
        if not reading.known and openers:
            # Before the first turn seen here nothing is this runtime's to read.
            reading.read_up_to = openers[-2] if len(openers) > 1 else None
            reading.said_up_to = reading.read_up_to
            reading.state_up_to = reading.read_up_to
            reading.first_line = openers[0]
            reading.known = True
        handles: list[TaskHandle] = []
        for spec in self.config.worker_specs:
            if not spec.enabled:
                self._emit(spec.kind, "disabled", revision, event.id)
                continue
            if spec.kind is BackgroundCognitionKind.GOAL_MOTIVATION and getattr(self.tasks.runtime, "goal_manager", None) is None:
                self._emit(spec.kind, "not_applicable", revision, event.id, detail="goal_manager_not_configured")
                continue
            if spec.kind is BackgroundCognitionKind.VISION_INTERPRETATION and not _is_vision_event(event):
                self._emit(spec.kind, "not_applicable", revision, event.id)
                continue
            if spec.kind is BackgroundCognitionKind.DIARY and spec.kind not in self.job_extras:
                # Her day is the host's to give: what she remembers of it
                # lies in stores this runtime does not read.
                self._emit(spec.kind, "not_applicable", revision, event.id, detail="no_day_given")
                continue
            if turn % spec.every_n_revisions != 0:
                self._emit(spec.kind, "cadence_skipped", revision, event.id)
                continue
            key = (spec.kind, event.id)
            if key in self._scheduled_keys:
                self._emit(spec.kind, "deduplicated", revision, event.id)
                continue
            job = payload
            if spec.kind is BackgroundCognitionKind.MEMORY_EXTRACTION:
                start = _unread(reading, openers, reading.read_up_to)
                if start is not None:
                    job = {
                        **payload,
                        "user_lines_to_extract": [
                            message.content
                            for message in openers[start:]
                            if message.role == "user"
                        ],
                    }
            if spec.kind is BackgroundCognitionKind.SELF_MEMORY_EXTRACTION:
                start = _unread(reading, openers, reading.said_up_to)
                if start is not None:
                    job = {
                        **payload,
                        "assistant_lines_to_extract": _replies_from(
                            self.tasks.runtime.history, openers[start:]
                        ),
                    }
            if spec.kind in (
                BackgroundCognitionKind.CHARACTER_MOOD,
                BackgroundCognitionKind.REPLY_CHECK,
            ):
                persona = _persona_summary(
                    getattr(self.tasks.runtime, "character", None),
                    limit=_REPLY_CHECK_PERSONA_CHARS
                    if spec.kind is BackgroundCognitionKind.REPLY_CHECK
                    else _PERSONA_SUMMARY_CHARS,
                )
                if persona:
                    job = {**payload, "character_persona": persona}
            if spec.kind is BackgroundCognitionKind.USER_STATE:
                start = _unread(reading, openers, reading.state_up_to)
                if start is not None:
                    job = {
                        **job,
                        "user_lines_for_state": [
                            message.content
                            for message in openers[start:]
                            if message.role == "user"
                        ],
                    }
            if spec.kind is BackgroundCognitionKind.DIARY:
                persona = _persona_summary(
                    getattr(self.tasks.runtime, "character", None), limit=_DIARY_PERSONA_CHARS
                )
                if persona:
                    job = {**job, "character_persona": persona}
            extras = self.job_extras.get(spec.kind)
            if extras is not None:
                extra = extras()
                if extra is None:
                    self._emit(spec.kind, "not_due", revision, event.id)
                    continue
                job = {**job, **extra}
            try:
                handle = await self.tasks.submit_background(
                    spec.kind.value,
                    job,
                    priority=spec.priority,
                    timeout_s=spec.timeout_s,
                    source="background_cognition",
                )
            except (TaskQueueFullError, TaskRuntimeClosedError) as exc:
                # Foreground already committed successfully. Background admission is
                # best-effort and must never retroactively fail that authoritative turn.
                self._emit(
                    spec.kind,
                    "schedule_failed",
                    revision,
                    event.id,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                continue
            if spec.kind is BackgroundCognitionKind.MEMORY_EXTRACTION and openers:
                reading.read_up_to = openers[-1]
                reading.known = True
            if spec.kind is BackgroundCognitionKind.SELF_MEMORY_EXTRACTION and openers:
                reading.said_up_to = openers[-1]
                reading.known = True
            if spec.kind is BackgroundCognitionKind.USER_STATE and openers:
                reading.state_up_to = openers[-1]
                reading.known = True
            self._scheduled_keys[key] = None
            self._handles[handle.task_id] = handle
            self._task_kind[handle.task_id] = spec.kind
            # The newest are kept, like the events: a host that runs for days
            # schedules work on every turn.
            for kept in (self._scheduled_keys, self._handles, self._task_kind):
                while len(kept) > self.config.event_history:
                    del kept[next(iter(kept))]
            handles.append(handle)
            self._emit(spec.kind, "scheduled", revision, event.id, task_id=handle.task_id)
        return tuple(handles)

    async def submit_now(
        self, kind: BackgroundCognitionKind, extra: Mapping[str, Any]
    ) -> TaskHandle:
        """A job of ``kind`` outside a turn, for a host that asks for one (her
        diary before a stream): no event, ``extra`` as what the job reads."""
        spec = self.config.spec_for(kind)
        if spec is None or not spec.enabled:
            raise ValueError(f"no background worker for {kind.value}")
        job: dict[str, Any] = {"foreground_event_type": "host_request", **extra}
        if self.output_language.strip():
            job["output_language"] = self.output_language.strip()
        if kind is BackgroundCognitionKind.DIARY:
            persona = _persona_summary(
                getattr(self.tasks.runtime, "character", None), limit=_DIARY_PERSONA_CHARS
            )
            if persona:
                job["character_persona"] = persona
        handle = await self.tasks.submit_background(
            kind.value,
            job,
            priority=spec.priority,
            timeout_s=spec.timeout_s,
            source="background_cognition",
        )
        self._handles[handle.task_id] = handle
        self._task_kind[handle.task_id] = kind
        self._emit(kind, "scheduled", self.tasks.revision, task_id=handle.task_id)
        return handle

    async def collect(self, handle: TaskHandle) -> Any:
        try:
            result = await handle.wait()
        except UnknownTaskError:
            # Finished so long ago that the task runtime has let it go.
            self._handles.pop(handle.task_id, None)
            self._task_kind.pop(handle.task_id, None)
            return None
        kind = self._task_kind.get(handle.task_id)
        if kind is not None:
            self._emit(
                kind,
                result.status.value,
                self.tasks.revision,
                task_id=handle.task_id,
                detail=result.error,
            )
        return result

    async def collect_all(self) -> tuple[Any, ...]:
        handles = tuple(self._handles.values())
        if not handles:
            return ()
        results = await asyncio.gather(*(self.collect(handle) for handle in handles))
        return tuple(result for result in results if result is not None)

    def _install_workers(self) -> None:
        for spec in self.config.worker_specs:
            worker = StructuredBackgroundWorker(
                models=self.models,
                spec=spec,
                history_messages=self.config.history_messages,
            )
            self.tasks.register(spec.kind.value, worker)

    def _payload_for(self, result: CharacterRunResult) -> dict[str, Any]:
        event = result.event
        evidence_type = (
            classify_user_text(event.content)
            if event.type == "user_message"
            else "event_observation"
        )
        payload: dict[str, Any] = {
            "foreground_event_id": event.id,
            "foreground_event_type": event.type,
            "foreground_event_source": event.source,
            "latest_user_or_event": event.content,
            "assistant_response": result.text,
            "memory_evidence_type": evidence_type,
        }
        if self.output_language.strip():
            payload["output_language"] = self.output_language.strip()
        if _is_vision_event(event):
            payload["vision_observation"] = event.content
        payload["goal_sources"] = self._goal_sources_for(result, evidence_type=evidence_type)
        return payload

    def _goal_sources_for(self, result: CharacterRunResult, *, evidence_type: str) -> dict[str, Any]:
        runtime = self.tasks.runtime
        beliefs = []
        if getattr(runtime, "long_term_cognition", None) is not None:
            for record in runtime.long_term_cognition.active_beliefs(character_id=runtime.cognition_scope_id):
                beliefs.append({
                    "id": record.id,
                    "claim": {
                        "subject": record.claim.subject,
                        "predicate": record.claim.predicate,
                        "object": record.claim.object,
                    },
                    "confidence": record.confidence,
                    "support_count": record.support_count,
                })
        memories = [
            {
                "id": item.record.id,
                "summary": item.record.summary,
                "evidence_type": item.record.evidence_type,
                "score": item.score,
            }
            for item in result.retrieved_memories
            if item.record.is_active
        ]
        state = result.state_after
        mood, _ = effective_mood(
            state.emotion,
            state.mood_intensity,
            state.mood_updated_at,
            now=self.clock(),
            half_life_seconds=self.mood_half_life_seconds,
            floor=self.mood_floor,
        )
        state_values = [
            {"id": "emotion", "value": mood},
            {"id": "energy", "value": state.energy},
            {"id": "trust", "value": state.trust},
            {"id": "favorability", "value": state.favorability},
            {"id": "relationship_stage", "value": state.relationship_stage},
        ]
        state_values.extend(
            {"id": f"custom:{key}", "value": _json_safe(value)}
            for key, value in sorted(state.custom.items())
        )
        return {
            "event": {
                "id": result.event.id,
                "type": result.event.type,
                "source": result.event.source,
                "evidence_type": evidence_type,
                "content": result.event.content,
            },
            "active_beliefs": beliefs,
            "retrieved_memories": memories,
            "state": state_values,
        }

    def _emit(
        self,
        kind: BackgroundCognitionKind,
        action: str,
        revision: int,
        foreground_event_id: str | None = None,
        *,
        task_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._events.append(
            BackgroundCognitionEvent(
                kind=kind,
                action=action,
                revision=revision,
                task_id=task_id,
                foreground_event_id=foreground_event_id,
                detail=detail,
            )
        )


# Placed last in the user message: measured on a local 9B model, the same rule
# in the system prompt was ignored and summaries came back in English.
_OUTPUT_LANGUAGE_RULE = (
    " Text values must be written in the language the user writes in,"
    " not in English unless the user writes English."
)

_MOTIVATION_SOURCE_RULE = "; ".join(
    f"{kind.value} -> {'|'.join(sorted(source_types))}"
    for kind, source_types in MOTIVATION_SOURCE_TYPES.items()
)

_SYSTEM_PROMPTS: Mapping[BackgroundCognitionKind, str] = MappingProxyType(
    {
        BackgroundCognitionKind.MEMORY_EXTRACTION: (
            "Extract durable facts the user stated in the user lines to extract from, about themselves or "
            "their world: name, people and pets, work, projects, preferences, plans, and corrections of "
            "earlier facts. One item per fact. The transcript is context only: do not repeat facts from "
            "lines that are not listed for extraction, and never take a fact from the assistant's lines. "
            "Ignore quoted/reference text unless the user explicitly adopts it, and skip greetings, "
            "questions, examples and small talk that state no fact; return an empty items array only when "
            "the listed user lines state no fact at all. "
            "Each item's evidence must be one exact quote copied from the listed user lines. "
            "Each summary restates its evidence quote as a short sentence in the same language as that quote. "
            "Return JSON: {\"items\":[{\"summary\":str,\"kind\":str,\"importance\":0..1,\"confidence\":0..1,"
            "\"evidence\":str}],\"confidence\":0..1,\"evidence\":[str]}."
            # 1.2.0, added at the end only: what comes before is cached.
            " A sentence the user practises or repeats because the character asked (a sentence in a "
            "language being learned, words said after the character) says nothing about the user. "
            "A summary says only what its quote says: it adds no right or wrong, nothing learned or "
            "failed; that is the character's judgement. The summary is written in the language of "
            "its quote: a Chinese quote gets a Chinese summary, a Japanese quote a Japanese one, "
            "never an English one unless the quote is English."
        ),
        BackgroundCognitionKind.EMOTION_ANALYSIS: (
            "Infer the user's currently expressed emotion conservatively; do not diagnose hidden mental states. "
            "The transcript holds the user's side of the conversation only. valence is how pleasant the user feels (-1 unpleasant, 0 neutral, "
            "1 pleasant). stance is how the user is treating the character in these lines (-1 hostile or "
            "insulting, 0 neutral, 1 warm, caring or grateful); stance is 0 when the user is not addressing "
            "the character's person. "
            "Return JSON: {\"emotion\":str,\"intensity\":0..1,\"valence\":-1..1,\"stance\":-1..1,"
            "\"confidence\":0..1,\"evidence\":[str]}."
            # 1.2.0, added at the end only: what comes before is cached.
            " Judge only from the Latest event/user content. The earlier user lines are background: "
            "a feeling that shows only in earlier lines is not the user's emotion now. When the latest "
            "content shows no particular feeling, answer neutral with a low intensity, valence 0 and "
            "stance 0."
        ),
        BackgroundCognitionKind.CONVERSATION_SUMMARY: (
            "Summarize the recent conversation faithfully, preserving corrections and unresolved items. "
            "Return JSON: {\"summary\":str,\"confidence\":0..1,\"evidence\":[str]}."
        ),
        BackgroundCognitionKind.REFLECTION: (
            "Produce one cautious character-level reflection useful for future reasoning: what the character "
            "privately notices or concludes about the user or their relationship after this exchange. "
            "Write it from the character's own point of view in the first person. It is not advice about how "
            "an assistant should behave. Do not invent facts or commit state. "
            "A belief_candidate is only a structured hypothesis key/value, never an authoritative fact; use null when the "
            "insight should not become a long-term belief. Return JSON: {\"insight\":str,\"belief_candidate\":null|"
            "{\"subject\":str,\"predicate\":str,\"object\":str},\"confidence\":0..1,\"evidence\":[str]}."
            # 1.2.0, added at the end only: what comes before is cached.
            " Keep what was seen apart from the character's interpretation, and write the "
            "interpretation as the character's own feeling or thought, in the first person. Each "
            "evidence item is an exact quote of the user's own words from the transcript; the "
            "character's own lines are not evidence about the user. A mistake the user made while the "
            "character was teaching or correcting them is not evidence, unless the user admitted it. "
            "When the insight rests only on the character's judgement of the user, belief_candidate "
            "is null. The insight is never written in English unless the conversation, or the "
            "language asked for at the end of the user message, is English."
        ),
        BackgroundCognitionKind.GOAL_MOTIVATION: (
            "Propose zero or more durable character goals only when supported by the supplied authoritative source ids. "
            "Goals are action intentions, never facts. Do not infer truth from a desired outcome; do not reference raw reflections; "
            "do not create a planner or step-by-step action tree. Each motivation signal must cite one supplied event/memory/belief/state source. "
            "Allowed kind -> source_type: " + _MOTIVATION_SOURCE_RULE + ". "
            "Every source_id must be copied exactly from an id in the supplied sources; never use a section name as an id. "
            "When no supplied source supports a goal, return exactly "
            "{\"goals\":[],\"confidence\":0,\"evidence\":[]} and do not explain why. "
            "Write each objective as what the character wants to do, as a short sentence in the language "
            "of the latest user content; write each rationale in that language too. "
            "Return JSON: {\"goals\":[{\"objective\":str,\"horizon\":\"short_term|long_term\",\"urgency\":0..1,"
            "\"conflict_key\":null|str,\"motivation_signals\":[{\"kind\":str,\"strength\":0..1,\"source_type\":\"event|memory|belief|state\","
            "\"source_id\":str,\"rationale\":str}],\"confidence\":0..1}],\"confidence\":0..1,\"evidence\":[str]}."
        ),
        BackgroundCognitionKind.SELF_MEMORY_EXTRACTION: (
            "Extract durable facts the character stated about themselves in the character lines to "
            "extract from: their tastes, habits, history, relationships, opinions they hold, and "
            "what they are working on or plan to do. One item per fact. Take facts only from the "
            "listed character lines, never from the user's lines; the transcript is context only. "
            "Skip reactions, greetings, questions, stage directions written between asterisks, and "
            "statements only about this moment, such as the time of day or being tired right now; "
            "return an empty items array when the listed lines state no such fact. "
            "Each item's evidence must be one exact quote copied from the listed character lines. "
            "Each summary is one short sentence in the same language as its evidence quote, and it "
            "must make sense on its own later, without the conversation: its subject is the "
            "character, by the name on the Character line, and words such as \"that\", \"it\" or "
            "\"those things\" are replaced by what they refer to in the transcript. "
            "\"<name>: is not interested in that kind of thing at all.\" is useless; write "
            "\"<name> is not interested in horror films at all.\" "
            "Return JSON: {\"items\":[{\"summary\":str,\"kind\":str,\"importance\":0..1,"
            "\"confidence\":0..1,\"evidence\":str}],\"confidence\":0..1,\"evidence\":[str]}."
            # 1.2.0, added at the end only: what comes before is cached.
            " kind is one of: " + ", ".join(SELF_MEMORY_KINDS) + ". What the character is doing in "
            "this conversation or plans to do next in it (teaching the user, a game, a task under "
            "way) is working_on or plan, not habit or history; habit is only what the character does "
            "in every conversation. Anything the character says about this user is not a fact about "
            "the character: what the user understands, did, typed or got wrong, how they are doing, "
            "whether the character likes them or is pleased or annoyed with them, a reproach, a "
            "threat or a judgement aimed at them. Its kind is view_of_user, never opinion, "
            "relationship, trait or history; opinion and relationship are only about things and "
            "people other than this user. A fact about the character keeps its own kind even when it "
            "is told to the user. The summary is written in the language of its quote, never in "
            "English unless the quote is English."
        ),
        BackgroundCognitionKind.CHARACTER_MOOD: (
            "Judge how the character feels at the moment of the character's latest line, the "
            "last line under \"Latest exchange\". The character's lines are marked with the name "
            "on the Character line, the user's with \"User\"; a line marked \"Event\" is "
            "something that happened, not said by anyone. The earlier conversation is "
            "background only: a feeling that shows only in earlier lines does not by itself set "
            "the mood now. What the character said counts as much as what the user said: "
            "talking about something sad can make the character sad, being praised can make "
            "them embarrassed, being insulted can make them angry or sad. "
            "Judge the feeling relative to the character's personality, given as background "
            "when known. Blushing, being flustered or shy stammering mean embarrassed, not "
            "happy or worried. The character's usual teasing or tsundere barbs are the "
            "character's normal manner, not anger, unless the exchange shows the character is "
            "really upset. "
            "Choose exactly one mood from this list and write it exactly as listed, in English, "
            "whatever language the conversation is in: " + ", ".join(CHARACTER_MOODS) + ". "
            "If the latest exchange shows no particular feeling, answer neutral. "
            "intensity is how strongly the character feels it now, from 0 to 1; neutral has "
            "intensity 0. As anchors: about 0.2 is a slight feeling, about 0.5 a clear feeling, "
            "and 0.8 or more only for a major event such as a loss, a real fight or a big "
            "surprise. Small talk with no particular feeling is neutral, or a low intensity "
            "when a faint feeling is there. confidence is how clearly the latest exchange shows "
            "the feeling; do not default to a high value. "
            "evidence is at most 3 short quotes; each must quote words from the latest exchange, "
            "and do not repeat the same quote. "
            "Return JSON: {\"mood\":str,\"intensity\":0..1,\"confidence\":0..1,\"evidence\":[str]}."
        ),
        BackgroundCognitionKind.REPLY_CHECK: (
            "Check the character's reply to check for slips of the five kinds below, and only "
            "these. Most replies have none; a false alarm costs more than a missed slip, so "
            "report a slip only when the words of the reply plainly show it. Write the kind "
            "exactly as listed, in English.\n"
            "- broke_character: the character speaks of what runs behind the conversation: the "
            "instructions, prompts, notes or hints the character is given, speech recognition "
            "or transcription, a model, a system prompt, or being an AI or a program. Teasing, "
            "bossing, refusing or changing the subject is the character's manner, not this "
            "slip.\n"
            "- leaked_markup: markup said as part of the character's words: a tag or field name "
            "written out (\"emotion: happy\", \"<smile>\"), code, JSON or markdown. An "
            "expression keyword in square brackets such as [joy], and an action between "
            "asterisks, are how expressions and actions are written: no slip.\n"
            "- off_persona: the character states a fact about the character that is the "
            "opposite of one written in who the character is: another age, another name or "
            "another way of writing the name, another home or family. against is the words of "
            "that fact, copied exactly from who the character is. How the character speaks, "
            "what the character teaches, a word the character leaves out, a mistake about "
            "anything else, or a made-up detail is no slip.\n"
            "- repeated: the reply opens with, or its main sentence is, nearly the same words "
            "as the character's previous reply. against is those words, copied exactly from the "
            "character's previous reply.\n"
            "- wrong_language: the whole reply is in another language than the user writes in "
            "and than who the character is says the character speaks. Foreign words, quotes, "
            "names and sentences the character is teaching are no slip.\n"
            "Do not judge anything else: whether what is said is true or right, the tone, the "
            "length, or whether it is interesting. "
            "evidence is the words of the reply that show the slip, one sentence at most, "
            "copied exactly from the reply to check, in the language it was said in. fix is one sentence telling "
            "the character what to do in the next reply, addressed to the character as \"you\", "
            "at most 40 characters. At most 2 issues. "
            "Return JSON: {\"issues\":[{\"kind\":str,\"evidence\":str,\"against\":str,"
            "\"fix\":str}]}, against empty for the other kinds; when there is no slip, return "
            "{\"issues\":[]}."
        ),
        BackgroundCognitionKind.USER_STATE: (
            "Sum up how the user has been lately, from the user's own lines and how the user "
            "seemed on each turn. energy is how much energy the user shows: low (tired, worn "
            "out, sleepy, unwell), high (excited, eager, lively) or normal. mood_trend is how "
            "the user's mood moved over these lines: down, flat or up. concerns are at most 3 "
            "things weighing on the user that the user spoke of (work, sleep, an exam, a "
            "person), each in a few words, each with evidence: the user's sentence that shows "
            "it, copied exactly from the user's lines. This is not a diagnosis and not a "
            "judgement of the user: never name an illness or a disorder, and do not say what "
            "the user should do. A sentence the user practises or says after the character is "
            "not about the user. When the lines show nothing particular, answer normal, flat "
            "and no concerns. evidence is at most 3 of the user's sentences, copied exactly "
            "from the user's lines, that show the energy and the trend. "
            "Return JSON: {\"energy\":\"low|normal|high\",\"mood_trend\":\"down|flat|up\","
            "\"concerns\":[{\"concern\":str,\"evidence\":str}],\"evidence\":[str]}."
        ),
        BackgroundCognitionKind.DIARY: (
            "Write the character's diary entry for the day, as the character: in the first "
            "person, in the character's own voice and way of speaking, as one short paragraph "
            "of 3 to 6 sentences that tells the day, not a list of details. Every sentence "
            "must rest on something in what happened: what was talked about, what the user "
            "told the character, what the character said and thought, how the character felt "
            "and what the character set out to do. Do not add events, people, places, times "
            "or feelings that are not in what happened, and do not guess what anyone did "
            "beyond it. Who the character is tells only how the character speaks and sees "
            "things: nothing in it happened on this day, and the entry does not tell it as "
            "part of the day. The diary is for the character alone: never address the user "
            "as \"you\"; write of the user by the name what happened gives, or as the person "
            "the character talked with. When little happened, three short sentences are "
            "enough. Do not mention notes, summaries, records, prompts or being an AI. "
            "evidence is at "
            "most 3 lines copied exactly from what happened, the ones the entry rests on "
            "most. Return JSON: {\"text\":str,\"evidence\":[str]}."
        ),
        BackgroundCognitionKind.VISION_INTERPRETATION: (
            "Interpret only the supplied textual vision observation and recent context. Do not claim unseen pixels. "
            "Return JSON: {\"interpretation\":str,\"tags\":[str],\"confidence\":0..1,\"evidence\":[str]}."
        ),
    }
)



def _json_safe(value: Any) -> Any:
    """Keep provider-neutral goal source snapshots JSON serializable.

    CharacterState.custom intentionally accepts arbitrary host values. Background
    cognition prompts must never fail merely because a host stored a non-JSON
    helper object there, so unsupported leaf values are represented for audit
    rather than mutating or interpreting them.
    """

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _supplied_goal_sources(goal_sources: Any) -> frozenset[tuple[str, str]] | None:
    """(source_type, id) of every source the worker was shown, or None when it
    was shown none and its citations cannot be checked here."""
    if not isinstance(goal_sources, Mapping):
        return None
    supplied: set[tuple[str, str]] = set()
    event = goal_sources.get("event")
    if isinstance(event, Mapping) and event.get("id"):
        supplied.add(("event", str(event["id"])))
    for source_type, section in (
        ("belief", "active_beliefs"),
        ("memory", "retrieved_memories"),
        ("state", "state"),
    ):
        for item in goal_sources.get(section) or ():
            if isinstance(item, Mapping) and item.get("id"):
                supplied.add((source_type, str(item["id"])))
    return frozenset(supplied)


def _citable(
    kind_value: str, source_type: str, source_id: str, supplied: frozenset[tuple[str, str]]
) -> bool:
    try:
        allowed = MOTIVATION_SOURCE_TYPES[MotivationKind(kind_value)]
    except ValueError:
        return False
    source_type = source_type.casefold()
    return source_type in allowed and (source_type, source_id) in supplied


def _parse_json_object(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # A stray backtick or two after the object, as a local model sometimes
        # closes an answer, is no other content.
        try:
            value, end = json.JSONDecoder().raw_decode(cleaned)
        except json.JSONDecodeError:
            value, end = None, 0
        if value is None or cleaned[end:].strip(" \t\r\n`"):
            raise ValueError("background cognition model must return one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("background cognition model response must be a JSON object")
    return value


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0 <= numeric <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return numeric


def _number(value: Any) -> float | None:
    """A finite number, also written as one in a string ("0.6", as models
    sometimes answer), or None."""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected numeric value between 0 and 1") from exc
    return max(0.0, min(1.0, numeric))


def _signed_unit_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number:
        return None
    return max(-1.0, min(1.0, number))


def _evidence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("evidence must be an array")
    return tuple(str(item).strip() for item in value if str(item).strip())[:12]


def _squeeze(text: str) -> str:
    # Models re-space quotes (Chinese "研究AI", "studying AI", comes back as
    # "研究 AI"); spacing is not evidence.
    return "".join(text.split())


def _user_lines(context: TaskContext, *, turns: int) -> list[str]:
    """What the user said in the turns this run is responsible for.

    Earlier lines were the previous run's job; accepting quotes from them makes
    the worker re-extract the same facts in new wording on every turn. Cadence
    counts foreground turns, so host events occupy a slot in the window without
    contributing a line: they are not the user speaking.
    """
    payload = context.request.payload
    unread = payload.get("user_lines_to_extract")
    if isinstance(unread, (list, tuple)):
        return [str(line) for line in unread]
    openers = [message for message in context.snapshot.history if message.role in {"user", "event"}]
    latest = str(payload.get("latest_user_or_event", ""))
    if not openers and payload.get("foreground_event_type") == "user_message" and latest.strip():
        # The host keeps no history (max_history_messages=0): only this turn is known.
        return [latest]
    window = openers[-max(1, turns):]
    return [message.content for message in window if message.role == "user"]


def _unread(
    reading: _Reading, openers: Sequence[Message], read_up_to: Message | None
) -> int | None:
    """Where the turns begin that were not read yet, after ``read_up_to``, as
    an index into ``openers``; None when that cannot be told: the conversation
    was replaced or trimmed past it.

    Counting turns does not find these lines. An interrupted turn leaves a
    line in the conversation without being a turn, and a turn the host keeps
    out of the conversation is a turn without a line.
    """
    if not openers or not reading.known:
        return None
    start = 0
    if read_up_to is None:
        if openers[0] is not reading.first_line:
            return None  # another conversation, or this one was trimmed
    else:
        last = next(
            (
                index
                for index in range(len(openers) - 1, -1, -1)
                if openers[index] is read_up_to
            ),
            None,
        )
        if last is None:
            return None
        start = last + 1
    return start


def _replies_from(history: Sequence[Message], openers: Sequence[Message]) -> list[str]:
    """What the character said in the turns that begin with ``openers``: her
    replies, and remarks she made on her own after an event."""
    if not openers:
        return []
    first = next((i for i, message in enumerate(history) if message is openers[0]), None)
    if first is None:
        return []
    return [
        message.content
        for message in history[first:]
        if message.role == "assistant" and message.content.strip()
    ]


def _assistant_lines(context: TaskContext, *, turns: int) -> list[str]:
    """What the character said in the turns this run is responsible for; see
    _user_lines."""
    payload = context.request.payload
    unread = payload.get("assistant_lines_to_extract")
    if isinstance(unread, (list, tuple)):
        return [str(line) for line in unread]
    history = context.snapshot.history
    openers = [message for message in history if message.role in {"user", "event"}]
    if not openers:
        # The host keeps no history: only this turn is known.
        reply = str(payload.get("assistant_response", ""))
        return [reply] if reply.strip() else []
    return _replies_from(history, openers[-max(1, turns):])


_PERSONA_SUMMARY_CHARS = 400
# The reply check holds her words against her persona, and her name and how it
# is written come late in one. The persona stands before the conversation in
# what the worker reads, the same on every turn, so a model server reuses it.
_REPLY_CHECK_PERSONA_CHARS = 1500
# Her diary is in her voice: more of who she is than her mood needs.
_DIARY_PERSONA_CHARS = 1500


def _persona_summary(profile: Any, *, limit: int = _PERSONA_SUMMARY_CHARS) -> str:
    """Who she is, short, for the mood worker: without it a 9B model read her
    surface tone, a shy character's stammer as worry and a tsundere's habitual
    barbs as anger. The whole persona is not needed, and the conversation must
    stay the larger part of what the worker reads.

    Prefers ``profile.background`` (who she is) when it is a non-empty
    string, and falls back to ``profile.description`` otherwise: a host
    (Tomoshibi) puts its whole system prompt, generic speech rules first,
    into description, so reading description made this a summary of the
    host's rules, not of her."""
    if profile is None:
        return ""
    background = str(getattr(profile, "background", "") or "").strip()
    base = background if background else str(getattr(profile, "description", "") or "")
    parts = [base]
    personality = [str(item).strip() for item in getattr(profile, "personality", None) or ()]
    if any(personality):
        parts.append("Personality: " + ", ".join(item for item in personality if item))
    summary = " ".join(" ".join(parts).split())
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def _mood_messages(context: TaskContext, history_messages: int) -> list[Message]:
    """Both sides of the conversation, her lines under her name, with the
    latest exchange apart from what came before.

    Unlike the emotion worker, this one must read her: what she said herself
    (something sad, being praised) moves her mood as much as what she was
    told. Her mood is read as of her latest line; given the conversation as
    one block, a 9B model kept citing a sigh from earlier turns as how she
    felt now. The turn notes are the runtime's, nobody's words, and stay out.
    """
    name = context.snapshot.character_name
    spoken = [
        message
        for message in context.snapshot.history[-history_messages:]
        if message.role in {"user", "assistant", "event"}
        and message.content.strip()
        and not is_turn_context(message)
    ]
    opener = next(
        (index for index in range(len(spoken) - 1, -1, -1) if spoken[index].role != "assistant"),
        None,
    )
    payload = context.request.payload
    if opener is None:
        # The host keeps no history (max_history_messages=0): only this turn is known.
        earlier: list[Message] = []
        said = payload.get("foreground_event_type") == "user_message"
        latest = [
            Message(role=role, content=content)
            for role, content in (
                ("user" if said else "event", str(payload.get("latest_user_or_event", "")).strip()),
                ("assistant", str(payload.get("assistant_response", "")).strip()),
            )
            if content
        ]
    else:
        earlier = [message for message in spoken[:opener] if message.role != "event"]
        latest = spoken[opener:]
        if latest[0].role == "event" and str(payload.get("latest_user_or_event", "")).strip():
            # The event as it happened, not the runtime's record of it.
            latest[0] = Message(role="event", content=str(payload["latest_user_or_event"]).strip())

    def line(message: Message) -> str:
        speaker = {"assistant": name, "user": "User"}.get(message.role, "Event")
        return f"{speaker}: {message.content.strip()}"

    persona = str(payload.get("character_persona") or "").strip()
    user = (
        f"Character: {name}\n"
        + (
            f"Who the character is (background, not part of the conversation):\n{persona}\n\n"
            if persona
            else ""
        )
        + "Earlier conversation, oldest first (background only):\n"
        + ("\n".join(line(message) for message in earlier) or "(empty)")
        + "\n\nLatest exchange (judge the mood as of the last line here):\n"
        + ("\n".join(line(message) for message in latest) or "(empty)")
        + "\n"
    )
    observed = context.snapshot.state.custom.get("observed_user_emotion")
    if isinstance(observed, Mapping) and str(observed.get("emotion") or "").strip():
        # The emotion of this very turn is read alongside; this one is older.
        user += f"\nHow the user seemed most recently: {str(observed['emotion']).strip()}\n"
    user += "\nReturn only the requested JSON object. The mood is one of the listed English words."
    return [
        Message(role="system", content=_SYSTEM_PROMPTS[BackgroundCognitionKind.CHARACTER_MOOD]),
        Message(role="user", content=user),
    ]


def _replies_around(context: TaskContext, history_messages: int) -> tuple[str, str, str]:
    """(her previous reply, her reply of this turn, what she replied to) as
    kept in the conversation; the turn notes are nobody's words. Without kept
    history, this turn as the runtime reported it."""
    spoken = [
        message
        for message in context.snapshot.history[-history_messages:]
        if message.role in {"user", "assistant", "event"}
        and message.content.strip()
        and not is_turn_context(message)
    ]
    payload = context.request.payload
    latest = str(payload.get("latest_user_or_event", "")).strip()
    opener = next(
        (index for index in range(len(spoken) - 1, -1, -1) if spoken[index].role != "assistant"),
        None,
    )
    if opener is None:
        return "", str(payload.get("assistant_response", "")).strip(), latest
    after = [m.content.strip() for m in spoken[opener + 1 :] if m.role == "assistant"]
    before = [m.content.strip() for m in spoken[:opener] if m.role == "assistant"]
    reply = after[-1] if after else str(payload.get("assistant_response", "")).strip()
    return (before[-1] if before else ""), reply, latest


# Words of what runs behind the conversation; a slip out of character names one.
_BACKSTAGE = re.compile(
    r"(?<![a-z])(?:a\.?i|artificial intelligence|language model|model|prompts?|system|"
    r"instructions?|notes?|speech recognition|transcri\w*|assistant|chatbot|bot|program)(?![a-z])"
    r"|人工智慧|人工智能|模型|提示|系統|系统|指令|指示|備註|备注|說明|说明|設定|设定|辨識|辨识|识别"
    r"|聽寫|听写|語音|语音|程式|程序|機器人|机器人|助手|助理|演算法|算法"
    r"|プロンプト|モデル|システム|アシスタント|音声認識|認識|文字起こし|人工知能|プログラム|メモ|注釈",
    re.IGNORECASE,
)
# How expressions and actions are written: an expression keyword in square
# brackets, an action between asterisks. Not markup left in her words.
_DIRECTIONS = re.compile(
    r"\[[^\[\]\n]{1,60}\]|\*[^*\n]+\*|（[^（）\n]{1,60}）|\([^()\n]{1,60}\)"
)
_MARKUP = re.compile(r"[\[\]<>{}#`|*]|(?<![A-Za-z])[A-Za-z_]+\s*[:=]\s*\S")
# How near the words of her previous reply a repetition must be.
_REPEATED_RATIO = 0.6
# Letters and digits in a sentence said again, not a phrase.
_WHOLE_SENTENCE = 12


def _script(text: str) -> str:
    """The writing a text is mostly in: ja (a quarter or more of its letters
    kana), zh, ko, latin or other; empty without letters."""
    counts = {"kana": 0, "zh": 0, "ko": 0, "latin": 0, "other": 0}
    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F:
            counts["kana"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
            counts["zh"] += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
            counts["ko"] += 1
        elif code < 0x250:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values())
    if not total:
        return ""
    if counts["kana"] * 4 >= total:
        return "ja"
    return max(("zh", "ko", "latin", "other"), key=counts.__getitem__)


_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")


def _word_pieces(text: str) -> set[str]:
    """What two texts must share to speak of the same thing: two characters in
    a row of Chinese, Japanese or Korean, the first four letters of a longer
    word, or any number at all (an age against an age)."""
    text = text.casefold()
    pieces = {
        run[index : index + 2]
        for run in _CJK_RUN.findall(text)
        for index in range(len(run) - 1)
    }
    pieces.update(word[:4] for word in re.findall(r"[a-z]{4,}", text))
    if re.search(r"\d", text):
        pieces.add("#")
    return pieces


# Where the first clause of what she says ends.
_CLAUSE_END = re.compile(r"[，,、。．.！!？?…～~；;：:—\n]")
_QUOTE_MARKS = frozenset("「」『』“”\"'《》〈〉")
# How much an opening clause needs to count: four characters of Chinese or
# Japanese (「哈↗哈↘哈↗」 counts, 「哈哈哈」 does not) or three words; and how
# much alike the start of two replies must be: six characters or three words.
_OPENING_CLAUSE_CHARACTERS = 4
_OPENING_WORDS = 3
_OPENING_CHARACTERS = 6
_LATIN_WORD = re.compile(r"[A-Za-z0-9']+")
_OPENING_FIXES = {
    "zh": "開頭別再用同一句，換個起手。",
    "zh-hans": "开头别再用同一句，换个起手。",
    "ja": "同じ書き出しはやめて、別の言葉で始めて。",
    "ko": "같은 말로 시작하지 말고 다르게 시작해.",
    "latin": "Do not open with the same words again.",
}


def _spoken(text: str) -> list[tuple[str, int]]:
    """Her words, each character with where it stands in ``text``: without
    expression tags, actions and quote marks, and from her first word."""
    hidden = [False] * len(text)
    for match in _DIRECTIONS.finditer(text):
        for index in range(match.start(), match.end()):
            hidden[index] = True
    kept = [
        (char, index)
        for index, char in enumerate(text)
        if not hidden[index] and char not in _QUOTE_MARKS
    ]
    start = 0
    while start < len(kept) and (
        kept[start][0].isspace() or unicodedata.category(kept[start][0]).startswith("P")
    ):
        start += 1
    return kept[start:]


def _pieces(spoken: list[tuple[str, int]]) -> list[tuple[str, int, int]]:
    """(piece, start, end) in the text: one per character of Chinese,
    Japanese or Korean, one per word of Latin letters."""
    pieces, index = [], 0
    while index < len(spoken):
        char, at = spoken[index]
        if char.isascii() and (char.isalnum() or char == "'"):
            end = index
            while end + 1 < len(spoken) and spoken[end + 1][0].isascii() and (
                spoken[end + 1][0].isalnum() or spoken[end + 1][0] == "'"
            ):
                end += 1
            word = "".join(c for c, _ in spoken[index : end + 1]).casefold()
            pieces.append((word, at, spoken[end][1] + 1))
            index = end + 1
            continue
        if char.isalnum():
            pieces.append((char, at, at + 1))
        index += 1
    return pieces


def _clause(spoken: list[tuple[str, int]]) -> list[tuple[str, int]]:
    for index, (char, _) in enumerate(spoken):
        if _CLAUSE_END.match(char):
            return spoken[:index]
    return spoken


def _clause_key(clause: list[tuple[str, int]]) -> tuple[str, int, int]:
    """(what the clause says, its characters of Chinese or Japanese and the
    like, its words): letters, digits and signs such as arrows, no spacing
    or punctuation."""
    key = "".join(
        char.casefold()
        for char, _ in clause
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )
    wide = sum(not char.isascii() for char in key)
    words = len(_LATIN_WORD.findall("".join(char for char, _ in clause)))
    return key, wide, words


def _same_opening(reply: str, previous: str) -> str:
    """The opening of her reply, as she wrote it, when her previous reply
    opened the same way: the same first clause of four characters or three
    words, or the same first six characters or three words; empty
    otherwise."""
    if not previous.strip():
        return ""
    now, before = _spoken(reply), _spoken(previous)
    clause = _clause(now)
    key, wide, words = _clause_key(clause)
    if (wide >= _OPENING_CLAUSE_CHARACTERS or (not wide and words >= _OPENING_WORDS)) and (
        key == _clause_key(_clause(before))[0]
    ):
        return reply[clause[0][1] : clause[-1][1] + 1].strip()
    pieces, pieces_before = _pieces(now), _pieces(before)
    if not pieces:
        return ""
    need = _OPENING_WORDS if len(pieces[0][0]) > 1 or pieces[0][0].isascii() else _OPENING_CHARACTERS
    head = [piece for piece, _, _ in pieces[:need]]
    if len(head) == need and head == [piece for piece, _, _ in pieces_before[:need]]:
        return reply[pieces[0][1] : pieces[need - 1][2]].strip()
    return ""


def _script_of_language(name: str) -> str:
    """The writing of a language named by the host ("繁體中文", "Japanese",
    "Traditional Chinese (Taiwan)"); empty when it cannot be told."""
    name = name.strip().casefold()
    if not name:
        return ""
    if "日本" in name or "japan" in name or name.startswith("ja"):
        return "ja"
    if "korea" in name or "한국" in name or "韓" in name or "韩" in name or name.startswith("ko"):
        return "ko"
    chinese = ("中文", "chinese", "漢語", "汉语", "華語", "华语", "國語", "国语", "普通")
    if any(word in name for word in chinese) or name.startswith("zh"):
        return "zh"
    if "english" in name or "英" in name or name.startswith("en"):
        return "latin"
    return ""


def _fix_script(language: str, said_by_user: str) -> str:
    """The writing a fix must be in: that of the host's language when it
    names one, else that of the user's latest line."""
    if language.strip():
        return _script_of_language(language)
    return _script(said_by_user)


def _written_in(text: str, script: str) -> bool:
    """Whether a short text is written in ``script``; a word quoted in
    another writing does not change it."""
    kana = sum(0x3040 <= ord(c) <= 0x30FF for c in text)
    han = sum(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF for c in text)
    hangul = sum(0xAC00 <= ord(c) <= 0xD7AF for c in text)
    if script == "ja":
        return kana > 0 and kana * 4 >= han + kana
    if script == "zh":
        return han > 0 and kana * 4 < han + kana
    if script == "ko":
        return hangul > 0
    if script == "latin":
        return kana + han + hangul == 0
    return _script(text) == script


def _opening_fix(script: str, payload: Mapping[str, Any]) -> str:
    language = str(payload.get("output_language") or "").casefold()
    if script == "zh" and ("简" in language or "simplified" in language or "hans" in language):
        return _OPENING_FIXES["zh-hans"]
    return _OPENING_FIXES.get(script, _OPENING_FIXES["latin"])


def _slip_holds(
    kind: str,
    quote: str,
    against: str,
    *,
    reply: str,
    previous: str,
    persona: str,
    said_by_user: str,
) -> bool:
    """Whether a slip the model reported is what its kind says, as far as the
    words show it. A 9B model called nearly every reply a slip of some kind:
    a teasing line out of character, a word it thought wrong off persona."""
    if kind == "off_persona":
        # Against a fact she was given, quoted from it, and about the same
        # thing as what she said.
        wanted = _squeeze(against.casefold())
        return (
            bool(wanted)
            and wanted in _squeeze(persona.casefold())
            and bool(_word_pieces(quote) & _word_pieces(against))
        )
    if kind == "repeated":
        said, quoted = _letters(previous), _letters(quote)
        if not against or _letters(against) not in said:
            return False
        near = (len(quoted) >= _SHORTEST_ECHO and quoted in said) or (
            difflib.SequenceMatcher(None, quoted, _letters(against)).ratio() >= _REPEATED_RATIO
        )
        # The same opening, or a whole sentence said again; words she is
        # teaching come back in reply after reply.
        opens = _squeeze(reply).startswith(_squeeze(quote)) and _squeeze(previous).startswith(
            _squeeze(against)
        )
        return near and (opens or len(quoted) >= _WHOLE_SENTENCE)
    if kind == "broke_character":
        named = {match.casefold() for match in _BACKSTAGE.findall(quote)}
        # What her persona is about (an AI she built) is her, not a slip.
        hers = {match.casefold() for match in _BACKSTAGE.findall(persona)}
        return bool(named) and not named & hers
    if kind == "leaked_markup":
        return _MARKUP.search(_DIRECTIONS.sub(" ", quote)) is not None
    if kind == "wrong_language":
        # Neither the user's language nor that of her persona, and the quote
        # shows it.
        hers, theirs, own = _script(reply), _script(said_by_user), _script(persona)
        return (
            bool(hers and theirs)
            and hers != theirs
            and hers != own
            and _script(quote) == hers
        )
    return False


# The kinds a second question settles; the others the words show.
_ASKED_AGAIN = ("off_persona", "broke_character")
# Measured on a local 9B model with sentences from real conversations: asked
# whether a sentence was "the opposite" of a fact, it answered no even to
# eighteen against 111; asked for a conflict and the word that shows it, it
# told them apart, and the word is checked against the sentence.
_CONFLICT_QUESTION = (
    "Compare one sentence a character said with one fact written about the character. "
    "Return JSON: {\"conflict\":\"yes\",\"word\":str} or {\"conflict\":\"no\"}; word is "
    "the different age, name or spelling, copied exactly from the sentence."
)
_BEHIND_QUESTION = (
    "Read one sentence a character said in a conversation. "
    "Return JSON: {\"behind\":\"yes\",\"word\":str} or {\"behind\":\"no\"}."
)


def _slip_question(item: Mapping[str, str], against: str, persona: str) -> list[Message]:
    if item["kind"] == "off_persona":
        system = _CONFLICT_QUESTION
        user = (
            f"Fact written about the character:\n{against}\n\n"
            f"What the character said:\n{item['evidence']}\n\n"
            "Is there a conflict: does the sentence give the character a different age, or "
            "call the character by a name or spelling that the fact does not allow? Only if "
            "the sentence itself contains that different age, name or spelling. If the "
            "sentence agrees with the fact, or is about something else, the answer is no."
        )
    else:
        system = _BEHIND_QUESTION
        user = (
            f"Who the character is:\n{persona or '(not given)'}\n\n"
            f"What the character said:\n{item['evidence']}\n\n"
            "Does the character step out of the story and speak of what runs the "
            "conversation: the character being an AI, a model or a program, the prompts, "
            "instructions or notes the character is given, or speech recognition getting "
            "the user's words wrong? word is the word that shows it, copied exactly from "
            "the sentence. Things in the character's own world, hints or notes the "
            "character gives the user, and anything that is part of who the character is, "
            "are no."
        )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _slip_confirmed(item: Mapping[str, str], against: str, answer: Mapping[str, Any]) -> bool:
    """A yes, and a word of her sentence that shows it; for a conflict, a word
    the fact does not hold itself."""
    key = "conflict" if item["kind"] == "off_persona" else "behind"
    if str(answer.get(key) or "").strip().casefold() != "yes":
        return False
    word = _squeeze(str(answer.get("word") or "").casefold())
    if not word or word not in _squeeze(item["evidence"].casefold()):
        return False
    return item["kind"] != "off_persona" or word not in _squeeze(against.casefold())


def _reply_check_messages(context: TaskContext, history_messages: int) -> list[Message]:
    """Who she is, her previous reply, what she replied to and her reply,
    each apart: the check looks at her reply alone, the rest is what it is
    held against."""
    previous, reply, latest = _replies_around(context, history_messages)
    payload = context.request.payload
    persona = str(payload.get("character_persona") or "").strip()
    said = payload.get("foreground_event_type") == "user_message"
    user = (
        f"Character: {context.snapshot.character_name}\n"
        + (
            f"Who the character is (background, not part of the conversation):\n{persona}\n\n"
            if persona
            else ""
        )
        + f"The character's previous reply:\n{previous or '(none)'}\n\n"
        + ("The user's latest line" if said else "What just happened (not said by anyone)")
        + f":\n{latest or '(none)'}\n\n"
        + f"The character's reply to check:\n{reply or '(none)'}\n"
    )
    language = str(payload.get("output_language") or "").strip()
    user += (
        "\nReturn only the requested JSON object. kind is one of the listed English words; "
        "evidence is copied as it was said."
        + (
            f" The fix is written in {language}."
            if language
            else " The fix is written in the language of the user's latest line,"
            " not in English unless that line is English."
        )
    )
    return [
        Message(role="system", content=_SYSTEM_PROMPTS[BackgroundCognitionKind.REPLY_CHECK]),
        Message(role="user", content=user),
    ]


# Names of an illness or a disorder: how the user has been is no diagnosis.
_DIAGNOSIS = re.compile(
    r"depress|disorder|syndrome|diagnos|adhd|ptsd|bipolar"
    r"|憂鬱症|抑鬱症|抑郁症|焦慮症|焦虑症|躁鬱|躁郁|失眠症|症候群|障礙|障碍"
    r"|うつ病|鬱病|障害|不眠症",
    re.IGNORECASE,
)


def _state_lines(context: TaskContext, *, turns: int) -> list[str]:
    """What the user said since the user state was last read; see _user_lines."""
    unread = context.request.payload.get("user_lines_for_state")
    if isinstance(unread, (list, tuple)):
        return [str(line) for line in unread]
    return _user_lines(context, turns=turns)


def _user_state_messages(context: TaskContext, turns: int) -> list[Message]:
    """The user's lines only, and how the user seemed on each turn: nothing
    she said is evidence of how the user has been."""
    payload = context.request.payload
    lines = _state_lines(context, turns=turns)
    readings = []
    for raw in payload.get("user_emotions") or ():
        if not isinstance(raw, Mapping) or not str(raw.get("emotion") or "").strip():
            continue
        scores = [
            f"{key} {score:.1f}"
            for key in ("valence", "stance")
            if (score := _number(raw.get(key))) is not None
        ]
        readings.append(
            f"- {str(raw['emotion']).strip()}" + (f" ({', '.join(scores)})" if scores else "")
        )
    user = (
        f"Character: {context.snapshot.character_name}\n"
        "The user's lines, oldest first:\n"
        + ("\n".join(f"- {line}" for line in lines) or "(none)")
        + "\n\nHow the user seemed on each turn, oldest first (valence from -1 unpleasant to "
        "1 pleasant; stance towards the character from -1 hostile to 1 warm):\n"
        + ("\n".join(readings) or "(not read)")
        + "\n"
    )
    language = str(payload.get("output_language") or "").strip()
    user += "\nReturn only the requested JSON object. " + (
        f"concerns are written in {language}."
        if language
        else "concerns are written in the language of the user's lines, not in English "
        "unless the user writes English."
    )
    return [
        Message(role="system", content=_SYSTEM_PROMPTS[BackgroundCognitionKind.USER_STATE]),
        Message(role="user", content=user),
    ]


_MOODS = "How the character felt through the day (time, mood)"
# What happened on her day, as the host gives it: (title, payload key).
_DIARY_SECTIONS = (
    ("Conversations of the day", "summaries"),
    ("What the user told the character", "user_told"),
    ("What the character said about themselves", "said_about_herself"),
    ("What the character thought of the user", "views_of_user"),
    (_MOODS, "moods"),
    ("What the character set out to do", "goals"),
)


def _diary_sections(day: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    sections = []
    for title, key in _DIARY_SECTIONS:
        lines = [one for raw in day.get(key) or () if (one := " ".join(str(raw).split()))]
        if lines:
            sections.append((title, lines))
    return sections


# A line of a list: "- ", "* ", "• ", "1. ", "1) ".
_LIST_ITEM = re.compile(r"^\s*(?:[-*•・]|\d+[.)、])\s*")
# Where a sentence ends: after 。！？!?… (and the closing marks after them), or
# after a full stop before a space.
_DIARY_SENTENCE_END = re.compile(
    r"(?<=[。！？!?…])(?![」』）)\"'”’。！？!?…])|(?<=[.!?][\"'”’)])\s+|(?<=\.)\s+"
)


def _diary_text(raw: str) -> str:
    """Her entry as one paragraph of at most DIARY_SENTENCES sentences; empty
    when it is a list of details, which is no diary."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if sum(bool(_LIST_ITEM.match(line)) for line in lines) >= 2:
        return ""
    wide = _script(raw) in ("zh", "ja", "ko")
    text = ("" if wide else " ").join(lines)
    return ("" if wide else " ").join(_diary_sentences(text)[:DIARY_SENTENCES])


# How much of a sentence of her entry must be found in what happened (pieces
# of _word_pieces: two characters of Chinese or Japanese, the start of a
# word): measured on entries of a 9B model, sentences of the day shared 0.23
# to 0.82 of their pieces with it, sentences from her persona or made up 0.00
# to 0.11. A sentence of fewer pieces ("It rained.") is not judged.
_OF_THE_DAY = 0.2
_PIECES_JUDGED = 3
# At least this many sentences of the day, or there is no entry.
_DIARY_SENTENCES_KEPT = 2
# The user spoken to, outside quoted words.
_SPEAKS_TO_THE_USER = re.compile(
    r"你|妳|您|あなた|(?<![A-Za-z])(?:you|your|yours)(?![A-Za-z])", re.IGNORECASE
)
_QUOTED_WORDS = re.compile(r"「[^」]*」|『[^』]*』|“[^”]*”|\"[^\"]*\"")


def _diary_sentences(text: str) -> list[str]:
    return [part.strip() for part in _DIARY_SENTENCE_END.split(text) if part.strip()]


def _of_the_day(sentence: str, day: set[str]) -> bool:
    pieces = _word_pieces(sentence)
    if len(pieces) < _PIECES_JUDGED:
        return True
    return len(pieces & day) >= _OF_THE_DAY * len(pieces)


def _not_of_the_day(text: str, happened: Sequence[str]) -> list[str]:
    """The sentences of her entry that share too little with what happened."""
    day = _word_pieces("\n".join(happened))
    return [sentence for sentence in _diary_sentences(text) if not _of_the_day(sentence, day)]


def _told_of_the_day(text: str, happened: Sequence[str]) -> str:
    """Her entry without the sentences that are not of the day; empty when
    fewer than _DIARY_SENTENCES_KEPT are left."""
    day = _word_pieces("\n".join(happened))
    kept = [sentence for sentence in _diary_sentences(text) if _of_the_day(sentence, day)]
    if len(kept) < _DIARY_SENTENCES_KEPT:
        return ""
    return ("" if _script(text) in ("zh", "ja", "ko") else " ").join(kept)


def _diary_messages(context: TaskContext) -> list[Message]:
    """Who she is, and what happened on her day, section by section."""
    payload = context.request.payload
    day = payload.get("diary")
    day = day if isinstance(day, Mapping) else {}
    persona = str(payload.get("character_persona") or "").strip()
    happened = "\n\n".join(
        f"{title}:\n" + "\n".join(f"- {line}" for line in lines)
        for title, lines in _diary_sections(day)
    )
    user = (
        f"Character: {context.snapshot.character_name}\n"
        + (
            f"Who the character is (background, not part of the day):\n{persona}\n\n"
            if persona
            else ""
        )
        + f"The day: {str(day.get('date') or '').strip() or '(today)'}\n\n"
        + f"What happened:\n\n{happened or '(nothing)'}\n"
    )
    language = str(payload.get("output_language") or "").strip()
    # Placed last, as the language rule: in the system prompt alone a 9B model
    # still wrote half its entries to the user.
    user += (
        "\nReturn only the requested JSON object. The diary is for the character alone: "
        'the user is he, she or their name in it, never "you". '
    ) + (
        f"The entry is written in {language}."
        if language
        else "The entry is written in the language of what happened, not in English unless "
        "what happened is in English."
    )
    return [
        Message(role="system", content=_SYSTEM_PROMPTS[BackgroundCognitionKind.DIARY]),
        Message(role="user", content=user),
    ]


def said_in(quote: str, lines: Sequence[str]) -> bool:
    """Whether the character said ``quote`` in one of ``lines``: spacing does
    not count, and stage directions between asterisks are not what she said.
    The self-memory worker and whoever checks its quotes later read alike."""
    return _line_quoted(quote, [_STAGE_DIRECTION.sub(" ", line) for line in lines]) is not None


def _line_quoted(quote: str, lines: list[str]) -> str | None:
    wanted = _squeeze(quote)
    if not wanted:
        return None
    return next((line for line in lines if wanted in _squeeze(line)), None)


# The title of what the emotion worker reads before the latest line.
_EARLIER_USER_LINES = "Earlier user lines (background only)"
# Quoted words with fewer letters than this ("はい", "ok") are not taken for
# hers when she said them too.
_SHORTEST_ECHO = 4


def _letters(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def _echoes(quote: str, her_lines: Sequence[str]) -> bool:
    """Whether the user's quote repeats words she said in the transcript.
    Letters and digits only: the user types "私はペコラ。" after her
    「私はペコラ」！"""
    said = _letters(quote)
    return len(said) >= _SHORTEST_ECHO and any(said in _letters(line) for line in her_lines)


def _her_lines(context: TaskContext, history_messages: int) -> list[str]:
    """What she said in the transcript the worker was shown, and her reply."""
    lines = [
        message.content
        for message in context.snapshot.history[-history_messages:]
        if message.role == "assistant" and message.content.strip()
    ]
    reply = str(context.request.payload.get("assistant_response", ""))
    return [*lines, reply] if reply.strip() else lines


def _lines_the_user_said(context: TaskContext, history_messages: int) -> list[str]:
    """What the user said in the transcript the worker was shown, and the
    latest line when the turn was the user's."""
    lines = [
        message.content
        for message in context.snapshot.history[-history_messages:]
        if message.role == "user" and not is_turn_context(message)
    ]
    payload = context.request.payload
    latest = str(payload.get("latest_user_or_event", ""))
    if payload.get("foreground_event_type") == "user_message" and latest.strip():
        lines.append(latest)
    return lines


def _earlier_user_lines(history: Sequence[Message]) -> list[Message]:
    """The user's lines before the turn being read: its own line, the last
    user line or event, is the latest content and is given apart."""
    opener = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if history[index].role in {"user", "event"}
        ),
        None,
    )
    earlier = history if opener is None else history[:opener]
    return [message for message in earlier if message.role == "user"]


def _is_vision_event(event: CharacterEvent) -> bool:
    return (
        event.type in {"vision_observation", "multimodal_user_message"}
        or event.source.startswith("vision:")
    )
