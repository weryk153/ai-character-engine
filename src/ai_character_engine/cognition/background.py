from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping, Sequence

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals.models import MOTIVATION_SOURCE_TYPES, MotivationKind
from ai_character_engine.llm.models import Message
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
            response = await self.models.generate(
                _ROLE_BY_KIND[self.spec.kind],
                messages,
                requirements=self.spec.requirements,
            )
            data = _parse_json_object(response.text)
            value, confidence, evidence, proposals = self._normalize(context, data)
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
            if not isinstance(raw_issues, list):
                return (), confidence, (), []
            reply = _replies_around(context, self.history_messages)[1]
            issues: list[dict[str, str]] = []
            for raw in raw_issues:
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
                if _line_quoted(quote, [reply]) is None:
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
                persona = _persona_summary(getattr(self.tasks.runtime, "character", None))
                if persona:
                    job = {**payload, "character_persona": persona}
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
            "Check the character's reply to check for slips of the kinds below, and only "
            "these. Write the kind exactly as listed, in English.\n"
            "- broke_character: the character talks about what runs behind the conversation "
            "as if it were part of it: instructions, prompts, notes or hints given to the "
            "character, speech recognition or transcription, a model, a system prompt, or being "
            "an AI, unless who the character is makes that part of the character.\n"
            "- leaked_markup: markup or a direction said as part of the character's words: a "
            "tag or field name written out (\"emotion: happy\", \"(expression: smile)\"), code, "
            "JSON or markdown. An expression keyword in square brackets such as [joy], and an "
            "action between asterisks, are how expressions and actions are written: no slip.\n"
            "- off_persona: the reply contradicts a fact given in who the character is: age, "
            "name or how it is written, or another stated fact. What the character makes up "
            "that nothing given contradicts is no slip.\n"
            "- repeated: the reply opens with, or its main sentence is, nearly the same as the "
            "character's previous reply.\n"
            "- wrong_language: the whole reply is in a language other than the one the user "
            "writes in or the character is meant to speak. Foreign words, quotes, names and "
            "sentences the character is teaching are no slip.\n"
            "Do not judge anything else: whether what is said is true or right, the tone, the "
            "length, or whether it is interesting. When in doubt there is no slip: most replies "
            "have none, and a false alarm costs more than a missed slip. "
            "evidence is the one sentence of the reply that has the slip, copied exactly from "
            "the reply to check, in the language it was said in. fix is one sentence telling "
            "the character what to do in the next reply, addressed to the character as \"you\", "
            "at most 40 characters. At most 2 issues. "
            "Return JSON: {\"issues\":[{\"kind\":str,\"evidence\":str,\"fix\":str}]}; when "
            "there is no slip, return {\"issues\":[]}."
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


def _persona_summary(profile: Any) -> str:
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
    if len(summary) > _PERSONA_SUMMARY_CHARS:
        summary = summary[: _PERSONA_SUMMARY_CHARS - 1].rstrip() + "…"
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
        + (f" Text values must be written in {language}." if language else _OUTPUT_LANGUAGE_RULE)
    )
    return [
        Message(role="system", content=_SYSTEM_PROMPTS[BackgroundCognitionKind.REPLY_CHECK]),
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
