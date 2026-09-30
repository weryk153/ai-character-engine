from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals.models import MOTIVATION_SOURCE_TYPES, MotivationKind
from ai_character_engine.llm.models import Message
from ai_character_engine.memory.evidence import classify_user_text
from ai_character_engine.runtime.models import CharacterRunResult
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


_ROLE_BY_KIND: dict[BackgroundCognitionKind, CognitiveRole] = {
    BackgroundCognitionKind.MEMORY_EXTRACTION: CognitiveRole.MEMORY,
    BackgroundCognitionKind.EMOTION_ANALYSIS: CognitiveRole.EMOTION,
    BackgroundCognitionKind.CONVERSATION_SUMMARY: CognitiveRole.SUMMARY,
    BackgroundCognitionKind.REFLECTION: CognitiveRole.REFLECTION,
    BackgroundCognitionKind.GOAL_MOTIVATION: CognitiveRole.GOAL,
    BackgroundCognitionKind.VISION_INTERPRETATION: CognitiveRole.VISION,
}


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
        history = list(context.snapshot.history[-self.history_messages :])
        payload = context.request.payload
        latest = str(payload.get("latest_user_or_event", "")).strip()
        assistant = str(payload.get("assistant_response", "")).strip()
        if self.spec.kind is BackgroundCognitionKind.EMOTION_ANALYSIS:
            # Nothing the character said is evidence of how the user feels or
            # treats her, and a small model cannot keep the two apart: with a
            # character that insults the user it rated his plain questions as
            # hostile. It is given his side of the conversation only.
            history = [message for message in history if message.role != "assistant"]
            assistant = ""
        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in history if message.content.strip()
        )
        vision = str(payload.get("vision_observation", "")).strip()
        goal_sources = payload.get("goal_sources")
        system = _SYSTEM_PROMPTS[self.spec.kind]
        user = (
            f"Character: {context.snapshot.character_name}\n"
            f"Authoritative snapshot revision: {context.snapshot.revision}\n"
            f"Recent transcript:\n{transcript or '(empty)'}\n\n"
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
            for raw in items[:8]:
                if not isinstance(raw, dict):
                    continue
                summary = str(raw.get("summary", "")).strip()
                if not summary:
                    continue
                item_provenance = provenance
                if "evidence" in raw:
                    # A quote the user never said means the fact came from the
                    # assistant or from nowhere; it must not become a memory.
                    quote = str(raw.get("evidence") or "")
                    source = _line_quoted(quote, user_lines)
                    # Judge the line the fact came from, not the latest turn: a
                    # question after "my name is Dawn" must not veto the name, and
                    # a fact after a quotation must not launder the quotation.
                    if source is None or classify_user_text(source) != "asserted_fact":
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
                    provenance=provenance,
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
            proposals.append(
                context.proposal(
                    "cognition.reflection_candidate",
                    value,
                    confidence=confidence,
                    provenance=provenance,
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
        # The last line of the user that memory extraction has been given.
        # The language the workers write in. Empty: the language the user
        # writes in, which the model has to work out from the transcript.
        self.output_language = ""
        # None: nothing of the conversation that begins with _first_line.
        self._read_up_to: Message | None = None
        self._first_line: Message | None = None
        self._reading_from_known = False
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
        if not self._reading_from_known and openers:
            # Before the first turn seen here nothing is this runtime's to read.
            self._read_up_to = openers[-2] if len(openers) > 1 else None
            self._first_line = openers[0]
            self._reading_from_known = True
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
            if revision % spec.every_n_revisions != 0:
                self._emit(spec.kind, "cadence_skipped", revision, event.id)
                continue
            key = (spec.kind, event.id)
            if key in self._scheduled_keys:
                self._emit(spec.kind, "deduplicated", revision, event.id)
                continue
            job = payload
            if spec.kind is BackgroundCognitionKind.MEMORY_EXTRACTION:
                unread = self._unread(openers)
                if unread is not None:
                    job = {**payload, "user_lines_to_extract": unread}
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
                self._read_up_to = openers[-1]
                self._reading_from_known = True
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

    def _unread(self, openers: Sequence[Message]) -> list[str] | None:
        """What the user said since memory extraction last ran, or None when
        that cannot be told: the conversation was replaced or trimmed past it.

        Counting turns does not find these lines. An interrupted turn leaves a
        line in the conversation without being a turn, and a turn the host
        keeps out of the conversation is a turn without a line.
        """
        if not openers or not self._reading_from_known:
            return None
        start = 0
        if self._read_up_to is None:
            if openers[0] is not self._first_line:
                return None  # another conversation, or this one was trimmed
        else:
            last = next(
                (
                    index
                    for index in range(len(openers) - 1, -1, -1)
                    if openers[index] is self._read_up_to
                ),
                None,
            )
            if last is None:
                return None
            start = last + 1
        return [message.content for message in openers[start:] if message.role == "user"]

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
        state_values = [
            {"id": "emotion", "value": state.emotion},
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
        ),
        BackgroundCognitionKind.EMOTION_ANALYSIS: (
            "Infer the user's currently expressed emotion conservatively; do not diagnose hidden mental states. "
            "The transcript holds the user's side of the conversation only. valence is how pleasant the user feels (-1 unpleasant, 0 neutral, "
            "1 pleasant). stance is how the user is treating the character in these lines (-1 hostile or "
            "insulting, 0 neutral, 1 warm, caring or grateful); stance is 0 when the user is not addressing "
            "the character's person. "
            "Return JSON: {\"emotion\":str,\"intensity\":0..1,\"valence\":-1..1,\"stance\":-1..1,"
            "\"confidence\":0..1,\"evidence\":[str]}."
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
    # Models re-space quotes ("研究AI" comes back as "研究 AI"); spacing is not evidence.
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


def _line_quoted(quote: str, lines: list[str]) -> str | None:
    wanted = _squeeze(quote)
    if not wanted:
        return None
    return next((line for line in lines if wanted in _squeeze(line)), None)


def _is_vision_event(event: CharacterEvent) -> bool:
    return (
        event.type in {"vision_observation", "multimodal_user_message"}
        or event.source.startswith("vision:")
    )
