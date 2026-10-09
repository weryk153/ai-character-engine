from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import time
from collections.abc import Awaitable, Callable

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import (
    BELIEF_LINE,
    MEMORY_LINE,
    USER_SEEMS_LINE,
    ContextBuilder,
    belief_line,
    is_picture_description,
    is_turn_context,
    one_line,
    without_lines,
)
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.base import LLMClient
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.memory.manager import MemoryManager, event_query
from ai_character_engine.long_term_cognition import LongTermCognitionManager
from ai_character_engine.goals import GoalManager
from ai_character_engine.memory.revision import guard_revision_response_text, revision_context
from ai_character_engine.memory.retriever import (
    AsyncMemoryRetriever,
    MemoryRetriever,
    retrieve_with_trace_async,
)
from ai_character_engine.observability import TraceContext, Tracer, TurnTelemetry
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.runtime.coordination import (
    PartialTurnError, RuntimeBusyError, TurnCoordinator, TurnEffects, TurnLease,
)
from ai_character_engine.state.models import CharacterState, StatePatch
from ai_character_engine.state.policy import CharacterStatePolicy, NoopStatePolicy
from ai_character_engine.tools.executor import ToolExecutor
from ai_character_engine.tools.models import ToolResult
from ai_character_engine.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CharacterRuntime:
    """Orchestrates one character's Observe -> Decide -> Act -> Record cycle."""

    def __init__(
        self,
        *,
        character: CharacterProfile,
        llm: LLMClient,
        context_builder: ContextBuilder | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        state: CharacterState | None = None,
        state_policy: CharacterStatePolicy | None = None,
        memory_manager: MemoryManager | None = None,
        max_history_messages: int = 20,
        max_tool_rounds: int = 5,
        memory_retriever: MemoryRetriever | AsyncMemoryRetriever | None = None,
        retrieval_limit: int = 5,
        memory_scope_id: str | None = None,
        long_term_cognition: LongTermCognitionManager | None = None,
        cognition_scope_id: str | None = None,
        goal_manager: GoalManager | None = None,
        goal_scope_id: str | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if max_history_messages < 0:
            raise ValueError("max_history_messages must be >= 0")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be >= 1")

        self.character = character
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder()
        self.state = state or CharacterState()
        self.state_policy = state_policy or NoopStatePolicy()
        self.memory_manager = memory_manager
        self.memory_retriever = memory_retriever
        self.memory_scope_id = memory_scope_id or character.id
        if not self.memory_scope_id.strip():
            raise ValueError("memory_scope_id must not be empty")
        self.long_term_cognition = long_term_cognition
        self.cognition_scope_id = cognition_scope_id or self.memory_scope_id
        if not self.cognition_scope_id.strip():
            raise ValueError("cognition_scope_id must not be empty")
        self.goal_manager = goal_manager
        self.goal_scope_id = goal_scope_id or self.cognition_scope_id
        if not self.goal_scope_id.strip():
            raise ValueError("goal_scope_id must not be empty")
        self.retrieval_limit = retrieval_limit
        self.max_history_messages = max_history_messages
        self.max_tool_rounds = max_tool_rounds
        self.history: list[Message] = []
        # Notes of the "transcript" context placement: (the message a note was
        # placed in front of, the note). History itself keeps only what was
        # said; the builder puts the notes back where they were.
        self.context_notes: list[tuple[Message, Message]] = []

        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_executor = tool_executor or ToolExecutor(self.tool_registry)
        self.tracer = tracer or Tracer()
        self.turns = TurnCoordinator()
        # Set to True to forward text as it is generated even when tools are
        # registered. Held back by default: a round that ends in a tool call may
        # have produced provisional text first. A voice host usually prefers
        # the words now, since otherwise nothing can be spoken until the whole
        # reply is finished.
        self.stream_text_with_tools = False

    async def run_turn(self, user_message: str) -> LLMResponse:
        """Compatibility helper for chat applications.

        A user message is modeled as a CharacterEvent internally. New host
        applications can call process_event() directly for non-chat events.
        """

        result = await self.process_event(CharacterEvent.user_message(user_message))
        return result.response

    async def process_event(
        self,
        event: CharacterEvent,
        *,
        trace_context: TraceContext | None = None,
        turn_lease: TurnLease | None = None,
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CharacterRunResult:
        """Process an exclusive turn; restore local history/state on any failure.

        Tools and storage are not transactional. PartialTurnError (or the
        requires_review attribute on propagated cancellation) marks uncertain
        external effects. A host must reconcile those before retrying.
        """
        if turn_lease is None:
            with self.turns.reserve() as lease:
                return await self.process_event(
                    event, trace_context=trace_context, turn_lease=lease,
                    on_text_delta=on_text_delta,
                )
        with self.turns.execute(turn_lease):
            history = copy.deepcopy(self.history)
            state = copy.deepcopy(self.state)
            context_notes = list(self.context_notes)
            effects = TurnEffects()
            try:
                return await self._process_event(
                    event, trace_context=trace_context, effects=effects,
                    on_text_delta=on_text_delta,
                )
            except BaseException as exc:
                self.history, self.state = history, state
                self.context_notes = context_notes
                if isinstance(exc, asyncio.CancelledError):
                    # Preserve the exact asyncio exception type for wait_for/timeout.
                    exc.requires_review = effects.requires_review
                    raise
                if isinstance(exc, Exception) and effects.requires_review:
                    raise PartialTurnError(effects.phase) from exc
                raise
            finally:
                turn_lease.requires_review |= effects.requires_review

    async def _process_event(
        self,
        event: CharacterEvent,
        *,
        trace_context: TraceContext | None = None,
        effects: TurnEffects,
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CharacterRunResult:
        """Process one observation through the character action loop.

        State transition: deterministic host policy may update runtime state.
        Observe: build context from profile, state, history, and event.
        Decide: ask the LLM for text and/or tool calls.
        Act: execute requested tools through ToolExecutor.
        Observe action results: policy may update state after tool execution.
        Record: persist event, tool trace, and final character response.
        """

        trace_context = (trace_context or TraceContext.create()).child(
            character_id=self.character.id
        )
        turn_started = time.perf_counter()

        with self.tracer.span(
            "runtime.process_event",
            context=trace_context,
            attributes={"event_id": event.id, "event_type": event.type},
        ) as root_span:
            child_base = trace_context.child(parent_span_id=root_span.span_id)
            state_before = self.state.snapshot()
            state_updates: list[StatePatch] = []
            event_patch = self.state_policy.on_event(event, state_before)
            self._apply_state_patch(event_patch, state_updates)

            revision_plan = (
                self.memory_manager.preview_revision(character_id=self.memory_scope_id, event=event)
                if self.memory_manager is not None
                else None
            )
            memory_operation_context = revision_context(revision_plan) if revision_plan is not None else None

            retrieval = None
            rewrite_context = self._retrieval_rewrite_context()
            with self.tracer.span("memory.retrieve", context=child_base) as retrieval_span:
                if self.memory_retriever is not None:
                    retrieval = await retrieve_with_trace_async(
                        self.memory_retriever, character_id=self.memory_scope_id,
                        query=event_query(event), limit=self.retrieval_limit,
                        rewrite_context=rewrite_context,
                    )
                elif self.memory_manager is not None:
                    retrieval = await self.memory_manager.retrieve_for_event_with_trace_async(
                        character_id=self.memory_scope_id, event=event,
                        rewrite_context=rewrite_context,
                    )
                if retrieval is not None:
                    retrieval_span.set_attribute("strategy", retrieval.trace.strategy)
                    retrieval_span.set_attribute("selected", len(retrieval.memories))
                    retrieval_span.set_attribute("elapsed_ms", retrieval.trace.elapsed_ms)
            retrieved_memories = list(retrieval.memories) if retrieval else []
            active_beliefs = (
                self.long_term_cognition.active_beliefs(character_id=self.cognition_scope_id)
                if self.long_term_cognition is not None
                else ()
            )
            active_goals = (
                self.goal_manager.active_goals(character_id=self.goal_scope_id)
                if self.goal_manager is not None
                else ()
            )

            tool_definitions = self.tool_registry.definitions() or None
            self._withdraw_what_no_longer_holds(active_beliefs)
            with self.tracer.span("context.build", context=child_base) as context_span:
                context_build = self.context_builder.build_for_event_with_trace(
                    character=self.character,
                    history=self.history,
                    event=event,
                    state=self.state,
                    memories=retrieved_memories,
                    tools=tool_definitions,
                    memory_operation_context=memory_operation_context,
                    beliefs=active_beliefs,
                    goals=active_goals,
                    notes=self.context_notes,
                )
                context_span.set_attribute(
                    "estimated_total_tokens", context_build.trace.estimated_total_tokens
                )
                context_span.set_attribute(
                    "included_history_messages", context_build.trace.included_history_messages
                )
                context_span.set_attribute("included_memories", context_build.trace.included_memories)
                context_span.set_attribute("included_beliefs", context_build.trace.included_beliefs)
                context_span.set_attribute("included_goals", context_build.trace.included_goals)
            turn_messages = list(context_build.messages)
            # The event is the last message built; this turn's context may sit
            # in front of it. History keeps only what happened from the event
            # on. A note of the "transcript" placement is kept next to it, so
            # that the next prompt extends this one; the whole context of the
            # "turn" placement is not, it would be sent again on every turn.
            event_index = len(turn_messages) - 1
            # A picture's description sits between the note and the event.
            note_index = event_index - 1
            while note_index > 0 and is_picture_description(turn_messages[note_index]):
                note_index -= 1
            if (
                self.context_builder.context_placement == "transcript"
                and note_index > 0
                and is_turn_context(turn_messages[note_index])
                and not any(turn_messages[note_index] is note for _, note in self.context_notes)
            ):
                self.context_notes.append(
                    (turn_messages[event_index], turn_messages[note_index])
                )
            total_input_tokens = 0
            total_output_tokens = 0
            total_latency_ms = 0.0
            total_estimated_cost = 0.0
            total_queue_wait_ms = 0.0
            last_model: str | None = None
            executed_results: list[ToolResult] = []

            for round_index in range(self.max_tool_rounds):
                with self.tracer.span(
                    "llm.generate",
                    context=child_base,
                    attributes={"round": round_index + 1},
                ) as llm_span:
                    response = await self._generate_response(
                        turn_messages,
                        tools=tool_definitions,
                        on_text_delta=on_text_delta,
                    )
                    llm_span.set_attribute("model", response.model)
                    llm_span.set_attribute("input_tokens", response.input_tokens)
                    llm_span.set_attribute("output_tokens", response.output_tokens)
                    llm_span.set_attribute("provider_latency_ms", response.latency_ms)
                    llm_span.set_attribute("tool_calls", len(response.tool_calls))
                    gateway_meta = response.metadata.get("gateway") if response.metadata else None
                    if isinstance(gateway_meta, dict):
                        llm_span.set_attribute("gateway.route", gateway_meta.get("route_name"))
                        llm_span.set_attribute("gateway.endpoint", gateway_meta.get("selected_endpoint_id"))
                        llm_span.set_attribute("gateway.fallback_used", gateway_meta.get("fallback_used"))
                        llm_span.set_attribute("gateway.attempts", len(gateway_meta.get("attempts", [])))
                    deployment_meta = response.metadata.get("deployment") if response.metadata else None
                    if isinstance(deployment_meta, dict):
                        llm_span.set_attribute("deployment.backend", deployment_meta.get("backend"))
                        llm_span.set_attribute("deployment.endpoint_type", deployment_meta.get("endpoint_type"))
                        llm_span.set_attribute("deployment.base_url", deployment_meta.get("base_url"))
                        llm_span.set_attribute("deployment.model", deployment_meta.get("model"))
                        runtime_meta = deployment_meta.get("runtime")
                        if isinstance(runtime_meta, dict):
                            llm_span.set_attribute("deployment.device", runtime_meta.get("device"))
                            llm_span.set_attribute("deployment.quantization", runtime_meta.get("quantization"))
                            llm_span.set_attribute("deployment.context_length", runtime_meta.get("context_length"))
                    inference_meta = response.metadata.get("inference") if response.metadata else None
                    if isinstance(inference_meta, dict):
                        llm_span.set_attribute("inference.profile", inference_meta.get("profile"))
                        llm_span.set_attribute("inference.queue_wait_ms", inference_meta.get("queue_wait_ms"))
                        llm_span.set_attribute("inference.ttft_ms", inference_meta.get("ttft_ms"))
                        llm_span.set_attribute(
                            "inference.decode_tokens_per_second",
                            inference_meta.get("decode_tokens_per_second"),
                        )
                        llm_span.set_attribute("inference.estimated_cost", inference_meta.get("estimated_cost"))
                last_model = response.model or last_model
                total_input_tokens += response.input_tokens or 0
                total_output_tokens += response.output_tokens or 0
                total_latency_ms += response.latency_ms or 0.0
                response_inference = response.metadata.get("inference") if response.metadata else None
                if isinstance(response_inference, dict):
                    if isinstance(response_inference.get("estimated_cost"), (int, float)):
                        total_estimated_cost += float(response_inference["estimated_cost"])
                    if isinstance(response_inference.get("queue_wait_ms"), (int, float)):
                        total_queue_wait_ms += float(response_inference["queue_wait_ms"])

                if not response.tool_calls:
                    final_metadata = dict(response.metadata)
                    final_inference = final_metadata.get("inference")
                    if isinstance(final_inference, dict):
                        final_inference = dict(final_inference)
                        final_inference["estimated_cost"] = total_estimated_cost
                        final_inference["queue_wait_ms"] = total_queue_wait_ms
                        final_metadata["inference"] = final_inference
                    final_response = LLMResponse(
                        text=response.text,
                        model=last_model,
                        input_tokens=total_input_tokens or response.input_tokens,
                        output_tokens=total_output_tokens or response.output_tokens,
                        latency_ms=total_latency_ms or response.latency_ms,
                        metadata=final_metadata,
                    )
                    if revision_plan is not None:
                        guarded_text, guard_applied = guard_revision_response_text(
                            final_response.text, revision_plan
                        )
                        if guard_applied:
                            guarded_metadata = dict(final_response.metadata)
                            guarded_metadata["memory_operation_guard"] = {
                                "requested_action": revision_plan.requested_action,
                                "planned_action": revision_plan.action,
                                "matched_targets": len(revision_plan.target_ids),
                            }
                            final_response = LLMResponse(
                                text=guarded_text,
                                tool_calls=final_response.tool_calls,
                                model=final_response.model,
                                input_tokens=final_response.input_tokens,
                                output_tokens=final_response.output_tokens,
                                latency_ms=final_response.latency_ms,
                                metadata=guarded_metadata,
                            )
                    state_after = self.state.snapshot()
                    self._commit_event(
                        turn_messages=turn_messages,
                        new_turn_start=event_index,
                        response=final_response,
                    )
                    memory_written = None
                    if self.memory_manager is not None:
                        with self.tracer.span("memory.record", context=child_base):
                            effects.begin("memory_record")
                            memory_written = self.memory_manager.record_interaction(
                                character_id=self.memory_scope_id,
                                event=event,
                                response=final_response,
                                state_before=state_before,
                                state_after=state_after,
                                revision_plan=revision_plan,
                            )
                    result = CharacterRunResult(
                        event=event,
                        response=final_response,
                        tool_results=tuple(executed_results),
                        state_updates=tuple(state_updates),
                        state_before=state_before,
                        state_after=state_after,
                        retrieved_memories=tuple(retrieved_memories),
                        memory_written=memory_written,
                        ledger_entry=(
                            self.memory_manager.last_ledger_entry
                            if self.memory_manager is not None
                            else None
                        ),
                        memory_consolidation=(
                            self.memory_manager.last_consolidation_result
                            if self.memory_manager is not None
                            else None
                        ),
                        memory_revision=(
                            self.memory_manager.last_revision_result
                            if self.memory_manager is not None
                            else None
                        ),
                        context_trace=context_build.trace,
                        retrieval_trace=retrieval.trace if retrieval else None,
                        rounds=round_index + 1,
                    )
                    duration_ms = (time.perf_counter() - turn_started) * 1000
                    self.tracer.sink.record_turn(
                        TurnTelemetry(
                            trace_id=trace_context.trace_id,
                            request_id=trace_context.request_id,
                            session_id=trace_context.session_id,
                            user_id=trace_context.user_id,
                            event_id=event.id,
                            character_id=self.character.id,
                            duration_ms=duration_ms,
                            model=final_response.model,
                            rounds=result.rounds,
                            tool_count=len(result.tool_results),
                            input_tokens=final_response.input_tokens,
                            output_tokens=final_response.output_tokens,
                            model_latency_ms=final_response.latency_ms,
                            estimated_context_tokens=context_build.trace.estimated_total_tokens,
                            retrieved_memory_count=len(retrieved_memories),
                            memory_written=memory_written is not None,
                            retrieval_strategy=retrieval.trace.strategy if retrieval else None,
                            queue_wait_ms=(
                                float(final_response.metadata.get("inference", {}).get("queue_wait_ms"))
                                if isinstance(final_response.metadata.get("inference"), dict)
                                and isinstance(final_response.metadata.get("inference", {}).get("queue_wait_ms"), (int, float))
                                else None
                            ),
                            ttft_ms=(
                                float(final_response.metadata.get("inference", {}).get("ttft_ms"))
                                if isinstance(final_response.metadata.get("inference"), dict)
                                and isinstance(final_response.metadata.get("inference", {}).get("ttft_ms"), (int, float))
                                else None
                            ),
                            decode_tokens_per_second=(
                                float(final_response.metadata.get("inference", {}).get("decode_tokens_per_second"))
                                if isinstance(final_response.metadata.get("inference"), dict)
                                and isinstance(final_response.metadata.get("inference", {}).get("decode_tokens_per_second"), (int, float))
                                else None
                            ),
                            estimated_cost=(
                                float(final_response.metadata.get("inference", {}).get("estimated_cost"))
                                if isinstance(final_response.metadata.get("inference"), dict)
                                and isinstance(final_response.metadata.get("inference", {}).get("estimated_cost"), (int, float))
                                else None
                            ),
                            inference_profile=(
                                str(final_response.metadata.get("inference", {}).get("profile"))
                                if isinstance(final_response.metadata.get("inference"), dict)
                                and final_response.metadata.get("inference", {}).get("profile") is not None
                                else None
                            ),
                        )
                    )
                    logger.debug(
                        "character_event_complete character_id=%s event_id=%s event_type=%s rounds=%s tools=%s state_updates=%s input_tokens=%s output_tokens=%s latency_ms=%s trace_id=%s",
                        self.character.id, event.id, event.type, result.rounds,
                        len(result.tool_results), len(result.state_updates),
                        final_response.input_tokens, final_response.output_tokens,
                        final_response.latency_ms, trace_context.trace_id,
                    )
                    return result

                assistant_tool_message = Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
                turn_messages.append(assistant_tool_message)

                for call in response.tool_calls:
                    effects.begin("tool_execute")
                    with self.tracer.span(
                        "tool.execute",
                        context=child_base,
                        attributes={"tool": call.name, "call_id": call.call_id},
                    ) as tool_span:
                        tool_result = await self.tool_executor.execute(call)
                        tool_span.set_attribute("is_error", tool_result.is_error)
                    executed_results.append(tool_result)
                    turn_messages.append(
                        Message(
                            role="tool",
                            content=tool_result.output,
                            tool_result=tool_result,
                        )
                    )
                    tool_patch = self.state_policy.on_tool_result(
                        tool_result,
                        self.state.snapshot(),
                    )
                    self._apply_state_patch(tool_patch, state_updates)

            effects.phase = "max_tool_rounds"
            raise RuntimeError(
                f"character exceeded max_tool_rounds={self.max_tool_rounds} without a final response"
            )


    async def _generate_response(
        self,
        messages: list[Message],
        *,
        tools,
        on_text_delta: Callable[[str], Awaitable[None] | None] | None,
    ) -> LLMResponse:
        """Generate one LLM round with optional provider streaming.

        Text is forwarded immediately when the round has no tool definitions,
        or when the host set ``stream_text_with_tools``. Otherwise deltas are
        buffered until the normalized final response proves that the round is
        user-visible rather than an intermediate tool-call turn. That
        conservative boundary avoids speaking provisional text. Function-call
        arguments never appear in text deltas either way.
        """
        immediate = not tools or self.stream_text_with_tools

        async def notify(text: str) -> None:
            if on_text_delta is None or not text:
                return
            result = on_text_delta(text)
            if inspect.isawaitable(result):
                await result

        stream_generate = getattr(self.llm, "stream_generate", None)
        if on_text_delta is not None and callable(stream_generate):
            parts: list[str] = []
            final: LLMResponse | None = None
            async for update in stream_generate(messages, tools=tools):
                text = getattr(update, "text", "") or ""
                if text:
                    parts.append(text)
                    if immediate:
                        await notify(text)
                if getattr(update, "final", False):
                    final = getattr(update, "response", None)
            if final is None:
                raise RuntimeError("streaming LLM ended without a final response")
            joined = "".join(parts)
            if joined and not final.text:
                final = LLMResponse(
                    text=joined,
                    tool_calls=final.tool_calls,
                    model=final.model,
                    input_tokens=final.input_tokens,
                    output_tokens=final.output_tokens,
                    latency_ms=final.latency_ms,
                    metadata=dict(final.metadata),
                )
            if not immediate and not final.tool_calls and joined:
                for part in parts:
                    await notify(part)
            metadata = dict(final.metadata)
            streaming = dict(metadata.get("streaming") or {})
            streaming.update(
                {
                    "runtime_streamed_text": bool(immediate and parts),
                    "tool_safe_buffering": not immediate,
                    "text_chunks": len(parts),
                }
            )
            metadata["streaming"] = streaming
            return LLMResponse(
                text=final.text,
                tool_calls=final.tool_calls,
                model=final.model,
                input_tokens=final.input_tokens,
                output_tokens=final.output_tokens,
                latency_ms=final.latency_ms,
                metadata=metadata,
            )

        response = await self.llm.generate(messages, tools=tools)
        if on_text_delta is not None and response.text and not response.tool_calls:
            await notify(response.text)
        return response

    def _retrieval_rewrite_context(self) -> tuple[str, ...]:
        """Small provider-neutral context for optional retrieval query rewrite."""
        recent = [message.content for message in self.history[-6:] if message.content.strip()]
        character_hint = (
            f"Character {self.character.name}: {self.character.description}"
        )
        return tuple([character_hint, *recent])

    def reset_history(self) -> None:
        if self.turns.busy:
            raise RuntimeBusyError("Cannot reset an active character turn.")
        self.history.clear()

    def reset_state(self, state: CharacterState | None = None) -> None:
        if self.turns.busy:
            raise RuntimeBusyError("Cannot reset an active character turn.")
        self.state = state or CharacterState()

    def _apply_state_patch(
        self,
        patch: StatePatch | None,
        state_updates: list[StatePatch],
    ) -> None:
        if patch is None or patch.is_noop:
            return
        self.state.apply(patch)
        state_updates.append(patch)

    def _commit_event(
        self,
        *,
        turn_messages: list[Message],
        new_turn_start: int,
        response: LLMResponse,
    ) -> None:
        # ContextBuilder prepends system + previous history. Persist only this event cycle.
        self.history.extend(turn_messages[new_turn_start:])
        self.history.append(Message(role="assistant", content=response.text))
        self._trim_history()

    def withdraw_from_notes(self, gone: Callable[[str], bool]) -> None:
        """Take every line ``gone`` is true for out of the notes kept with the
        conversation. The server then reads the conversation again from the
        first note that changed; that is the price of taking something back."""
        kept = []
        for anchor, note in self.context_notes:
            rest = without_lines(note, gone)
            if rest is not None:
                kept.append((anchor, rest))
        self.context_notes = kept

    def _withdraw_what_no_longer_holds(self, active_beliefs) -> None:
        """A memory she was asked to forget, a belief that was retracted, an
        observation that was cleared: notes written while they held are still
        in the conversation."""
        if not self.context_notes:
            return
        remembered = None
        if self.memory_manager is not None:
            remembered = {
                one_line(record.summary)
                for record in self.memory_manager.store.list_for_character(
                    self.memory_scope_id
                )
                if record.is_active
            }
        believed = {belief_line(record) for record in active_beliefs}
        observed = isinstance(self.state.custom.get("observed_user_emotion"), dict)

        def gone(line: str) -> bool:
            if line.startswith(MEMORY_LINE):
                return remembered is not None and line.split("]: ", 1)[-1] not in remembered
            if line.startswith(BELIEF_LINE):
                return line not in believed
            return line.startswith(USER_SEEMS_LINE) and not observed

        self.withdraw_from_notes(gone)

    def _trim_history(self) -> None:
        if self.max_history_messages == 0:
            self.history.clear()
            self.context_notes.clear()
            return

        keep = self.max_history_messages
        if len(self.history) > keep >= 16:
            # Old turns leave a quarter of the conversation at a time. Dropping
            # the oldest one on every turn gives every prompt a new beginning,
            # and an inference server can then reuse nothing of the previous
            # prompt. A conversation kept shorter than this is read again in
            # no time.
            keep = keep * 3 // 4

        # Preserve complete event cycles. A raw slice could keep a tool result
        # while dropping the assistant function call that produced it.
        while len(self.history) > keep:
            next_event_index = next(
                (
                    index
                    for index, message in enumerate(self.history[1:], start=1)
                    if message.role in {"user", "event"}
                ),
                None,
            )
            if next_event_index is None:
                # The newest complete cycle may itself exceed the soft cap. Keep
                # it intact rather than corrupting the tool-call sequence.
                break
            del self.history[:next_event_index]
        # A note goes with the message it was written for.
        self.context_notes = [
            (anchor, note)
            for anchor, note in self.context_notes
            if any(anchor == kept for kept in self.history)
        ]
