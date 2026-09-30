from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.errors import LLMError
from ai_character_engine.observability import ProductionEvalRunner, TraceContext, Tracer
from ai_character_engine.session import (
    CharacterRuntimeFactory,
    ManagedCharacterSession,
    SessionManager,
    SessionNotFoundError,
    SessionRecord,
    SessionUnavailableError,
)
from ai_character_engine.tools.errors import ToolError
from ai_character_engine.vision import ImageInput, VisionFrame, VisionInputError, VisionPipeline, VisionProviderError, VisionRateLimitError

from .errors import (
    CharacterServiceError,
    ServiceNotFoundError,
    ServiceRateLimitError,
    ServiceSessionUnavailableError,
    ServiceTimeoutError,
    ServiceToolError,
    ServiceUpstreamError,
    ServiceValidationError,
)
from .models import ImageMessageRequest, MessageResponse, SessionCreateRequest, VisionMessageResponse
from .streaming import CharacterStreamSource, ServiceStreamEvent

logger = logging.getLogger(__name__)


class CharacterService:
    """Application/service boundary above SessionManager and CharacterRuntime."""

    def __init__(
        self,
        *,
        runtime_factory: CharacterRuntimeFactory,
        session_manager: SessionManager,
        stream_source: CharacterStreamSource,
        default_timeout_seconds: float = 30.0,
        tracer: Tracer | None = None,
        production_eval: ProductionEvalRunner | None = None,
        readiness_probe: Callable[[], Awaitable[dict]] | None = None,
        vision_pipeline: VisionPipeline | None = None,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.session_manager = session_manager
        self.stream_source = stream_source
        self.default_timeout_seconds = default_timeout_seconds
        self.tracer = tracer or runtime_factory.tracer
        self.production_eval = production_eval
        self.readiness_probe = readiness_probe
        self.vision_pipeline = vision_pipeline


    async def readiness(self) -> dict:
        if self.readiness_probe is None:
            return {"ready": True}
        return await self.readiness_probe()

    def create_session(self, request: SessionCreateRequest) -> SessionRecord:
        try:
            return self.runtime_factory.create(
                user_id=request.user_id,
                character_id=request.character_id,
                session_id=request.session_id,
                metadata=request.metadata,
                ttl_seconds=request.ttl_seconds,
            ).record
        except KeyError as exc:
            raise ServiceNotFoundError(str(exc)) from exc
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc

    def get_session(self, session_id: str) -> SessionRecord:
        try:
            return self.session_manager.require(session_id, require_active=False)
        except SessionNotFoundError as exc:
            raise ServiceNotFoundError(f"session not found: {session_id}") from exc

    def close_session(self, session_id: str) -> SessionRecord:
        try:
            return self.session_manager.close(session_id)
        except SessionNotFoundError as exc:
            raise ServiceNotFoundError(f"session not found: {session_id}") from exc
        except SessionUnavailableError as exc:
            raise ServiceSessionUnavailableError(str(exc)) from exc

    def restore_active_session(self, session_id: str) -> ManagedCharacterSession:
        try:
            return self.runtime_factory.restore(session_id)
        except SessionNotFoundError as exc:
            raise ServiceNotFoundError(f"session not found: {session_id}") from exc
        except SessionUnavailableError as exc:
            raise ServiceSessionUnavailableError(str(exc)) from exc
        except KeyError as exc:
            raise ServiceNotFoundError(str(exc)) from exc

    async def send_message(
        self,
        session_id: str,
        content: str,
        *,
        request_id: str,
        trace_id: str,
        timeout_seconds: float | None = None,
    ) -> MessageResponse:
        session = self.restore_active_session(session_id)
        timeout = timeout_seconds or self.default_timeout_seconds
        context = TraceContext(
            trace_id=trace_id, request_id=request_id, session_id=session.record.id,
            user_id=session.record.user_id, character_id=session.record.character_id,
        )
        event = CharacterEvent.user_message(content)
        history_before = tuple(
            message.content for message in session.runtime.history[-6:] if message.content.strip()
        )
        try:
            with self.tracer.span(
                "service.send_message",
                context=context,
                attributes={"session_id": session_id},
            ) as service_span:
                result = await asyncio.wait_for(
                    session.process_event(
                        event,
                        trace_context=context.child(parent_span_id=service_span.span_id),
                    ),
                    timeout=timeout,
                )
        except asyncio.TimeoutError as exc:
            raise ServiceTimeoutError(
                f"character turn timed out after {timeout}s"
            ) from exc
        except LLMError as exc:
            raise ServiceUpstreamError(str(exc)) from exc
        except ToolError as exc:
            raise ServiceToolError(str(exc)) from exc
        except CharacterServiceError:
            raise

        if self.production_eval is not None:
            try:
                await self.production_eval.evaluate_turn(
                    context=context,
                    character=session.runtime.character,
                    event=event,
                    result=result,
                    history=history_before,
                )
            except Exception:  # observability/eval must not fail user serving
                logger.exception("production_eval_failed trace_id=%s", trace_id)

        return MessageResponse(
            session_id=session_id,
            text=result.text,
            request_id=request_id,
            trace_id=trace_id,
            model=result.response.model,
            input_tokens=result.response.input_tokens,
            output_tokens=result.response.output_tokens,
            latency_ms=result.response.latency_ms,
            rounds=result.rounds,
            tool_count=len(result.tool_results),
        )

    async def send_image(
        self,
        session_id: str,
        request: ImageMessageRequest,
        *,
        request_id: str,
        trace_id: str,
    ) -> VisionMessageResponse:
        if self.vision_pipeline is None:
            raise ServiceValidationError("vision pipeline is not configured")
        session = self.restore_active_session(session_id)
        timeout = request.timeout_seconds or self.default_timeout_seconds
        context = TraceContext(
            trace_id=trace_id, request_id=request_id, session_id=session.record.id,
            user_id=session.record.user_id, character_id=session.record.character_id,
        )

        try:
            if request.image_base64 is not None:
                try:
                    raw = base64.b64decode(request.image_base64, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ServiceValidationError("image_base64 is not valid base64") from exc
                image = ImageInput.from_bytes(
                    raw, mime_type=request.mime_type, metadata=request.metadata
                )
            else:
                image = ImageInput.from_url(
                    request.image_url or "", mime_type=request.mime_type, metadata=request.metadata
                )
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc

        frame = VisionFrame(
            image=image, source_type=request.source_type,
            metadata={"remember": request.remember},
        )
        history_before = tuple(
            message.content for message in session.runtime.history[-6:] if message.content.strip()
        )

        async def run() -> VisionMessageResponse:
            with self.tracer.span(
                "service.send_image", context=context,
                attributes={"session_id": session_id, "source_type": request.source_type},
            ) as service_span:
                child = context.child(parent_span_id=service_span.span_id)
                result = await self.vision_pipeline.analyze_frame(
                    frame, prompt=request.prompt, trace_context=child
                )
                if result is None:
                    raise ServiceValidationError("vision frame was skipped by sampling/dedup policy")
                event = CharacterEvent(
                    type="multimodal_user_message" if request.prompt else "vision_observation",
                    source=f"vision:{request.source_type}",
                    content=result.event_content,
                    payload=result.event_payload,
                )
                run_result = await session.process_event(event, trace_context=child)

            if self.production_eval is not None:
                try:
                    await self.production_eval.evaluate_turn(
                        context=context, character=session.runtime.character, event=event,
                        result=run_result, history=history_before,
                    )
                except Exception:
                    logger.exception("production_eval_failed trace_id=%s", trace_id)

            vision_meta = result.event_payload.get("vision", {})
            return VisionMessageResponse(
                session_id=session_id, text=run_result.text, request_id=request_id,
                trace_id=trace_id, model=run_result.response.model,
                input_tokens=run_result.response.input_tokens,
                output_tokens=run_result.response.output_tokens,
                latency_ms=run_result.response.latency_ms, rounds=run_result.rounds,
                tool_count=len(run_result.tool_results),
                vision_provider=result.analysis.provider, vision_model=result.analysis.model,
                vision_observation=result.analysis.text,
                vision_memory="persistent" if result.persistent_memory else "ephemeral",
                source_type=str(vision_meta.get("source_type", request.source_type)),
            )

        try:
            return await asyncio.wait_for(run(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ServiceTimeoutError(f"vision turn timed out after {timeout}s") from exc
        except VisionInputError as exc:
            raise ServiceValidationError(str(exc)) from exc
        except VisionRateLimitError as exc:
            raise ServiceRateLimitError(str(exc)) from exc
        except VisionProviderError as exc:
            raise ServiceUpstreamError(str(exc)) from exc
        except LLMError as exc:
            raise ServiceUpstreamError(str(exc)) from exc
        except ToolError as exc:
            raise ServiceToolError(str(exc)) from exc

    async def stream_managed_session(
        self,
        session: ManagedCharacterSession,
        content: str,
        *,
        request_id: str,
        trace_id: str,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ServiceStreamEvent]:
        timeout = timeout_seconds or self.default_timeout_seconds
        context = TraceContext(
            trace_id=trace_id, request_id=request_id, session_id=session.record.id,
            user_id=session.record.user_id, character_id=session.record.character_id,
        )
        try:
            with self.tracer.span(
                "service.stream_message", context=context,
                attributes={"session_id": session.record.id},
            ):
                async for event in self.stream_source.stream_message(
                    session,
                    content=content,
                    request_id=request_id,
                    trace_id=trace_id,
                    timeout_seconds=timeout,
                ):
                    yield event
        except asyncio.TimeoutError as exc:
            raise ServiceTimeoutError(
                f"character stream timed out after {timeout}s"
            ) from exc
        except LLMError as exc:
            raise ServiceUpstreamError(str(exc)) from exc
        except ToolError as exc:
            raise ServiceToolError(str(exc)) from exc
    async def stream_message(
        self,
        session_id: str,
        content: str,
        *,
        request_id: str,
        trace_id: str,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ServiceStreamEvent]:
        session = self.restore_active_session(session_id)
        async for event in self.stream_managed_session(
            session,
            content,
            request_id=request_id,
            trace_id=trace_id,
            timeout_seconds=timeout_seconds,
        ):
            yield event

