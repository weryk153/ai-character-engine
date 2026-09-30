from __future__ import annotations

from dataclasses import dataclass

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext, Tracer

from .base import VisionProvider
from .errors import VisionInputError
from .memory import VisionMemoryPolicy
from .models import ImageInput, VisionFrame, VisionPipelineResult
from .sampling import FrameGate
from .validation import VisionInputPolicy


@dataclass(slots=True, frozen=True)
class VisionPipelineConfig:
    max_context_chars: int = 4_000
    include_user_prompt: bool = True

    def __post_init__(self) -> None:
        if self.max_context_chars < 128:
            raise ValueError("max_context_chars must be >= 128")


class VisionPipeline:
    def __init__(
        self,
        *,
        provider: VisionProvider,
        input_policy: VisionInputPolicy | None = None,
        frame_gate: FrameGate | None = None,
        memory_policy: VisionMemoryPolicy | None = None,
        config: VisionPipelineConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.input_policy = input_policy or VisionInputPolicy()
        self.frame_gate = frame_gate or FrameGate()
        self.memory_policy = memory_policy or VisionMemoryPolicy()
        self.config = config or VisionPipelineConfig()
        self.tracer = tracer or Tracer()

    async def analyze_frame(
        self,
        frame: VisionFrame,
        *,
        prompt: str | None = None,
        trace_context: TraceContext | None = None,
    ) -> VisionPipelineResult | None:
        context = trace_context or TraceContext.create()
        with self.tracer.span(
            "vision.capture",
            context=context,
            attributes={"frame_id": frame.id, "source_type": frame.source_type},
        ) as capture_span:
            image = frame.image
            capture_span.set_attribute("mime_type", image.mime_type)
            capture_span.set_attribute("source", image.source)

        with self.tracer.span("vision.preprocess", context=context) as preprocess_span:
            width, height, size = self.input_policy.validate(frame.image)
            preprocess_span.set_attribute("width", width)
            preprocess_span.set_attribute("height", height)
            preprocess_span.set_attribute("bytes", size)
            accepted = self.frame_gate.accept(frame)
            preprocess_span.set_attribute("accepted", accepted)
            if not accepted:
                return None

        with self.tracer.span("vision.infer", context=context) as infer_span:
            analysis = await self.provider.analyze(frame.image, prompt=prompt)
            infer_span.set_attribute("provider", analysis.provider)
            infer_span.set_attribute("model", analysis.model)
            infer_span.set_attribute("width", width)
            infer_span.set_attribute("height", height)
            infer_span.set_attribute("bytes", size)
            if "latency_ms" in analysis.metadata:
                infer_span.set_attribute("provider_latency_ms", analysis.metadata["latency_ms"])

        with self.tracer.span("vision.context", context=context) as context_span:
            observation = analysis.text.strip()
            truncated = len(observation) > self.config.max_context_chars
            if truncated:
                observation = observation[: self.config.max_context_chars].rstrip() + "…"
            content_parts = []
            if prompt and self.config.include_user_prompt:
                content_parts.append(f"User visual prompt: {prompt.strip()}")
            content_parts.append(f"Visual observation ({frame.source_type}): {observation}")
            event_content = "\n".join(content_parts)
            memory_payload = self.memory_policy.event_payload(frame, analysis)
            persistent = memory_payload["vision_memory"] == "persistent"
            payload = {
                "vision": {
                    "frame_id": frame.id,
                    "source_type": frame.source_type,
                    "mime_type": frame.image.mime_type,
                    "width": width,
                    "height": height,
                    "bytes": size,
                    "provider": analysis.provider,
                    "model": analysis.model,
                    "tags": list(analysis.tags),
                    "confidence": analysis.confidence,
                    "truncated": truncated,
                },
                **memory_payload,
            }
            context_span.set_attribute("context_chars", len(event_content))
            context_span.set_attribute("persistent_memory", persistent)

        return VisionPipelineResult(
            frame=frame,
            analysis=analysis,
            event_content=event_content,
            event_payload=payload,
            persistent_memory=persistent,
        )

    async def to_character_event(
        self,
        frame: VisionFrame,
        *,
        prompt: str | None = None,
        trace_context: TraceContext | None = None,
    ) -> CharacterEvent | None:
        result = await self.analyze_frame(frame, prompt=prompt, trace_context=trace_context)
        if result is None:
            return None
        event_type = "multimodal_user_message" if prompt else "vision_observation"
        return CharacterEvent(
            type=event_type,
            source=f"vision:{frame.source_type}",
            content=result.event_content,
            payload=result.event_payload,
        )

    async def image_event(
        self,
        image: ImageInput,
        *,
        prompt: str | None = None,
        source_type: str = "upload",
        remember: bool | None = None,
        trace_context: TraceContext | None = None,
    ) -> CharacterEvent | None:
        frame_metadata = {}
        if remember is not None:
            frame_metadata["remember"] = remember
        frame = VisionFrame(image=image, source_type=source_type, metadata=frame_metadata)  # type: ignore[arg-type]
        return await self.to_character_event(frame, prompt=prompt, trace_context=trace_context)
