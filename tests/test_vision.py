from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.observability import InMemoryObservabilitySink, TraceContext, Tracer
from ai_character_engine.service import (
    BufferedCharacterStreamSource,
    CharacterService,
    CharacterServiceConfig,
    create_app,
)
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from ai_character_engine.vision import (
    CallableVisionProvider,
    FrameGate,
    ImageInput,
    OpenAICompatibleVisionProvider,
    VisionAnalysis,
    VisionFrame,
    VisionInputError,
    VisionInputPolicy,
    VisionMemoryPolicy,
    VisionPipeline,
    VisionPipelineConfig,
)
from tests.fakes import FakeLLMClient

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z0iUAAAAASUVORK5CYII="
PNG = base64.b64decode(PNG_B64)


class FakeVision:
    def __init__(self, text: str = "A red status icon is visible on the screen.") -> None:
        self.text = text
        self.calls = 0

    async def analyze(self, image, *, prompt=None):
        self.calls += 1
        return VisionAnalysis(
            text=self.text,
            provider="fake-vlm",
            model="vision-test",
            tags=("screen", "status"),
            confidence=0.9,
            metadata={"latency_ms": 3.5},
        )


def make_pipeline(*, sink=None, gate=None, memory_policy=None):
    tracer = Tracer(sink) if sink is not None else Tracer()
    return VisionPipeline(
        provider=FakeVision(),
        input_policy=VisionInputPolicy(max_image_bytes=1024),
        frame_gate=gate or FrameGate(min_interval_seconds=0, max_frames_per_minute=20),
        memory_policy=memory_policy or VisionMemoryPolicy(),
        config=VisionPipelineConfig(max_context_chars=500),
        tracer=tracer,
    )


@pytest.mark.asyncio
async def test_vision_pipeline_normalizes_image_into_character_event():
    pipeline = make_pipeline()
    image = ImageInput.from_bytes(PNG, mime_type="image/png")
    event = await pipeline.image_event(image, prompt="What changed?", source_type="screenshot")

    assert event is not None
    assert event.type == "multimodal_user_message"
    assert event.source == "vision:screenshot"
    assert "What changed?" in event.content
    assert "red status icon" in event.content
    assert event.payload["vision"]["width"] == 1
    assert event.payload["vision"]["height"] == 1
    assert event.payload["vision_memory"] == "ephemeral"
    assert event.payload["memory_importance"] == 0.0
    assert "data" not in event.payload["vision"]


def test_image_validation_rejects_mime_mismatch_and_oversize():
    policy = VisionInputPolicy(max_image_bytes=64)
    with pytest.raises(VisionInputError, match="max_image_bytes"):
        policy.validate(ImageInput.from_bytes(PNG, mime_type="image/png"))

    policy = VisionInputPolicy(max_image_bytes=1024)
    with pytest.raises(VisionInputError, match="does not match"):
        policy.validate(ImageInput.from_bytes(PNG, mime_type="image/jpeg"))


@pytest.mark.asyncio
async def test_frame_gate_deduplicates_identical_frames():
    pipeline = make_pipeline(gate=FrameGate(min_interval_seconds=0, deduplicate=True))
    image = ImageInput.from_bytes(PNG, mime_type="image/png")
    first = await pipeline.to_character_event(VisionFrame(image=image, source_type="game"))
    second = await pipeline.to_character_event(VisionFrame(image=image, source_type="game"))
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_visual_memory_is_ephemeral_by_default_and_explicitly_persistent():
    pipeline = make_pipeline(
        gate=FrameGate(min_interval_seconds=0, deduplicate=False),
        memory_policy=VisionMemoryPolicy(importance=0.8),
    )
    image = ImageInput.from_bytes(PNG, mime_type="image/png")

    ephemeral = await pipeline.to_character_event(
        VisionFrame(image=image, source_type="camera")
    )
    persistent = await pipeline.to_character_event(
        VisionFrame(image=image, source_type="upload", metadata={"remember": True})
    )
    assert ephemeral is not None and ephemeral.payload["memory_importance"] == 0.0
    assert persistent is not None and persistent.payload["memory_importance"] == 0.8
    assert persistent.payload["vision_memory"] == "persistent"


@pytest.mark.asyncio
async def test_vision_trace_has_metadata_but_not_raw_image():
    sink = InMemoryObservabilitySink()
    pipeline = make_pipeline(sink=sink)
    image = ImageInput.from_bytes(PNG, mime_type="image/png")
    context = TraceContext.create(request_id="r1")
    event = await pipeline.image_event(image, source_type="screenshot", trace_context=context)
    assert event is not None

    spans = [span for span in sink.spans if span.trace_id == context.trace_id]
    names = {span.name for span in spans}
    assert {"vision.capture", "vision.preprocess", "vision.infer", "vision.context"} <= names
    serialized = repr([span.attributes for span in spans])
    assert PNG.hex() not in serialized
    infer = next(span for span in spans if span.name == "vision.infer")
    assert infer.attributes["provider"] == "fake-vlm"
    assert infer.attributes["bytes"] == len(PNG)


@pytest.mark.asyncio
async def test_context_summary_is_truncated_before_runtime_context():
    provider = FakeVision("x" * 1000)
    pipeline = VisionPipeline(
        provider=provider,
        frame_gate=FrameGate(min_interval_seconds=0),
        config=VisionPipelineConfig(max_context_chars=128),
    )
    event = await pipeline.image_event(
        ImageInput.from_bytes(PNG, mime_type="image/png"), source_type="upload"
    )
    assert event is not None
    assert event.payload["vision"]["truncated"] is True
    assert len(event.content) < 220


@pytest.mark.asyncio
async def test_callable_vision_provider_adapter():
    async def analyze(image, prompt):
        return VisionAnalysis(text=f"seen:{prompt}", provider="local-vlm")

    provider = CallableVisionProvider(analyze)
    result = await provider.analyze(ImageInput.from_bytes(PNG, mime_type="image/png"), prompt="hello")
    assert result.text == "seen:hello"
    assert result.provider == "local-vlm"


@pytest.mark.asyncio
async def test_openai_compatible_vision_provider_uses_data_url_without_leaking_api_details():
    captured = {}

    class Create:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                model="vlm-local",
                choices=[SimpleNamespace(message=SimpleNamespace(content="There is a cat."))],
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Create()))
    provider = OpenAICompatibleVisionProvider(
        model="vlm-local",
        base_url="http://localhost:8000/v1?secret=x",
        api_key="do-not-log",
        client=client,
    )
    result = await provider.analyze(
        ImageInput.from_bytes(PNG, mime_type="image/png"), prompt="Describe"
    )
    content = captured["messages"][0]["content"]
    assert content[0]["text"] == "Describe"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.text == "There is a cat."
    assert result.metadata["base_url"] == "http://localhost:8000/v1"


@pytest.mark.asyncio
async def test_openai_compatible_vision_provider_passes_request_options_and_a_default_prompt():
    """A local model with reasoning switched on took 23 s to describe two
    shapes. A host has to be able to switch it off and to bound the answer."""
    captured = {}

    class Create:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                model="vlm-local",
                choices=[SimpleNamespace(message=SimpleNamespace(content="A cat."))],
            )

    provider = OpenAICompatibleVisionProvider(
        model="vlm-local",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Create())),
        request_options={"max_tokens": 160, "extra_body": {"reasoning_effort": "none"}},
        default_prompt="Say in two sentences what is in the picture.",
    )
    await provider.analyze(ImageInput.from_bytes(PNG, mime_type="image/png"))

    assert captured["max_tokens"] == 160
    assert captured["extra_body"] == {"reasoning_effort": "none"}
    assert captured["messages"][0]["content"][0]["text"] == (
        "Say in two sentences what is in the picture."
    )
    with pytest.raises(ValueError):
        OpenAICompatibleVisionProvider(model="vlm-local", request_options={"model": "other"})


def build_service_client():
    character = CharacterProfile(id="mei", name="Mei", description="test")
    manager = SessionManager(default_ttl_seconds=None)
    factory = CharacterRuntimeFactory(
        characters={character.id: character},
        llm_factory=lambda _record: FakeLLMClient("I can see it."),
        session_manager=manager,
    )
    pipeline = VisionPipeline(
        provider=FakeVision("A blue button is on the screen."),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=False),
    )
    config = CharacterServiceConfig(default_timeout_seconds=1)
    service = CharacterService(
        runtime_factory=factory,
        session_manager=manager,
        stream_source=BufferedCharacterStreamSource(),
        default_timeout_seconds=1,
        vision_pipeline=pipeline,
    )
    return TestClient(create_app(service=service, config=config))


def test_service_image_endpoint_accepts_base64_and_returns_vision_metadata():
    client = build_service_client()
    created = client.post(
        "/sessions",
        json={"user_id": "alice", "character_id": "mei", "session_id": "v1"},
    )
    assert created.status_code == 201

    response = client.post(
        "/sessions/v1/images",
        json={
            "prompt": "What do you see?",
            "image_base64": PNG_B64,
            "mime_type": "image/png",
            "source_type": "screenshot",
            "remember": False,
        },
        headers={"X-Trace-ID": "vision-trace"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "I can see it."
    assert body["vision_provider"] == "fake-vlm"
    assert body["vision_model"] == "vision-test"
    assert body["vision_memory"] == "ephemeral"
    assert body["source_type"] == "screenshot"
    assert body["trace_id"] == "vision-trace"


def test_service_image_endpoint_rejects_bad_base64_and_unconfigured_pipeline():
    client = build_service_client()
    client.post(
        "/sessions",
        json={"user_id": "alice", "character_id": "mei", "session_id": "v1"},
    )
    bad = client.post(
        "/sessions/v1/images",
        json={"image_base64": "not-base64!!", "mime_type": "image/png"},
    )
    assert bad.status_code == 400
    assert bad.json()["error"] == "validation_error"
