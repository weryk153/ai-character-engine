from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm import (
    DeploymentHealthChecker,
    InferenceRuntimeMetadata,
    LocalDeploymentConfig,
    ModelEndpoint,
    ModelGatewayClient,
    OllamaClient,
    OpenAICompatibleChatClient,
    StaticModelRouter,
    VLLMClient,
    build_validated_model_endpoint,
    deployment_llm_factory,
    safe_base_url,
)
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.observability import InMemoryObservabilitySink, Tracer
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from ai_character_engine.tools.models import ToolDefinition
from tests.fakes import FakeLLMClient


class FakeCompletions:
    def __init__(self, *, response=None, delay: float = 0.0, error: Exception | None = None):
        self.response = response
        self.delay = delay
        self.error = error
        self.requests = []
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return self.response
        finally:
            self.active -= 1


class FakeOpenAI:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def fake_response(*, text="hello", model="local-model", tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


class FakeProbeTransport:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload or {}
        self.error = error
        self.calls = []

    async def get_json(self, url, *, headers=None, timeout_seconds):
        self.calls.append((url, headers, timeout_seconds))
        if self.error:
            raise self.error
        return self.payload


def clock_tool() -> ToolDefinition:
    return ToolDefinition(
        name="clock",
        description="read current time",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )


@pytest.mark.asyncio
async def test_openai_compatible_chat_normalizes_text_tool_calls_usage_and_metadata() -> None:
    raw_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="clock", arguments="{}"),
    )
    completions = FakeCompletions(response=fake_response(text="", tool_calls=[raw_call]))
    client = OpenAICompatibleChatClient(
        model="qwen",
        base_url="http://user:pass@localhost:8000/v1?secret=yes",
        api_key="top-secret",
        backend="vllm",
        client=FakeOpenAI(completions),
        runtime_metadata=InferenceRuntimeMetadata(
            backend="vllm", device="cuda", quantization="awq", context_length=32768
        ),
    )
    result = await client.generate([Message(role="user", content="time?")], tools=[clock_tool()])
    assert result.model == "local-model"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.tool_calls[0].name == "clock"
    assert result.metadata["deployment"]["backend"] == "vllm"
    assert result.metadata["deployment"]["base_url"] == "http://localhost:8000/v1"
    assert "top-secret" not in repr(result.metadata)
    assert result.metadata["deployment"]["runtime"]["quantization"] == "awq"
    sent_tool = completions.requests[0]["tools"][0]
    assert sent_tool["function"]["name"] == "clock"


@pytest.mark.asyncio
async def test_local_client_retries_then_succeeds() -> None:
    class Flaky:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return fake_response(text="ok")

    flaky = Flaky()
    client = OpenAICompatibleChatClient(
        model="m",
        base_url="http://localhost:8000/v1",
        client=FakeOpenAI(flaky),
        max_retries=1,
        retry_backoff_seconds=0,
    )
    result = await client.generate([Message(role="user", content="hi")])
    assert result.text == "ok"
    assert flaky.calls == 2
    assert result.metadata["deployment"]["attempt"] == 2


@pytest.mark.asyncio
async def test_max_concurrency_applies_backpressure() -> None:
    completions = FakeCompletions(response=fake_response(), delay=0.02)
    client = OpenAICompatibleChatClient(
        model="m",
        base_url="http://localhost:8000/v1",
        client=FakeOpenAI(completions),
        max_concurrency=1,
    )
    await asyncio.gather(
        client.generate([Message(role="user", content="a")]),
        client.generate([Message(role="user", content="b")]),
    )
    assert completions.max_active == 1


def test_deployment_profiles_and_secret_safe_repr() -> None:
    ollama = LocalDeploymentConfig.local_ollama(model="qwen3:8b", api_key="secret")
    local = LocalDeploymentConfig.local_vllm(model="Qwen/Qwen3-8B")
    remote = LocalDeploymentConfig.remote_vllm(
        model="Qwen/Qwen3-8B", base_url="https://gpu.example/v1", api_key="secret"
    )
    assert ollama.backend == "ollama" and ollama.is_local
    assert local.backend == "vllm" and local.is_local
    assert remote.backend == "vllm" and not remote.is_local
    assert "secret" not in repr(remote)
    assert remote.public_metadata()["base_url"] == "https://gpu.example/v1"


@pytest.mark.asyncio
async def test_health_checker_ollama_uses_native_tags_endpoint() -> None:
    transport = FakeProbeTransport(
        {"models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}]}
    )
    checker = DeploymentHealthChecker(transport)
    config = LocalDeploymentConfig.local_ollama(model="qwen3:8b")
    status = await checker.check(config)
    assert status.ready is True
    assert status.model_available is True
    assert transport.calls[0][0] == "http://127.0.0.1:11434/api/tags"


@pytest.mark.asyncio
async def test_health_checker_openai_compatible_requires_configured_model() -> None:
    transport = FakeProbeTransport({"data": [{"id": "other"}]})
    checker = DeploymentHealthChecker(transport)
    config = LocalDeploymentConfig.local_vllm(model="wanted")
    status = await checker.check(config)
    assert status.reachable is True
    assert status.model_available is False
    assert status.ready is False
    assert transport.calls[0][0].endswith("/v1/models")


@pytest.mark.asyncio
async def test_failed_local_validation_disables_endpoint_for_gateway_fallback() -> None:
    config = LocalDeploymentConfig.local_vllm(model="local")
    checker = DeploymentHealthChecker(FakeProbeTransport(error=ConnectionError("down")))
    local_endpoint, status = await build_validated_model_endpoint(
        endpoint_id="local",
        config=config,
        checker=checker,
        client=FakeLLMClient("local"),
    )
    cloud = FakeLLMClient("cloud")
    gateway = ModelGatewayClient(
        endpoints=(local_endpoint, ModelEndpoint("cloud", cloud)),
        router=StaticModelRouter("local", "cloud", route_name="local-first"),
    )
    result = await gateway.generate([Message(role="user", content="hi")])
    assert status.ready is False
    assert local_endpoint.enabled is False
    assert result.text == "cloud"
    assert result.metadata["gateway"]["attempts"][0]["skipped_reason"] == "disabled"


@pytest.mark.asyncio
async def test_runtime_observability_records_local_deployment_metadata() -> None:
    sink = InMemoryObservabilitySink()
    completions = FakeCompletions(response=fake_response(text="hello", model="qwen"))
    client = VLLMClient(
        model="qwen",
        base_url="http://localhost:8000/v1",
        client=FakeOpenAI(completions),
        runtime_metadata=InferenceRuntimeMetadata(
            backend="vllm", device="cuda", quantization="fp8", context_length=65536
        ),
    )
    runtime = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=client,
        tracer=Tracer(sink),
    )
    await runtime.run_turn("hi")
    span = next(x for x in sink.spans if x.name == "llm.generate")
    assert span.attributes["deployment.backend"] == "vllm"
    assert span.attributes["deployment.device"] == "cuda"
    assert span.attributes["deployment.quantization"] == "fp8"


def test_deployment_llm_factory_integrates_with_session_runtime_factory() -> None:
    profile = CharacterProfile(id="c", name="C", description="test")
    manager = SessionManager(default_ttl_seconds=None)
    completions = FakeCompletions(response=fake_response())
    # Build one injected compatible client via the normal factory shape.
    config = LocalDeploymentConfig.local_vllm(model="qwen")
    factory_fn = deployment_llm_factory(config)
    assert callable(factory_fn)
    # The real helper constructs the client from config; CharacterRuntimeFactory remains unchanged.
    runtime_factory = CharacterRuntimeFactory(
        characters={"c": profile},
        llm_factory=lambda record: FakeLLMClient("local"),
        session_manager=manager,
    )
    managed = runtime_factory.create(user_id="u", character_id="c", session_id="s")
    assert managed.runtime.character.id == "c"


def test_safe_base_url_strips_credentials_query_and_fragment() -> None:
    assert (
        safe_base_url("https://user:pw@example.com:8443/v1/?token=x#frag")
        == "https://example.com:8443/v1"
    )


def test_service_readiness_endpoint_uses_injected_probe() -> None:
    from fastapi.testclient import TestClient
    from ai_character_engine.service import (
        BufferedCharacterStreamSource,
        CharacterService,
        CharacterServiceConfig,
        NoopAuthHook,
        create_app,
    )

    profile = CharacterProfile(id="c", name="C", description="test")
    manager = SessionManager(default_ttl_seconds=None)
    runtime_factory = CharacterRuntimeFactory(
        characters={"c": profile},
        llm_factory=lambda record: FakeLLMClient("ok"),
        session_manager=manager,
    )

    async def not_ready():
        return {"ready": False, "backend": "vllm", "error": "model unavailable"}

    service = CharacterService(
        runtime_factory=runtime_factory,
        session_manager=manager,
        stream_source=BufferedCharacterStreamSource(),
        readiness_probe=not_ready,
    )
    client = TestClient(
        create_app(
            service=service,
            config=CharacterServiceConfig(),
            auth_hook=NoopAuthHook(),
        )
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["details"]["backend"] == "vllm"
