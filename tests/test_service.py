from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.service import (
    BufferedCharacterStreamSource,
    CharacterService,
    CharacterServiceConfig,
    NoopAuthHook,
    create_app,
)
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from tests.fakes import FakeLLMClient


class SlowLLMClient:
    async def generate(self, messages, *, tools=None):
        await asyncio.sleep(0.05)
        return LLMResponse(text="slow")


def build_client(*, text: str = "hello service", slow: bool = False):
    character = CharacterProfile(
        id="mei",
        name="Mei",
        description="test character",
    )
    manager = SessionManager(default_ttl_seconds=None)
    factory = CharacterRuntimeFactory(
        characters={character.id: character},
        llm_factory=(lambda record: SlowLLMClient()) if slow else (lambda record: FakeLLMClient(text)),
        session_manager=manager,
    )
    config = CharacterServiceConfig(
        default_timeout_seconds=1,
        stream_chunk_chars=5,
        cors_origins=("https://example.com",),
    )
    service = CharacterService(
        runtime_factory=factory,
        session_manager=manager,
        stream_source=BufferedCharacterStreamSource(chunk_chars=config.stream_chunk_chars),
        default_timeout_seconds=config.default_timeout_seconds,
    )
    app = create_app(service=service, config=config, auth_hook=NoopAuthHook())
    return TestClient(app), manager


def create_session(client: TestClient, session_id: str = "s1") -> dict:
    response = client.post(
        "/sessions",
        json={
            "user_id": "alice",
            "character_id": "mei",
            "session_id": session_id,
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health_and_request_trace_headers() -> None:
    client, _ = build_client()
    response = client.get(
        "/health",
        headers={"X-Request-ID": "req-1", "X-Trace-ID": "trace-1"},
    )
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"
    assert response.headers["X-Request-ID"] == "req-1"
    assert response.headers["X-Trace-ID"] == "trace-1"


def test_session_lifecycle_and_single_turn_message() -> None:
    client, _ = build_client()
    created = create_session(client)
    assert created["status"] == "active"

    fetched = client.get("/sessions/s1")
    assert fetched.status_code == 200
    assert fetched.json()["user_id"] == "alice"

    message = client.post(
        "/sessions/s1/messages",
        json={"content": "hello"},
        headers={"X-Request-ID": "r1", "X-Trace-ID": "t1"},
    )
    assert message.status_code == 200
    body = message.json()
    assert body["text"] == "hello service"
    assert body["request_id"] == "r1"
    assert body["trace_id"] == "t1"
    assert body["rounds"] == 1

    closed = client.post("/sessions/s1/close")
    assert closed.status_code == 200
    assert closed.json()["status"] == "closed"

    unavailable = client.post("/sessions/s1/messages", json={"content": "again"})
    assert unavailable.status_code == 410
    assert unavailable.json()["error"] == "session_unavailable"


def test_missing_session_maps_to_404() -> None:
    client, _ = build_client()
    response = client.post("/sessions/missing/messages", json={"content": "hi"})
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_validation_error_has_service_shape() -> None:
    client, _ = build_client()
    response = client.post(
        "/sessions",
        json={"user_id": "", "character_id": "mei"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["request_id"]
    assert body["trace_id"]


def test_turn_timeout_maps_to_504() -> None:
    client, _ = build_client(slow=True)
    create_session(client)
    response = client.post(
        "/sessions/s1/messages",
        json={"content": "hi", "timeout_seconds": 0.001},
    )
    assert response.status_code == 504
    assert response.json()["error"] == "timeout"


def test_sse_stream_emits_delta_trace_final() -> None:
    client, _ = build_client(text="abcdefghijk")
    create_session(client)
    with client.stream(
        "POST",
        "/sessions/s1/messages/stream",
        json={"content": "stream this"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "\n".join(response.iter_lines())
    assert "event: delta" in payload
    assert '"text":"abcde"' in payload
    assert "event: trace" in payload
    assert "event: final" in payload


def test_sse_missing_session_returns_http_404_before_stream() -> None:
    client, _ = build_client()
    response = client.post(
        "/sessions/missing/messages/stream",
        json={"content": "hello"},
    )
    assert response.status_code == 404


def test_websocket_persistent_bidirectional_session() -> None:
    client, manager = build_client(text="ws reply")
    create_session(client)
    with client.websocket_connect("/ws/sessions/s1") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"

        ws.send_json({"type": "message", "content": "first"})
        types = []
        while True:
            event = ws.receive_json()
            types.append(event["type"])
            if event["type"] == "final":
                break
        assert "delta" in types
        assert "trace" in types
        assert types[-1] == "final"

        ws.send_json({"type": "message", "content": "second"})
        while True:
            event = ws.receive_json()
            if event["type"] == "final":
                break

    record = manager.require("s1")
    assert record.runtime_snapshot is not None
    contents = [message.content for message in record.runtime_snapshot.history]
    assert "first" in contents
    assert "second" in contents


def test_cors_preflight() -> None:
    client, _ = build_client()
    response = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://example.com"
