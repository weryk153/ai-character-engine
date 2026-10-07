"""The companion service: a game's NPCs over HTTP and WebSocket.

A Godot or Unity game talks to it with JSON only. Each test plays what a game
does: open an NPC with its persona, talk, pass time, save, load, start again.
"""
from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from ai_character_engine.companion_service import ModelConfig, ServiceConfig, create_app, load_config
from ai_character_engine.companion_service.registry import GameClock
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from tests.test_companion import DAWN, dawns_work

MIRA = {"name": "Mira", "description": "The keeper of the lighthouse."}
MORNING = datetime(2001, 4, 1, 6, 0, tzinfo=UTC).timestamp()


class Speaker:
    """Her model: streams each line in two parts, the last line again after."""

    def __init__(self, *lines: str) -> None:
        self.lines = list(lines) or ["Good morning. The sea is calm."]
        self.calls: list = []

    def _next(self) -> str:
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        return LLMResponse(text=self._next(), model="speaker")

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        text = self._next()
        half = len(text) // 2
        yield LLMStreamChunk(text=text[:half])
        yield LLMStreamChunk(text=text[half:])
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="speaker"))


class Thinker:
    """The background model: the memory worker's answer, empty for the rest."""

    async def generate(self, messages, *, tools=None):
        payload = dawns_work(messages) if "User lines to extract from:" in messages[1].content else {}
        return LLMResponse(text=json.dumps(payload), model="thinker")


class Glad:
    """The mood worker's model: she is happy."""

    async def generate(self, messages, *, tools=None):
        payload = {"mood": "happy", "intensity": 0.8, "confidence": 0.9, "evidence": []}
        return LLMResponse(text=json.dumps(payload), model="glad")


def service(tmp_path, *, speaker=None, token=None, host="127.0.0.1", background=None, **settings):
    speaker = speaker or Speaker()
    config = ServiceConfig(
        data_dir=tmp_path / "data",
        foreground=ModelConfig(base_url="http://speaker", model="speaker"),
        background=background or ModelConfig(base_url="http://thinker", model="thinker"),
        host=host,
        token=token,
        settings={
            "emotion_every": 0,
            "reply_check_every": 0,
            "mood_every": 0,
            "memory_every": 1,
            "memory_conflicts": False,
            "self_memory_every": 0,
            "goal_every": 0,
            "reflection_every": 0,
            "user_state_every": 0,
            "diary_every_hours": 0,
            **settings,
        },
    )
    models = {"speaker": speaker, "thinker": Thinker(), "glad": Glad()}
    return TestClient(create_app(config, make_llm=lambda model: models[model.model])), speaker


def opened(client, slot="s1", npc="mira", **body):
    response = client.put(f"/slots/{slot}/npcs/{npc}", json={"character": MIRA, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_game_opens_an_npc_and_talks_to_her(tmp_path):
    client, speaker = service(tmp_path)
    with client:
        assert opened(client) == {"opened": True}
        assert opened(client) == {"opened": False}
        response = client.post("/slots/s1/npcs/mira/reply", json={"text": "hello", "conversation_id": "pier"})
        state = client.get("/slots/s1/npcs/mira/state").json()
    assert response.status_code == 200
    assert response.json()["text"] == "Good morning. The sea is calm."
    assert response.json()["state"]["relationship_stage"] == state["relationship_stage"]
    assert "You are Mira." in speaker.calls[-1][0].content


def test_what_she_writes_down_is_dated_by_the_games_time(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        client.post("/slots/s1/npcs/mira/reply", json={"text": DAWN, "game_time": MORNING})
        client.post("/slots/s1/npcs/mira/save")  # waits for her background work
        app = client.app
        companion = app.state.npcs.companion("s1", "mira")
        records = companion.runtime.memory_manager.store.list_for_character("mira")
        inspected = client.get("/slots/s1/npcs/mira/inspect").json()
    assert [record.created_at.timestamp() for record in records] == [MORNING]
    assert inspected["memories"] == ["Dawn works at a print shop."]
    assert inspected["clock"] == MORNING


def test_a_slot_saved_loaded_and_started_again(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        client.post("/slots/s1/npcs/mira/reply", json={"text": DAWN})
        saved = client.post("/slots/s1/save").json()
        assert list(saved["npcs"]) == ["mira"]

        assert client.delete("/slots/s1").status_code == 200
        assert client.get("/slots/s1/npcs/mira/state").status_code == 404
        opened(client)
        assert client.get("/slots/s1/npcs/mira/inspect").json()["memories"] == []

        assert client.post("/slots/s1/load", json=saved).status_code == 200
        after_load = client.get("/slots/s1/npcs/mira/inspect").json()["memories"]

        one = client.post("/slots/s1/npcs/mira/save").json()["data"]
        opened(client, slot="s2")
        assert client.post("/slots/s2/npcs/mira/load", json={"data": one}).status_code == 200
        in_slot_two = client.get("/slots/s2/npcs/mira/inspect").json()["memories"]
    assert after_load == ["Dawn works at a print shop."]
    assert in_slot_two == ["Dawn works at a print shop."]


def test_a_slot_load_empties_the_npcs_the_save_does_not_hold(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        saved = client.post("/slots/s1/save").json()
        opened(client, npc="ren")
        client.post("/slots/s1/npcs/ren/reply", json={"text": DAWN})
        client.post("/slots/s1/npcs/ren/save")
        assert client.get("/slots/s1/npcs/ren/inspect").json()["memories"]
        client.post("/slots/s1/load", json=saved)
        ren = client.get("/slots/s1/npcs/ren/inspect").json()["memories"]
    assert ren == []


def test_memories_across_runs_reach_her_in_every_slot(tmp_path):
    client, speaker = service(tmp_path)
    with client:
        opened(client, slot="s1")
        opened(client, slot="s2")
        client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"})
        added = client.post("/npcs/mira/across-runs", json={"text": "Someone warned you of the storm.", "run": 1})
        client.post("/slots/s2/npcs/mira/reply", json={"text": "hi"})
        in_slot_two = speaker.calls[-1][0].content
        client.post("/slots/s1/npcs/mira/reply", json={"text": "hi again"})
        in_slot_one = speaker.calls[-1][0].content
        listed = client.get("/npcs/mira/across-runs").json()
        forgotten = client.delete(f"/npcs/mira/across-runs/{added.json()['id']}")
        again = client.delete(f"/npcs/mira/across-runs/{added.json()['id']}")
    assert "- Someone warned you of the storm." in in_slot_two
    assert "- Someone warned you of the storm." in in_slot_one
    assert [(m["text"], m["run"]) for m in listed] == [("Someone warned you of the storm.", 1)]
    assert forgotten.status_code == 200
    assert again.status_code == 404


def test_the_socket_streams_her_reply_and_takes_an_interruption(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        with client.websocket_connect("/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hello", "conversation_id": "pier"})
            deltas, done = [], None
            while done is None:
                message = socket.receive_json()
                if message["type"] == "delta":
                    deltas.append(message["text"])
                elif message["type"] == "done":
                    done = message
            socket.send_json({"type": "interrupt", "id": "r1", "npc": "mira", "heard": "Good morning."})
            socket.send_json({"type": "reply", "id": "r2", "npc": "nobody", "text": "hi"})
            error = socket.receive_json()
        companion = client.app.state.npcs.companion("s1", "mira")
        history = [message.content for message in companion.runtime.history]
    assert "".join(deltas) == "Good morning. The sea is calm."
    assert done["text"] == "Good morning. The sea is calm." and done["interrupted"] is False
    assert done["id"] == "r1" and done["npc"] == "mira" and "trust" in done["state"]
    assert history == ["hello", "Good morning. [Interrupted by user]"]
    assert error == {
        "type": "error",
        "id": "r2",
        "npc": "nobody",
        "code": "npc_not_open",
        "message": "nobody is not open in slot s1: PUT it first",
    }


def test_errors_are_codes_a_game_can_act_on(tmp_path):
    client, _ = service(tmp_path)
    with client:
        not_open = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"})
        bad_name = client.put("/slots/..%2Fescape/npcs/mira", json={"character": MIRA})
        bad_body = client.put("/slots/s1/npcs/mira", json={"character": {"name": "Mira"}})
        opened(client)
        damaged = client.post("/slots/s1/npcs/mira/load", json={"data": base64.b64encode(b"nonsense").decode()})
        not_base64 = client.post("/slots/s1/npcs/mira/load", json={"data": "%%%"})
        opened(client, npc="ren")
        rens = client.post("/slots/s1/npcs/ren/save").json()["data"]
        other = client.post("/slots/s1/npcs/mira/load", json={"data": rens})
    assert (not_open.status_code, not_open.json()["code"]) == (404, "npc_not_open")
    assert bad_name.status_code in (400, 404)
    assert (bad_body.status_code, bad_body.json()["code"]) == (400, "invalid_request")
    assert (damaged.status_code, damaged.json()["code"]) == (422, "bad_save")
    assert (not_base64.status_code, not_base64.json()["code"]) == (422, "bad_save")
    assert (other.status_code, other.json()["code"]) == (422, "other_character")


def test_a_name_that_could_leave_the_data_dir_is_refused(tmp_path):
    client, _ = service(tmp_path)
    with client:
        response = client.put("/slots/..dots/npcs/mira", json={"character": MIRA})
    assert (response.status_code, response.json()["code"]) == (400, "invalid_request")
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").rglob("*dots*"))


def test_a_token_is_asked_for_when_one_is_set(tmp_path):
    client, _ = service(tmp_path, token="s3cret")
    with client:
        without = client.put("/slots/s1/npcs/mira", json={"character": MIRA})
        with_it = client.put(
            "/slots/s1/npcs/mira", json={"character": MIRA}, headers={"Authorization": "Bearer s3cret"}
        )
        health = client.get("/health")
    assert (without.status_code, without.json()["code"]) == (401, "unauthorized")
    assert with_it.status_code == 200
    assert health.json()["status"] == "ok"
    with pytest.raises(ValueError):
        service(tmp_path, host="0.0.0.0")


def test_an_idle_npc_is_closed_and_comes_back_as_she_was(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        client.post("/slots/s1/npcs/mira/reply", json={"text": DAWN})
        client.post("/slots/s1/npcs/mira/save")
        npcs = client.app.state.npcs
        closed = client.portal.call(npcs.close_idle, float("inf"))
        remembered = client.get("/slots/s1/npcs/mira/inspect").json()["memories"]
    assert closed == [("s1", "mira")]
    assert remembered == ["Dawn works at a print shop."]


def test_the_clock_stands_still_between_requests():
    clock = GameClock()
    clock.set(MORNING)
    clock.set(None)
    assert clock() == MORNING


def test_a_config_file_names_the_models_and_where_npcs_live(tmp_path):
    path = tmp_path / "game.toml"
    path.write_text(
        'data_dir = "npcs"\n'
        "[models.foreground]\n"
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "qwen/qwen3.5-9b"\n'
        "temperature = 0.8\n"
        "[models.background.memory]\n"
        'base_url = "http://127.0.0.1:1235/v1"\n'
        'model = "qwen/qwen3.5-9b"\n'
        "[settings]\n"
        'language = "English"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.data_dir == tmp_path / "npcs"
    assert config.foreground.request_options() == {"temperature": 0.8}
    assert set(config.background) == {"memory"}
    assert config.companion_settings().language == "English"
    path.write_text(path.read_text() + "\n[settings.extra]\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)


def test_the_socket_tells_the_game_when_her_mood_changed_after_a_reply(tmp_path):
    client, _ = service(
        tmp_path,
        background={"mood": ModelConfig(base_url="http://glad", model="glad")},
        memory_every=0,
        mood_every=1,
    )
    with client:
        opened(client)
        with client.websocket_connect("/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "I brought you tea"})
            told = None
            while told is None:
                message = socket.receive_json()
                if message["type"] == "state":
                    told = message
    assert told["npc"] == "mira"
    assert told["state"]["emotion"] == "happy"


class Down(Speaker):
    async def stream_generate(self, messages, *, tools=None):
        from ai_character_engine.llm.errors import LLMError

        raise LLMError("connection refused")
        yield  # pragma: no cover

    async def generate(self, messages, *, tools=None):
        from ai_character_engine.llm.errors import LLMError

        raise LLMError("connection refused")


def test_a_model_that_cannot_be_reached_is_named_as_such(tmp_path):
    client, _ = service(tmp_path, speaker=Down())
    with client:
        opened(client)
        response = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"})
        with client.websocket_connect("/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hi again"})
            streamed = socket.receive_json()
    assert (response.status_code, response.json()["code"]) == (502, "model_unavailable")
    assert (streamed["type"], streamed["code"]) == ("error", "model_unavailable")
