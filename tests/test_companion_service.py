"""The companion service: a game's NPCs over HTTP and WebSocket.

A Godot or Unity game talks to it with JSON only. Each test plays what a game
does: open an NPC with its persona, talk, pass time, save, load, start again.
"""
from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from ai_character_engine.companion_service import ModelConfig, ServiceConfig, create_app, load_config
from ai_character_engine.companion_service.registry import GameClock, NpcRegistry
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


class Slow(Speaker):
    """Her model, held mid-line until ``gate`` is set."""

    def __init__(self, *lines: str) -> None:
        super().__init__(*lines)
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        yield LLMStreamChunk(text="Good morning. ")  # she forwards whole sentences
        self.started.set()
        await self.gate.wait()
        yield LLMStreamChunk(text="The sea is calm.")
        yield LLMStreamChunk(
            final=True, response=LLMResponse(text="Good morning. The sea is calm.", model="slow")
        )


def service(
    tmp_path, *, speaker=None, token=None, host="127.0.0.1", background=None, raise_errors=True, **settings
):
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
    app = create_app(config, make_llm=lambda model: models[model.model])
    return TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=raise_errors), speaker


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
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
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
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
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
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hi again"})
            streamed = socket.receive_json()
    assert (response.status_code, response.json()["code"]) == (502, "model_unavailable")
    assert (streamed["type"], streamed["code"]) == ("error", "model_unavailable")


def test_a_model_is_told_not_to_think_and_her_background_work_is_kept_short(tmp_path):
    """A reasoning model (qwen3.5) thinks first unless told not to: with her
    whole prompt that took over a minute, and the call timed out. Her
    background work asks for JSON: a low temperature, a cap, the same switch."""
    path = tmp_path / "game.toml"
    path.write_text(
        "[models.foreground]\n"
        'base_url = "http://127.0.0.1:1234/v1"\n'
        'model = "qwen/qwen3.5-9b"\n'
        "temperature = 0.8\n"
        "[models.foreground.extra_body]\n"
        'reasoning_effort = "none"\n'
        "top_k = 20\n",
        encoding="utf-8",
    )
    config = load_config(path)
    built = []
    NpcRegistry(config, make_llm=lambda model: built.append(model) or Speaker())
    foreground, background = built
    assert foreground.request_options() == {
        "temperature": 0.8,
        "extra_body": {"reasoning_effort": "none", "top_k": 20},
    }
    assert background.request_options() == {
        "temperature": 0.1,
        "max_tokens": 600,
        "extra_body": {"reasoning_effort": "none"},
    }



# --- what a game meets when things go wrong (review of the service) -----------------


def test_a_load_or_new_game_during_her_line_ends_it_with_a_code(tmp_path):
    client, speaker = service(tmp_path, speaker=Slow())
    with client:
        opened(client)
        saved = client.post("/slots/s1/npcs/mira/save").json()["data"]
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hello"})
            assert socket.receive_json()["type"] == "delta"
            client.post("/slots/s1/npcs/mira/load", json={"data": saved})
            ended = socket.receive_json()
        opened(client)
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r2", "npc": "mira", "text": "hello"})
            assert socket.receive_json()["type"] == "delta"
            client.delete("/slots/s1")
            gone = socket.receive_json()
    assert (ended["type"], ended["id"], ended["code"]) == ("error", "r1", "npc_reloaded")
    assert (gone["type"], gone["id"], gone["code"]) == ("error", "r2", "npc_reloaded")


def test_a_slot_load_that_cannot_be_done_changes_nothing(tmp_path):
    client, _ = service(tmp_path)
    with client:
        for npc in ("ann", "mira", "zed"):
            opened(client, npc=npc)
        client.post("/slots/s1/npcs/ann/reply", json={"text": DAWN})
        client.post("/slots/s1/npcs/ann/save")
        miras = client.post("/slots/s1/npcs/mira/save").json()["data"]
        refused = client.post("/slots/s1/load", json={"npcs": {"mira": miras, "zed": miras}})
        damaged = client.post("/slots/s1/npcs/ann/load", json={"data": base64.b64encode(b"PK nonsense").decode()})
        ann = client.get("/slots/s1/npcs/ann/inspect").json()["memories"]
    assert (refused.status_code, refused.json()["code"]) == (422, "other_character")
    assert (damaged.status_code, damaged.json()["code"]) == (422, "bad_save")
    assert ann == ["Dawn works at a print shop."]


def test_closing_idle_npcs_survives_an_npc_opened_meanwhile(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"})
        client.post("/slots/s1/npcs/mira/save")  # waits for her background work
        npcs = client.app.state.npcs
        companion = npcs.companion("s1", "mira")
        close = companion.close

        async def close_while_a_game_opens_another():
            npcs.open("s1", "ren", npcs.entry("s1", "mira").character)
            await close()

        companion.close = close_while_a_game_opens_another
        closed = client.portal.call(npcs.close_idle, float("inf"))
    assert closed == [("s1", "mira")]


def test_two_loads_at_once_leave_one_companion_and_none_running_unseen(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        npcs = client.app.state.npcs
        built = []
        build = npcs._build
        npcs._build = lambda slot, npc, entry: built.append(build(slot, npc, entry)) or built[-1]
        first = npcs.companion("s1", "mira")
        close = first.close

        async def slowly():  # her workers take a moment to stop
            await asyncio.sleep(0.05)
            await close()

        first.close = slowly

        async def twice():
            return await asyncio.gather(npcs.reopen("s1", "mira"), npcs.reopen("s1", "mira"))

        client.portal.call(twice)
        current = npcs.entry("s1", "mira").companion
        left_open = [companion for companion in built if not companion._closed and companion is not current]
    assert left_open == []


def test_a_page_in_the_players_browser_cannot_reach_the_npcs(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    client, _ = service(tmp_path)
    with client:
        opened(client)
        rebound = client.get("/slots/s1/npcs/mira/state", headers={"Host": "evil.example"})
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://127.0.0.1/slots/s1/ws", headers={"Origin": "https://evil.example"}):
                pass
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:  # Godot sends no Origin
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hi"})
            assert socket.receive_json()["type"] in ("delta", "done")
    assert rebound.status_code == 400


def test_a_numeric_token_and_the_websocket_token(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    path = tmp_path / "game.toml"
    path.write_text(
        "token = 1234\n[models.foreground]\nbase_url = \"http://x\"\nmodel = \"m\"\n", encoding="utf-8"
    )
    assert load_config(path).token == "1234"
    client, _ = service(tmp_path, token="s3cret")
    with client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://127.0.0.1/slots/s1/ws?token=wrong"):
                pass
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws?token=s3cret") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hi"})
            assert socket.receive_json()["code"] == "npc_not_open"


def test_names_differing_only_in_case_are_refused(tmp_path):
    client, _ = service(tmp_path)
    with client:
        upper = client.put("/slots/Slot1/npcs/mira", json={"character": MIRA})
        npc = client.put("/slots/s1/npcs/Mira", json={"character": MIRA})
    assert (upper.status_code, upper.json()["code"]) == (400, "invalid_request")
    assert (npc.status_code, npc.json()["code"]) == (400, "invalid_request")


def test_every_error_is_a_code_and_a_message(tmp_path):
    client, _ = service(tmp_path, raise_errors=False)
    with client:
        opened(client)
        far = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi", "game_time": 1e300})
        after = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"})
        blank = client.post("/npcs/mira/across-runs", json={"text": "   "})
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira"})
            missing = socket.receive_json()
        npcs = client.app.state.npcs

        def broken(npc):
            raise RuntimeError("disk on fire")

        npcs.across_runs = broken
        crashed = client.get("/npcs/mira/across-runs")
    assert (far.status_code, far.json()["code"]) == (400, "invalid_request")
    assert after.status_code == 200
    assert (blank.status_code, blank.json()["code"]) == (400, "invalid_request")
    assert (missing["type"], missing["id"], missing["code"]) == ("error", "r1", "invalid_request")
    assert crashed.status_code == 500 and crashed.json() == {"code": "internal_error", "message": "disk on fire"}



# --- the smaller things a game or its player may meet ------------------------------


class HeldThinker(Thinker):
    """Her background model, held until ``gate`` is set."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        await self.gate.wait()
        return await super().generate(messages, tools=tools)


def test_an_npc_with_background_work_under_way_is_not_closed_as_idle(tmp_path):
    thinker = HeldThinker()
    config = ServiceConfig(
        data_dir=tmp_path / "data",
        foreground=ModelConfig(base_url="http://speaker", model="speaker"),
        background=ModelConfig(base_url="http://thinker", model="thinker"),
        settings={"emotion_every": 0, "reply_check_every": 0, "mood_every": 0, "memory_every": 1,
                  "memory_conflicts": False, "self_memory_every": 0, "goal_every": 0,
                  "reflection_every": 0, "user_state_every": 0, "diary_every_hours": 0},
    )
    models = {"speaker": Speaker(), "thinker": thinker}
    client = TestClient(create_app(config, make_llm=lambda model: models[model.model]), base_url="http://127.0.0.1")
    with client:
        opened(client)
        client.post("/slots/s1/npcs/mira/reply", json={"text": DAWN})
        npcs = client.app.state.npcs
        while_working = client.portal.call(npcs.close_idle, float("inf"))
        client.portal.call(thinker.gate.set)
        client.portal.call(npcs.companion("s1", "mira").settle)
        after = client.portal.call(npcs.close_idle, float("inf"))
        remembered = client.get("/slots/s1/npcs/mira/inspect").json()["memories"]
    assert while_working == []
    assert after == [("s1", "mira")]
    assert remembered == ["Dawn works at a print shop."]


def test_a_dropped_socket_does_not_cost_her_the_line_she_was_saying(tmp_path):
    speaker = Slow()
    client, _ = service(tmp_path, speaker=speaker)
    with client:
        opened(client)
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hello", "conversation_id": "pier"})
            assert socket.receive_json()["type"] == "delta"
        client.portal.call(speaker.gate.set)
        companion = client.app.state.npcs.companion("s1", "mira")

        async def until_she_is_done():
            while companion.busy:
                await asyncio.sleep(0.01)

        client.portal.call(until_she_is_done)
        history = [message.content for message in companion.runtime.history]
    assert history == ["hello", "Good morning. The sea is calm."]


def test_a_socket_message_needs_an_id_and_text_frames(tmp_path):
    client, _ = service(tmp_path)
    with client:
        opened(client)
        with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
            socket.send_json({"type": "reply", "npc": "mira", "text": "hi"})
            no_id = socket.receive_json()
            socket.send_bytes(b"\x00\x01")
            binary = socket.receive_json()
            socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": "hi"})
            still_there = socket.receive_json()
    assert (no_id["type"], no_id["code"]) == ("error", "invalid_request")
    assert (binary["type"], binary["code"]) == ("error", "invalid_request")
    assert still_there["id"] == "r1"


def test_a_socket_remembers_only_its_recent_requests():
    from ai_character_engine.companion_service.app import RecentTurns

    turns = RecentTurns(kept=2)
    for number in range(3):
        turns.add(f"r{number}", "mira", "pier")
    assert turns.get("r0") is None
    assert turns.get("r2") == ("mira", "pier")


def test_reading_memories_across_runs_writes_nothing(tmp_path):
    client, _ = service(tmp_path)
    with client:
        listed = client.get("/npcs/ghost/across-runs")
    assert listed.json() == []
    assert not (tmp_path / "data" / "meta" / "ghost").exists()


def test_an_unknown_path_or_method_is_a_code_and_a_message(tmp_path):
    client, _ = service(tmp_path)
    with client:
        missing = client.get("/nowhere")
        wrong = client.patch("/health")
    assert (missing.status_code, missing.json()["code"]) == (404, "not_found")
    assert (wrong.status_code, wrong.json()["code"]) == (405, "method_not_allowed")
    assert set(missing.json()) == {"code", "message"}


def test_the_model_clients_are_closed_with_the_service(tmp_path):
    class Closable(Speaker):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

            class Inner:
                async def close(inner) -> None:
                    self.closed = True

            self.client = Inner()

    speaker, thinker = Closable(), Closable()
    config = ServiceConfig(
        data_dir=tmp_path / "data",
        foreground=ModelConfig(base_url="http://speaker", model="speaker"),
        background=ModelConfig(base_url="http://thinker", model="thinker"),
    )
    models = {"speaker": speaker, "thinker": thinker}
    with TestClient(create_app(config, make_llm=lambda model: models[model.model]), base_url="http://127.0.0.1"):
        pass
    assert (speaker.closed, thinker.closed) == (True, True)


def test_the_token_does_not_reach_the_server_log():
    import logging

    from ai_character_engine.companion_service.app import HideTokens

    record = logging.LogRecord(
        "uvicorn.error", logging.INFO, __file__, 1, '%s - "WebSocket %s" [accepted]',
        ("127.0.0.1:5000", "/slots/s1/ws?token=s3cret&x=1"), None,
    )
    assert HideTokens().filter(record)
    assert "s3cret" not in record.getMessage()
    assert "token=***&x=1" in record.getMessage()
