"""Her lines' faces and gestures through the companion service."""

from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from ai_character_engine.companion import LineActions
from ai_character_engine.companion_service import ModelConfig, ServiceConfig, create_app
from ai_character_engine.companion_service.line_picks import LinePicks, cut_lines
from ai_character_engine.llm.models import LLMResponse
from tests.test_companion_service import MIRA, Speaker

AVATAR = {
    "expressions": ["smile", "sad"],
    "motions": {"wave": "waving"},
    "mood_faces": {"neutral": "smile"},
}


class Director:
    """The background model: a face for lines it is asked about, nothing for other work."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    async def generate(self, messages, *, tools=None):
        asked = messages[-1].content
        if "Line: " not in asked:
            return LLMResponse(text="{}", model="director")
        line = asked.split("Line: ", 1)[1]
        self.lines.append(line)
        if "nothing" in line:
            return LLMResponse(text='{"expression":null,"motion":null}', model="director")
        face = "sad" if "tired" in line else "smile"
        motion = "wave" if "morning" in line.lower() else None
        return LLMResponse(text=json.dumps({"expression": face, "motion": motion, "intensity": 0.5}))


def service(tmp_path, speaker):
    director = Director()
    config = ServiceConfig(
        data_dir=tmp_path / "data",
        foreground=ModelConfig(base_url="http://speaker", model="speaker"),
        background=ModelConfig(base_url="http://director", model="director"),
        settings={
            "emotion_every": 0, "reply_check_every": 0, "mood_every": 0, "memory_every": 0,
            "memory_conflicts": False, "self_memory_every": 0, "goal_every": 0, "reflection_every": 0,
            "user_state_every": 0, "diary_every_hours": 0,
        },
    )
    models = {"speaker": speaker, "director": director}
    app = create_app(config, make_llm=lambda model: models[model.model])
    return TestClient(app, base_url="http://127.0.0.1"), director


def opened(client, **body):
    response = client.put("/slots/s1/npcs/mira", json={"character": MIRA, **body})
    assert response.status_code == 200, response.text


def socket_reply(client, picks, text="hello"):
    """Every message of one reply, up to done and then ``picks`` picks."""
    with client.websocket_connect("ws://127.0.0.1/slots/s1/ws") as socket:
        socket.send_json({"type": "reply", "id": "r1", "npc": "mira", "text": text})
        messages = []
        while not (
            any(m["type"] == "done" for m in messages)
            and sum(m["type"] == "actions" for m in messages) >= picks
        ):
            messages.append(socket.receive_json())
    return messages


# --- cutting lines ----------------------------------------------------------------------------


def test_lines_end_at_sentence_marks_and_new_lines():
    assert cut_lines("早安。今天好累！還好嗎") == (["早安。", "今天好累！"], "還好嗎")
    assert cut_lines("Really?! Yes…\nok") == (["Really?!", " Yes…\n"], "ok")
    assert cut_lines("It is 3.5 meters. Then") == (["It is 3.5 meters."], " Then")
    assert cut_lines("The end.") == ([], "The end.")
    assert cut_lines("「好！」她說。然後") == (["「好！」", "她說。"], "然後")
    assert cut_lines("真的？") == ([], "真的？")    # ？！ or 」 may still come


def test_lines_are_picked_in_order_and_a_line_without_words_is_skipped():
    class Actions:
        def __init__(self):
            self.asked = []

        async def pick(self, line):
            self.asked.append(line)
            return LineActions("smile", None, 1.0)

        def voice(self):
            return "smile"

    async def go():
        actions, handed = Actions(), []
        picks = LinePicks(actions, handed.append)
        picks.feed("Good morn")
        picks.feed("ing. 「Yes!」 …… ")
        picks.finish()
        await picks.task
        return actions.asked, handed

    asked, handed = asyncio.run(go())
    assert asked == ["Good morning.", "「Yes!」"]   # "……" alone has nothing to show
    assert [m["index"] for m in handed] == [0, 1]
    assert handed[0] == {"index": 0, "line": "Good morning.", "expression": "smile", "motion": None,
                         "intensity": 1.0, "voice": "smile"}


# --- through the service ------------------------------------------------------------------------


def test_each_line_of_her_streamed_reply_gets_its_face(tmp_path):
    client, director = service(tmp_path, Speaker("Good morning. I am tired today."))
    with client:
        opened(client, avatar=AVATAR)
        messages = socket_reply(client, picks=2)
    picks = [m for m in messages if m["type"] == "actions"]
    assert [(m["index"], m["line"], m["expression"], m["motion"]) for m in picks] == [
        (0, "Good morning.", "smile", "wave"),
        (1, "I am tired today.", "sad", None),
    ]
    assert all(m["id"] == "r1" and m["npc"] == "mira" for m in picks)
    assert picks[0]["voice"] == "smile" and picks[1]["voice"] == "smile"
    assert director.lines == ["Good morning.", "I am tired today."]


def test_the_last_line_without_a_mark_and_a_line_with_nothing_to_show(tmp_path):
    client, _ = service(tmp_path, Speaker("Good morning. nothing to see here. I am tired"))
    with client:
        opened(client, avatar=AVATAR)
        messages = socket_reply(client, picks=2)
    picks = [m for m in messages if m["type"] == "actions"]
    assert [(m["index"], m["line"]) for m in picks] == [(0, "Good morning."), (2, "I am tired")]


def test_an_npc_opened_without_an_avatar_gets_no_picks(tmp_path):
    client, director = service(tmp_path, Speaker("Good morning."))
    with client:
        opened(client)
        messages = socket_reply(client, picks=0)
        reply = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"}).json()
    assert [m["type"] for m in messages if m["type"] == "actions"] == []
    assert "actions" not in reply and director.lines == []


def test_a_reply_over_http_brings_its_picks(tmp_path):
    client, _ = service(tmp_path, Speaker("Good morning. I am tired today."))
    with client:
        opened(client, avatar=AVATAR)
        reply = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"}).json()
    assert reply["text"] == "Good morning. I am tired today."
    assert [(m["index"], m["expression"]) for m in reply["actions"]] == [(0, "smile"), (1, "sad")]


def test_a_new_avatar_comes_with_a_new_persona_and_none_takes_it_away(tmp_path):
    client, director = service(tmp_path, Speaker("Good morning."))
    with client:
        opened(client, avatar=AVATAR)
        opened(client)
        reply = client.post("/slots/s1/npcs/mira/reply", json={"text": "hi"}).json()
    assert "actions" not in reply and director.lines == []
