"""The companion service with scripted models, for trying the example without
a language model and for smoke_test.gd.

    uv run --extra service python examples/godot/tools/fake_service.py [--port 8765]

Her replies are a few fixed lines in turn (one that said the player's words
back would keep them from being remembered: they would be hers); the memory
worker remembers the player's first line; every other worker finds nothing. Data goes to a temporary
directory that is removed when the service stops.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import uvicorn

from ai_character_engine.companion_service import ModelConfig, ServiceConfig, create_app
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk

USER_LINES = "User lines to extract from:"


LINES = (
    "The lamp is lit, as it is every evening.",
    "The sea has been restless since dawn.",
    "Did you come over on the morning boat?",
    "I keep the logbook by the window, if you want to look.",
    "Gulls only gather like that before bad weather.",
)


class ScriptedSpeaker:
    def __init__(self) -> None:
        self.said = 0

    def _line(self) -> str:
        self.said += 1
        return LINES[(self.said - 1) % len(LINES)]

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=self._line(), model="scripted")

    async def stream_generate(self, messages, *, tools=None):
        text = self._line()
        for word in text.split(" "):
            yield LLMStreamChunk(text=word + " ")
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="scripted"))


class NotingWorker:
    async def generate(self, messages, *, tools=None):
        prompt = messages[1].content
        payload: dict = {}
        if "Line: " in prompt:  # a face and a gesture for one of her lines
            line = prompt.split("Line: ", 1)[1].lower()
            payload = {"expression": "thoughtful" if "?" in line else "smile",
                       "motion": "point" if "logbook" in line else None, "intensity": 0.7}
        if USER_LINES in prompt:
            lines = [line.strip() for line in prompt.split(USER_LINES, 1)[1].splitlines() if line.strip()]
            said = lines[0].removeprefix("- ") if lines else ""
            if said:
                payload = {
                    "items": [
                        {"summary": f"The player said: {said}", "kind": "fact", "importance": 0.6,
                         "confidence": 0.9, "evidence": said}
                    ],
                    "confidence": 0.9,
                    "evidence": [],
                }
        return LLMResponse(text=json.dumps(payload), model="noting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="companion-fake-") as data_dir:
        config = ServiceConfig(
            data_dir=Path(data_dir),
            foreground=ModelConfig(base_url="fake://", model="scripted"),
            background=ModelConfig(base_url="fake://", model="noting"),
            port=args.port,
            settings={
                "emotion_every": 0, "reply_check_every": 0, "mood_every": 0, "memory_every": 1,
                "memory_conflicts": False, "self_memory_every": 0, "goal_every": 0,
                "reflection_every": 0, "user_state_every": 0, "diary_every_hours": 0,
            },
        )
        models = {"scripted": ScriptedSpeaker(), "noting": NotingWorker()}
        app = create_app(config, make_llm=lambda model: models[model.model])
        uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
