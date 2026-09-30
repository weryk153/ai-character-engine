"""Offline v0.27 expressive-avatar demo; no network, microphone or VRM renderer."""

from __future__ import annotations

import asyncio
from array import array

from ai_character_engine import (
    AvatarRuntime,
    CharacterProfile,
    CharacterRuntime,
    DuplexVoiceConfig,
    LiveCharacterOrchestrator,
    LiveEventType,
    LiveRuntimeConfig,
)
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.voice import AudioChunk, AudioFormat
from ai_character_engine.llm import LLMResponse, LLMStreamChunk


def pcm(value: int, samples: int = 320) -> bytes:
    return array("h", [value] * samples).tobytes()


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="こんにちは。")

    async def stream_generate(self, messages, *, tools=None):
        yield LLMStreamChunk(text="こんにちは。")
        yield LLMStreamChunk(final=True, response=LLMResponse(text="こんにちは。"))


class DemoTTS:
    async def synthesize(self, text, *, voice=None):
        raise AssertionError("demo expects streaming TTS")

    async def synthesize_stream(self, text, *, voice=None):
        yield AudioChunk(
            pcm(8000),
            AudioFormat(),
            metadata={
                "visemes": [
                    {"viseme": "oh", "start_ms": 0, "duration_ms": 8},
                    {"viseme": "ih", "start_ms": 8, "duration_ms": 12},
                ]
            },
        )


class NoopSink:
    async def play(self, chunk: AudioChunk) -> None:
        await asyncio.sleep(0)


async def main() -> None:
    runtime = CharacterRuntime(
        character=CharacterProfile("mei-demo", "Mei Demo", "offline avatar demo"),
        llm=DemoLLM(),
    )
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime),
        tts=DemoTTS(),
        audio_sink=NoopSink(),
        avatar_runtime=AvatarRuntime(),
        config=LiveRuntimeConfig(streaming_output=True),
        duplex=DuplexVoiceConfig(),
    )
    live.submit_text("挨拶して")
    for event in await live.run_once():
        if event.type in (LiveEventType.AVATAR_CUE, LiveEventType.AVATAR_RESET):
            print(event.type.value, event.data)


if __name__ == "__main__":
    asyncio.run(main())
