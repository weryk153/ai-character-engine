"""Offline v0.19 voice-pipeline example; no microphone, model API, or TTS service required."""

from __future__ import annotations

import asyncio

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context import ContextBuilder
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from ai_character_engine.session.store import InMemorySessionStore
from ai_character_engine.voice import AudioFormat, SynthesizedAudio, TranscriptResult, VoicePipeline


class FakeLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="原來如此。今天先別把自己逼太緊。", model="offline-fake")


class FakeSTT:
    async def transcribe(self, audio: bytes, *, audio_format: AudioFormat):
        return TranscriptResult(text="我今天有點累", language="zh-TW", latency_ms=18.0)


class FakeTTS:
    async def synthesize(self, text: str, *, voice: str | None = None):
        return SynthesizedAudio(
            data=f"AUDIO<{text}>".encode(),
            format=AudioFormat(sample_rate_hz=24_000),
            latency_ms=12.0,
            metadata={"voice": voice or "default"},
        )


async def main() -> None:
    character = CharacterProfile(
        id="mei",
        name="Mei",
        description="理性、直接的研究者。",
        personality=["理性", "偶爾吐槽"],
        speaking_style=["自然口語", "避免客服腔"],
    )
    manager = SessionManager(InMemorySessionStore())
    factory = CharacterRuntimeFactory(
        characters={character.id: character},
        llm_factory=lambda _record: FakeLLM(),
        context_builder_factory=ContextBuilder,
        session_manager=manager,
    )
    session = factory.create(user_id="demo-user", character_id="mei")
    pipeline = VoicePipeline(stt=FakeSTT(), tts=FakeTTS())

    print("Streaming voice events:")
    async for event in pipeline.stream_turn(session, audio=b"fake-pcm-audio"):
        if event.type == "tts_audio":
            print(f"  {event.type}: {event.text!r}, bytes={len(event.audio.data)}")
        else:
            print(f"  {event.type}: {event.text or event.data}")


if __name__ == "__main__":
    asyncio.run(main())
