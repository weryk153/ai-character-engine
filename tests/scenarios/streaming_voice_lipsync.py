"""Offline demo: streamed LLM text -> streamed TTS -> mouth cues.

Run:
    PYTHONPATH=src python tests/scenarios/streaming_voice_lipsync.py
"""
from __future__ import annotations

import asyncio

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import LiveCharacterOrchestrator, LiveRuntimeConfig
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.voice import AudioChunk, AudioFormat


class DemoStreamingLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="你好。這是 fallback。")

    async def stream_generate(self, messages, *, tools=None):
        parts = ("你好。", "我現在是邊生成邊說。", "嘴型也會收到時間提示。")
        text = ""
        for part in parts:
            await asyncio.sleep(0.01)
            text += part
            yield LLMStreamChunk(text=part)
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="demo-stream"))


class DemoStreamingTTS:
    async def synthesize(self, text, *, voice=None):
        raise AssertionError("v0.26 demo expects synthesize_stream")

    async def synthesize_stream(self, text, *, voice=None):
        # Two tiny PCM chunks per sentence. Real adapters would yield provider audio.
        for _ in range(2):
            await asyncio.sleep(0.005)
            yield AudioChunk(b"\x00\x00" * 320, AudioFormat(sample_rate_hz=16_000))


class DemoSpeaker:
    async def play(self, chunk):
        await asyncio.sleep(len(chunk.data) / (16_000 * 2))

    async def stop(self):
        return None


async def main() -> None:
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "v0.26 streaming character"),
        llm=DemoStreamingLLM(),
    )
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime),
        tts=DemoStreamingTTS(),
        audio_sink=DemoSpeaker(),
        config=LiveRuntimeConfig(streaming_output=True),
    )
    live.submit_text("介紹 v0.26")
    events = await live.run_once()
    for event in events:
        if event.type.value == "character_delta":
            print("TEXT >", event.text)
        elif event.type.value == "mouth_cue":
            print("MOUTH>", event.data)
        elif event.type.value == "stream_metrics":
            print("METRIC>", event.data)


if __name__ == "__main__":
    asyncio.run(main())
