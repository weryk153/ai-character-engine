"""Offline v0.25 duplex voice / VAD barge-in demonstration.

No microphone, speaker, cloud model, or TTS service is required. The example uses
synthetic PCM16 chunks and a cooperative sink whose second sentence blocks until
VAD barge-in stops playback.
"""
from __future__ import annotations

import asyncio
from array import array

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import DuplexVoiceConfig, LiveCharacterOrchestrator
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.voice import AudioChunk, AudioFormat, SynthesizedAudio, TranscriptResult


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="第一句已經播完。第二句會被使用者打斷。")


class DemoSTT:
    async def transcribe(self, audio, *, audio_format):
        return TranscriptResult("等等，我有新問題", language="zh", confidence=0.99)


class DemoTTS:
    async def synthesize(self, text, *, voice=None):
        return SynthesizedAudio(text.encode("utf-8"), AudioFormat(), duration_ms=300)


class DemoSink:
    def __init__(self):
        self.count = 0
        self.second_started = asyncio.Event()

    async def play(self, chunk):
        self.count += 1
        print(f"speaker segment {self.count}: {chunk.data.decode('utf-8')}")
        if self.count == 2:
            self.second_started.set()
            await asyncio.Event().wait()

    async def stop(self):
        print("speaker: stop")


def pcm(value: int, sequence: int) -> AudioChunk:
    return AudioChunk(array("h", [value] * 160).tobytes(), AudioFormat(), sequence=sequence)


async def main():
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "v0.25 duplex demo"),
        llm=DemoLLM(),
    )
    sink = DemoSink()
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime),
        stt=DemoSTT(),
        tts=DemoTTS(),
        audio_sink=sink,
        duplex=DuplexVoiceConfig(
            automatic_barge_in=True,
            echo_cancellation_confirmed=True,  # host assertion for this synthetic demo
            barge_in_min_speech_chunks=2,
            end_silence_chunks=2,
        ),
    )

    live.submit_text("請說兩句")
    old_turn = asyncio.create_task(live.run_once())
    await sink.second_started.wait()

    for chunk in (pcm(2000, 0), pcm(2000, 1), pcm(0, 2), pcm(0, 3)):
        for event in await live.ingest_audio_chunk(chunk):
            print("control:", event.type, event.data)

    old_events = await old_turn
    print("old turn:", [event.type for event in old_events])
    print("history assistant:", runtime.history[-1].content)

    new_events = await live.run_once()
    print("new turn:", [(event.type, event.text) for event in new_events])


if __name__ == "__main__":
    asyncio.run(main())
