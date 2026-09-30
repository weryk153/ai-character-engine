"""v0.24 offline live-runtime example.

This demonstrates the orchestration boundary without microphones, cameras,
network calls or host imports. Real hosts replace the fake providers and
push device observations into LiveCharacterOrchestrator.
"""

import asyncio
from datetime import UTC, datetime

from ai_character_engine import (
    CharacterProfile,
    CharacterRuntime,
    LiveCharacterOrchestrator,
    ProactiveCandidate,
)
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.voice.models import AudioFormat, SynthesizedAudio, TranscriptResult


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=f"I heard: {messages[-1].content[:60]}")


class DemoSTT:
    async def transcribe(self, audio, *, audio_format):
        return TranscriptResult("hello from the microphone fixture", language="en")


class DemoTTS:
    async def synthesize(self, text, *, voice=None):
        return SynthesizedAudio(b"demo-audio", AudioFormat())


async def main():
    runtime = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "Warm and concise"),
        llm=DemoLLM(),
    )
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime),
        stt=DemoSTT(),
        tts=DemoTTS(),
    )

    live.submit_audio(b"fixture")
    live.submit_candidate(
        ProactiveCandidate(
            "The user has been idle for a while.",
            source="demo_host",
            dedupe_key="idle",
            created_at=datetime.now(UTC),
        )
    )

    # Foreground audio is always considered before autonomy at the next
    # scheduling boundary.
    for event in await live.run_once():
        print(event.type, event.text or event.data)

    for event in await live.run_once():
        print(event.type, event.text or event.data)

    await live.close()


if __name__ == "__main__":
    asyncio.run(main())
