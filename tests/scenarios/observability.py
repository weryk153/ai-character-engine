from __future__ import annotations

import asyncio

from ai_character_engine import CharacterEvent, CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.observability import InMemoryObservabilitySink, TraceContext, Tracer
from ai_character_engine.runtime import CharacterRuntime


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(
            text="觀測資料不應改變角色答案。",
            model="demo-model",
            input_tokens=42,
            output_tokens=9,
            latency_ms=18.5,
        )


async def main() -> None:
    sink = InMemoryObservabilitySink()
    tracer = Tracer(sink)
    runtime = CharacterRuntime(
        character=CharacterProfile(id="mei", name="Mei", description="demo"),
        llm=DemoLLM(),
        tracer=tracer,
    )

    result = await runtime.process_event(
        CharacterEvent.user_message("今天狀況如何？"),
        trace_context=TraceContext(trace_id="demo-trace", request_id="demo-request"),
    )
    print(result.text)
    print("spans:", [span.name for span in sink.spans])
    print("turn:", sink.turns[-1])
    print("summary:", sink.summary())


if __name__ == "__main__":
    asyncio.run(main())
