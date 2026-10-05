# Integration

For SDK 1.1.0. `ENGINE_ROOT` is the engine checkout; the host has its own working directory and virtual environment.

## One character for a host

`CharacterCompanion` is the engine assembled for a host that talks to one character. See `ENGINE_ROOT/docs/companion.md` for the full example and `examples/companion_chat.py` for a terminal chat. Everything is stored under `storage_dir`; run again and the character still knows the user. Turn hidden reasoning off for a local reasoning model (`request_options={"extra_body": {"reasoning_effort": "none"}}` on LM Studio), or the thought alone uses up the output limit. Call `settle()` before reading `snapshot()` or `memories()` and before closing, so that what she took from the last turns is not lost.

## Local model: two turns

Start the compatible server first and take the real model identifier. The address below is an example, not a running service.

```python
import asyncio
from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.llm.local import OpenAICompatibleChatClient


async def main():
    llm = OpenAICompatibleChatClient(
        base_url="http://127.0.0.1:1234/v1",
        model="your-loaded-model",
    )
    runtime = CharacterRuntime(
        character=CharacterProfile(
            id="mei", name="Mei",
            description="A friendly companion who gives concise answers.",
        ),
        llm=llm,
    )
    try:
        print((await runtime.run_turn("Call me Alex.")).text)
        print((await runtime.run_turn("What name did I ask you to use?")).text)
    finally:
        await llm.client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

This has the history of the current process only: no persistent memory, session store, background reflection or goal manager.

`run_turn(text)` returns an `LLMResponse`; read `.text`. For memory retrieval, events and turn data, use `process_event(CharacterEvent.user_message(text))`, which returns a `CharacterRunResult`; `CharacterEvent` is imported from `ai_character_engine.events.models`. The two methods return different types.

`OpenAIResponsesClient` is the other provider path; `examples/basic_chat.py` and `examples/tool_chat.py` configure it from `OPENAI_API_KEY` and `AI_CHARACTER_MODEL` in `.env`. Changing the model name alone does not connect to LM Studio.

A custom client implements an asynchronous `generate(messages, *, tools=None)` returning `LLMResponse`. Real provider streaming also needs `stream_generate`; cutting a complete string into pieces is no streaming verification.

## Tools

```python
from ai_character_engine.tools.models import ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry


def build_tools():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="add_numbers",
            description="Add two numbers exactly.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        ),
        lambda a, b: a + b,
    )
    return registry
```

Pass the result as `CharacterRuntime(..., tool_registry=build_tools())`. The model must support tool calling; where a tool acts on the host, the handler and the permission layer still check what is allowed.

`ENGINE_ROOT/examples/tool_chat.py` shows the whole flow. Tests check the handler's actual arguments and that the result reaches the model's context, not only the final answer.

## Session save and restore

The snippet assumes `character`, `data_dir` and `make_llm` exist. `data_dir` is a durable directory the host can write; `make_llm` returns a configured client each time.

```python
from ai_character_engine.session import (
    CharacterRuntimeFactory,
    JsonFileRelationshipStore,
    JsonFileSessionStore,
    SessionManager,
)

manager = SessionManager(
    JsonFileSessionStore(data_dir / "sessions.json"),
    default_ttl_seconds=None,
)
factory = CharacterRuntimeFactory(
    characters={character.id: character},
    llm_factory=lambda record: make_llm(),
    session_manager=manager,
    relationship_store=JsonFileRelationshipStore(data_dir / "relationships.json"),
)
session = factory.create(
    user_id="user-1",
    character_id=character.id,
    session_id="session-1",
)
```

Use `await session.run_turn(text)` so that the session layer saves. After a restart, rebuild the manager and factory on the same paths and call `factory.restore("session-1")`; do not `create` with the same id and call it a restore.

`ENGINE_ROOT/examples/session_runtime.py` shows the whole flow with a temporary directory. Real use needs a durable location and management of the client's close lifecycle.

## Long-term memory

```python
from ai_character_engine.memory import JsonlMemoryStore, MemoryManager

memory = MemoryManager(store=JsonlMemoryStore(data_dir / "memory.jsonl"))
```

Pass `memory_manager=memory` to the runtime. Set the scope by user, character and session; the session factory binds the scope, and a runtime built directly takes a `memory_scope_id`. One file in an example is not multi-tenant isolation.

To verify retrieval, build a new runtime with an empty history, the same store and the same scope, run a new message through `process_event` and check `result.retrieved_memories` on its return value. Keep the earlier history out, or the source of the answer cannot be told. See `ENGINE_ROOT/tests/scenarios/long_term_memory.py` and `memory_revision.py`. A forgotten memory leaves retrieval; that is not the deletion of every history and backup.

## Common problems

- **Module not found**: check the Python, the virtual environment and `ai_character_engine.__file__`; install into the same environment. pytest's `pythonpath` does not reach separate subprocesses.
- **Every turn is like the first**: check whether the runtime is rebuilt each time, whether the session key changes, and whether the history or context budget is too small.
- **The local server cannot be reached**: check the URL, port and whether the model is loaded; do not add a cloud service in its place.
- **No continuity**: tell history, session restore and memory retrieval apart, then check the paths, the scope and the write policy.
- **The API does not match**: read the signatures of the installed version and `ENGINE_ROOT/docs/api-reference.md`; do not guess methods from names.
