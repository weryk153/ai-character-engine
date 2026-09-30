import asyncio

import pytest

from ai_character_engine.tools.executor import ToolExecutor
from ai_character_engine.tools.models import ToolCall, ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry


@pytest.fixture
def calculator_definition() -> ToolDefinition:
    return ToolDefinition(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


@pytest.mark.asyncio
async def test_executor_runs_sync_tool(calculator_definition: ToolDefinition) -> None:
    registry = ToolRegistry()
    registry.register(calculator_definition, lambda a, b: a + b)
    executor = ToolExecutor(registry)

    result = await executor.execute(ToolCall(call_id="1", name="add", arguments={"a": 2, "b": 3}))

    assert result.output == "5"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_executor_validates_arguments(calculator_definition: ToolDefinition) -> None:
    registry = ToolRegistry()
    registry.register(calculator_definition, lambda a, b: a + b)
    executor = ToolExecutor(registry)

    result = await executor.execute(ToolCall(call_id="1", name="add", arguments={"a": 2}))

    assert result.is_error is True
    assert "missing required arguments" in result.output


@pytest.mark.asyncio
async def test_executor_handles_unknown_tool() -> None:
    executor = ToolExecutor(ToolRegistry())

    result = await executor.execute(ToolCall(call_id="1", name="missing", arguments={}))

    assert result.is_error is True
    assert "unknown tool" in result.output


@pytest.mark.asyncio
async def test_executor_times_out() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow",
            description="A deliberately slow tool.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: asyncio.sleep(0.05),
    )
    executor = ToolExecutor(registry, timeout_seconds=0.001)

    result = await executor.execute(ToolCall(call_id="1", name="slow", arguments={}))

    assert result.is_error is True
    assert "timed out" in result.output

@pytest.mark.asyncio
async def test_executor_requires_approval() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="unlock_gallery",
            description="Unlock paid gallery content.",
            parameters={
                "type": "object",
                "properties": {"gallery_id": {"type": "string"}},
                "required": ["gallery_id"],
                "additionalProperties": False,
            },
            requires_approval=True,
        ),
        lambda gallery_id: f"unlocked:{gallery_id}",
    )
    executor = ToolExecutor(registry)

    result = await executor.execute(
        ToolCall(call_id="1", name="unlock_gallery", arguments={"gallery_id": "g-1"})
    )

    assert result.is_error is True
    assert "requires approval" in result.output


@pytest.mark.asyncio
async def test_executor_runs_approved_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="unlock_gallery",
            description="Unlock paid gallery content.",
            parameters={
                "type": "object",
                "properties": {"gallery_id": {"type": "string"}},
                "required": ["gallery_id"],
                "additionalProperties": False,
            },
            requires_approval=True,
        ),
        lambda gallery_id: f"unlocked:{gallery_id}",
    )
    executor = ToolExecutor(registry, authorizer=lambda call: call.arguments["gallery_id"] == "g-1")

    result = await executor.execute(
        ToolCall(call_id="1", name="unlock_gallery", arguments={"gallery_id": "g-1"})
    )

    assert result.is_error is False
    assert result.output == "unlocked:g-1"
