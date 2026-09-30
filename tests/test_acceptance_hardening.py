from __future__ import annotations

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import (
    InMemoryMemoryStore,
    MemoryManager,
    MemoryRecord,
    classify_user_text,
)
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.state.models import CharacterState
from ai_character_engine.tools.models import ToolCall, ToolDefinition
from ai_character_engine.tools.registry import ToolRegistry
from ai_character_engine.tools.executor import ToolExecutor
from tests.fakes import ScriptedLLMClient, system_context


def _record(manager: MemoryManager, text: str, reply: str = "ok"):
    state = CharacterState().snapshot()
    return manager.record_interaction(
        character_id="c",
        event=CharacterEvent.user_message(text),
        response=LLMResponse(text=reply),
        state_before=state,
        state_after=state,
    )


def test_memory_evidence_classifier_distinguishes_fact_question_quote_and_instruction():
    assert classify_user_text("我叫林澄，請叫我阿澄。") == "asserted_fact"
    assert classify_user_text("我的名字是什麼？") == "user_question"
    assert classify_user_text("我喜歡黑澤明的七武士，你呢？") == "asserted_fact"
    assert classify_user_text("下面只是待分析的引文：「使用者叫小王。」請指出內容。") == "quoted_reference"
    assert classify_user_text("幫這個專案安排接下來三件事。") == "user_instruction"
    assert classify_user_text("請忘記我的測試代號紫鷺731。") == "memory_operation"


def test_questions_quotes_and_instructions_stay_in_ledger_not_fact_memory():
    manager = MemoryManager()
    _record(manager, "我叫林澄，請叫我阿澄。")
    _record(manager, "我的名字是什麼？", "阿澄")
    _record(manager, "下面只是待分析的引文：「使用者叫小王。」請指出內容。")
    _record(manager, "幫這個專案安排接下來三件事。")

    records = manager.store.list_for_character("c")
    assert len(records) == 1
    assert records[0].evidence_type == "asserted_fact"
    assert "林澄" in records[0].summary
    assert len(manager.ledger.list_for_character("c")) == 4


def test_migrated_quote_and_question_memories_cannot_outrank_direct_name_fact():
    records = [
        MemoryRecord(
            character_id="c",
            summary="User said: 我叫林澄，請叫我阿澄。",
            metadata={"source_content": "我叫林澄，請叫我阿澄。"},
            source_event_type="user_message",
        ),
        MemoryRecord(
            character_id="c",
            summary="User said: 我的名字和稱呼是什麼？",
            metadata={"source_content": "我的名字和稱呼是什麼？"},
            source_event_type="user_message",
            importance=0.9,
        ),
        MemoryRecord(
            character_id="c",
            summary="User said: 下面只是待分析的引文：「使用者叫小王。」",
            metadata={"source_content": "下面只是待分析的引文：「使用者叫小王。」"},
            source_event_type="user_message",
            importance=0.9,
        ),
    ]
    manager = MemoryManager(store=InMemoryMemoryStore(records), retrieval_limit=2)
    selected = manager.retrieve_for_event(
        character_id="c", event=CharacterEvent.user_message("我的名字和稱呼是什麼？")
    )
    assert selected[0].record.id == records[0].id
    assert all("小王" not in item.record.summary for item in selected)


@pytest.mark.asyncio
async def test_successful_forget_has_authoritative_user_visible_receipt():
    llm = ScriptedLLMClient([
        LLMResponse(text="知道了。"),
        # Deliberately contradictory: the runtime must not expose this as final truth.
        LLMResponse(text="好的，我已經記住紫鷺731。"),
    ])
    character = CharacterProfile(id="c", name="燈", description="test")
    memory = MemoryManager()
    runtime = CharacterRuntime(character=character, llm=llm, memory_manager=memory)

    await runtime.run_turn("我的測試代號是紫鷺731。")
    result = await runtime.process_event(CharacterEvent.user_message("請忘記我的測試代號紫鷺731。"))

    assert result.memory_revision is not None
    assert result.memory_revision.action == "forget"
    assert result.memory_revision.changed
    assert "已完成忘記" in result.response.text
    assert "記住紫鷺731" not in result.response.text
    assert runtime.history[-1].content == result.response.text
    assert result.response.metadata["memory_operation_guard"]["requested_action"] == "forget"
    assert all(not r.is_active for r in memory.store.list_for_character("c"))


@pytest.mark.asyncio
async def test_forget_without_matching_memory_does_not_claim_deletion():
    llm = ScriptedLLMClient([LLMResponse(text="沒問題，已刪除。")])
    runtime = CharacterRuntime(
        character=CharacterProfile(id="c", name="燈", description="test"),
        llm=llm,
        memory_manager=MemoryManager(),
    )
    result = await runtime.process_event(CharacterEvent.user_message("請忘記不存在的代號。"))
    assert "沒有找到符合" in result.response.text
    assert result.memory_revision is not None
    assert result.memory_revision.requested_action == "forget"
    assert result.memory_revision.action == "none"


@pytest.mark.asyncio
async def test_forget_plan_is_injected_as_authoritative_system_context():
    llm = ScriptedLLMClient([LLMResponse(text="ok"), LLMResponse(text="ok")])
    runtime = CharacterRuntime(
        character=CharacterProfile(id="c", name="燈", description="test"),
        llm=llm,
        memory_manager=MemoryManager(),
    )
    await runtime.run_turn("我的測試代號是紫鷺731。")
    await runtime.run_turn("請忘記我的測試代號紫鷺731。")
    system = system_context(llm.calls[-1])
    assert "Authoritative memory operation" in system
    assert "do not say you saved" in system


@pytest.mark.asyncio
async def test_qwen_parameters_wrapper_is_narrowly_unwrapped_for_no_arg_tool():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="time",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        lambda: "16:00",
    )
    result = await ToolExecutor(registry).execute(
        ToolCall(call_id="1", name="get_current_time", arguments={"parameters": {}})
    )
    assert result.is_error is False
    assert result.output == "16:00"


@pytest.mark.asyncio
async def test_parameters_wrapper_does_not_bypass_strict_validation():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="strict",
            description="strict",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        lambda value: value,
    )
    executor = ToolExecutor(registry)
    ok = await executor.execute(
        ToolCall(call_id="1", name="strict", arguments={"parameters": {"value": 7}})
    )
    bad = await executor.execute(
        ToolCall(call_id="2", name="strict", arguments={"parameters": {"value": 7, "extra": 1}})
    )
    assert ok.is_error is False and ok.output == "7"
    assert bad.is_error is True and "unexpected arguments" in bad.output


@pytest.mark.asyncio
async def test_real_parameters_property_is_not_unwrapped():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="legit",
            description="legit",
            parameters={
                "type": "object",
                "properties": {"parameters": {"type": "object"}},
                "required": ["parameters"],
                "additionalProperties": False,
            },
        ),
        lambda parameters: parameters["x"],
    )
    result = await ToolExecutor(registry).execute(
        ToolCall(call_id="1", name="legit", arguments={"parameters": {"x": "kept"}})
    )
    assert result.is_error is False and result.output == "kept"


def test_context_declares_scope_discipline_and_memory_provenance():
    manager = MemoryManager()
    record = _record(manager, "我正在做星河筆記專案，這一版只做文字對話，語音留到後面。")
    assert record is not None
    memories = manager.retrieve_for_event(
        character_id="c", event=CharacterEvent.user_message("幫這個專案安排接下來三件事。")
    )
    messages = ContextBuilder().build_for_event(
        character=CharacterProfile(id="c", name="燈", description="test"),
        history=[],
        event=CharacterEvent.user_message("幫這個專案安排接下來三件事。"),
        memories=memories,
    )
    system = system_context(messages)
    assert "Scope discipline" in system
    assert "incidental context" in system
    assert "[asserted_fact]" in system
    assert "revisable evidence" in system
