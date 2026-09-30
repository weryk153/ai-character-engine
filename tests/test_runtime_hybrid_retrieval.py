import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context import ContextBudget, ContextBuilder
from ai_character_engine.events import CharacterEvent
from ai_character_engine.memory import (
    HybridMemoryRetriever,
    InMemoryMemoryStore,
    MemoryManager,
    MemoryRecord,
    RetrievedMemory,
    VectorMemoryRetriever,
)
from ai_character_engine.runtime import CharacterRuntime
from tests.fakes import FakeLLMClient, system_context
from tests.test_hybrid_retrieval import SemanticTestEmbedding


def make_runtime(store, *, direct=False, builder=None):
    retriever = HybridMemoryRetriever(store, embedding=SemanticTestEmbedding())
    manager = MemoryManager(
        store=store, retriever=retriever, auto_consolidate_threshold=None
    )
    llm = FakeLLMClient("ok")
    runtime = CharacterRuntime(
        character=CharacterProfile(
            id="test", name="Test", description="Test character"
        ),
        llm=llm,
        memory_manager=None if direct else manager,
        memory_retriever=retriever if direct else None,
        context_builder=builder,
    )
    return runtime, manager, retriever, llm


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_runtime_hybrid_interface_and_context_trace(direct):
    memory = MemoryRecord(id="coffee", character_id="test", summary="coffee")
    runtime, manager, retriever, llm = make_runtime(
        InMemoryMemoryStore([memory]), direct=direct
    )
    result = await runtime.process_event(CharacterEvent.user_message("espresso"))
    assert result.retrieval_trace.strategy == "hybrid"
    assert result.retrieval_trace.selected_memory_ids == ("coffee",)
    assert result.context_trace.selected_memory_ids == ("coffee",)
    assert "coffee" in system_context(llm.calls[0])
    assert "embedding_metadata" not in system_context(llm.calls[0])
    assert "fusion_score" not in system_context(llm.calls[0])
    if not direct:
        assert manager.last_retrieval_trace == result.retrieval_trace
    # Future calls cannot mutate an earlier run's trace.
    retriever.retrieve(character_id="test", query="cinema")
    assert result.retrieval_trace.selected_memory_ids == ("coffee",)


@pytest.mark.asyncio
async def test_retrieval_is_distinct_from_token_budget_selection():
    memory = MemoryRecord(id="coffee", character_id="test", summary="coffee")
    runtime, _, _, llm = make_runtime(
        InMemoryMemoryStore([memory]),
        builder=ContextBuilder(
            budget=ContextBudget(max_memory_tokens=0),
        ),
    )
    result = await runtime.process_event(CharacterEvent.user_message("espresso"))
    assert result.retrieval_trace.selected_memory_ids == ("coffee",)
    assert result.context_trace.selected_memory_ids == ()
    assert result.context_trace.dropped_memories == 1
    assert "Relevant long-term memories" not in system_context(llm.calls[0])


@pytest.mark.asyncio
async def test_runtime_accepts_retrieve_only_custom_interface():
    memory = MemoryRecord(id="custom", character_id="test", summary="coffee")

    class Custom:
        def retrieve(self, *, character_id, query, limit=5):
            assert "coffee" in query and "topic" in query
            return [RetrievedMemory(memory, 0.8)]

    runtime = CharacterRuntime(
        character=CharacterProfile(id="test", name="Test", description="Test"),
        llm=FakeLLMClient(),
        memory_retriever=Custom(),
    )
    result = await runtime.process_event(
        CharacterEvent(
            type="user_message",
            source="user",
            content="remember",
            payload={"topic": "coffee"},
        )
    )
    assert result.retrieval_trace.strategy == "custom"
    assert result.retrieved_memories[0].record.id == "custom"


@pytest.mark.asyncio
async def test_runtime_correction_and_forgetting_invalidate_warmed_index():
    memory = MemoryRecord(id="old", character_id="test", summary="I like coffee")
    store = InMemoryMemoryStore([memory])
    runtime, manager, retriever, _ = make_runtime(store)
    assert (
        retriever.retrieve(character_id="test", query="espresso")[0].record.id == "old"
    )
    correction = await runtime.process_event(
        CharacterEvent(
            type="user_message",
            source="user",
            content="I no longer like coffee",
            payload={"memory_action": "correct", "memory_target_ids": ["old"]},
        )
    )
    replacement = correction.memory_written
    assert replacement and correction.memory_revision.action == "supersede"
    result = retriever.retrieve_with_trace(character_id="test", query="espresso")
    assert [m.record.id for m in result.memories] == [replacement.id]
    assert result.trace.removed_vectors == 1
    forgotten = await runtime.process_event(
        CharacterEvent(
            type="user_message",
            source="user",
            content="forget my coffee preference",
            payload={"memory_action": "forget", "memory_target_ids": [replacement.id]},
        )
    )
    assert forgotten.memory_revision.action == "forget"
    assert not retriever.retrieve(character_id="test", query="espresso")
    assert len(manager.ledger.list_for_character("test")) == 2


def test_consolidation_replaces_warmed_vector_candidates():
    store = InMemoryMemoryStore(
        [
            MemoryRecord(id="a", character_id="test", summary="I like coffee"),
            MemoryRecord(id="b", character_id="test", summary="I like coffee"),
        ]
    )
    retriever = VectorMemoryRetriever(store, embedding=SemanticTestEmbedding())
    retriever.retrieve(character_id="test", query="espresso")
    manager = MemoryManager(store=store, retriever=retriever)
    consolidation = manager.consolidate(character_id="test")
    assert consolidation.changed
    result = retriever.retrieve_with_trace(character_id="test", query="espresso")
    assert result.trace.removed_vectors == 2
    assert result.trace.embedded_records == 1
    assert [m.record.id for m in result.memories] == [consolidation.created[0].id]


@pytest.mark.asyncio
async def test_runtime_write_is_visible_on_next_retrieval():
    runtime, _, retriever, _ = make_runtime(InMemoryMemoryStore())
    first = await runtime.process_event(CharacterEvent.user_message("I like coffee"))
    assert not first.retrieved_memories
    assert first.memory_written is not None
    second = await runtime.process_event(CharacterEvent.user_message("espresso"))
    assert first.memory_written.id in second.retrieval_trace.selected_memory_ids


@pytest.mark.asyncio
async def test_runtime_without_memory_has_no_retrieval_trace():
    runtime = CharacterRuntime(
        character=CharacterProfile(id="test", name="Test", description="Test"),
        llm=FakeLLMClient(),
    )
    result = await runtime.process_event(CharacterEvent.user_message("hello"))
    assert result.retrieval_trace is None and not result.retrieved_memories
