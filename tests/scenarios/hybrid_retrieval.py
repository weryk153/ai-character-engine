"""Offline end-to-end retrieval/context example; no API key or network required."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime

from ai_character_engine import CharacterEvent, CharacterProfile, CharacterRuntime
from ai_character_engine.context import ContextBudget, ContextBuilder
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import (
    HashEmbeddingProvider,
    HybridMemoryRetriever,
    ImportanceRecencyReranker,
    InMemoryMemoryStore,
    MemoryManager,
    MemoryRecord,
)


class DemoLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(
            text="這是離線範例：檢索與 context 已建立。", model="offline-demo"
        )


async def main():
    timestamp = datetime(2026, 9, 16, tzinfo=UTC)
    store = InMemoryMemoryStore(
        [
            MemoryRecord(
                id="coffee",
                character_id="akari",
                summary="使用者喜歡咖啡 coffee",
                importance=0.8,
                created_at=timestamp,
            ),
            MemoryRecord(
                id="movies",
                character_id="akari",
                summary="使用者喜歡黑澤明的電影",
                importance=0.6,
                created_at=timestamp,
            ),
            MemoryRecord(
                id="forgotten",
                character_id="akari",
                summary="使用者以前喜歡咖啡加糖",
                status="forgotten",
                created_at=timestamp,
            ),
            MemoryRecord(
                id="other",
                character_id="another-character",
                summary="coffee preference",
                created_at=timestamp,
            ),
        ]
    )
    retriever = HybridMemoryRetriever(
        store,
        embedding=HashEmbeddingProvider(),
        reranker=ImportanceRecencyReranker(),
    )
    manager = MemoryManager(
        store=store, retriever=retriever, auto_consolidate_threshold=None
    )
    runtime = CharacterRuntime(
        character=CharacterProfile(
            id="akari", name="Akari", description="A friendly character"
        ),
        llm=DemoLLM(),
        memory_manager=manager,
        context_builder=ContextBuilder(budget=ContextBudget(max_memory_tokens=150)),
    )
    result = await runtime.process_event(CharacterEvent.user_message("coffee 咖啡"))
    print("Embedding: local hash features (not learned semantic embeddings)")
    print(
        "Retrieved:",
        [(item.record.id, round(item.score, 4)) for item in result.retrieved_memories],
    )
    print("Included in context:", result.context_trace.selected_memory_ids)
    print("Retrieval trace:")
    print(json.dumps(asdict(result.retrieval_trace), ensure_ascii=False, indent=2))
    # A second lookup reuses vectors instead of embedding unchanged documents.
    second = retriever.retrieve_with_trace(
        character_id="akari", query="coffee 咖啡"
    )
    print("Cached vectors on next lookup:", second.trace.cached_vectors)
    assert "coffee" in result.retrieval_trace.selected_memory_ids
    assert "forgotten" not in result.retrieval_trace.selected_memory_ids
    assert "other" not in result.retrieval_trace.selected_memory_ids


if __name__ == "__main__":
    asyncio.run(main())
