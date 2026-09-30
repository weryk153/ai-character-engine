"""Offline v0.33 Commit Coordinator example.

Run:
    PYTHONPATH=src python tests/scenarios/commit_coordinator.py
"""
from __future__ import annotations

import asyncio

from ai_character_engine import (
    CharacterProfile,
    CharacterRuntime,
    CognitiveCommitCoordinator,
    CommitStatus,
    MemoryManager,
    MultiTaskRuntime,
    TaskProposal,
)
from ai_character_engine.llm.models import LLMResponse


class OfflineLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="foreground reply", model="offline")


def proposal(target: str, payload: dict, *, revision: int, worker: str, confidence: float = 0.9):
    return TaskProposal(
        target=target,
        payload=payload,
        base_revision=revision,
        source_task_id=f"task-{worker}",
        confidence=confidence,
        provenance={
            "worker_kind": worker,
            "foreground_event_id": "event-1",
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "evidence_type": "asserted_fact",
        },
    )


async def main() -> None:
    engine = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "commit coordinator demo"),
        llm=OfflineLLM(),
        memory_manager=MemoryManager(),
    )
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)

    memory = proposal(
        "memory.append_candidate",
        {"summary": "User prefers tea", "kind": "preference", "importance": 0.8},
        revision=0,
        worker="memory_extraction",
        confidence=0.93,
    )
    memory_result = await commits.commit(memory)
    print("MEMORY", memory_result.status.value, memory_result.commit_sequence)

    emotion = proposal(
        "state.emotion_candidate",
        {"emotion": "frustrated", "intensity": 0.7},
        revision=0,
        worker="emotion_analysis",
        confidence=0.86,
    )
    emotion_result = await commits.commit(emotion)
    print("EMOTION", emotion_result.status.value, engine.state.custom["observed_user_emotion"])
    print("CHARACTER_EMOTION", engine.state.emotion)

    reflection = proposal(
        "cognition.reflection_candidate",
        {"insight": "Ask about the unresolved work issue later."},
        revision=0,
        worker="reflection",
        confidence=0.92,
    )
    reflection_result = await commits.commit(reflection)
    print("REFLECTION", reflection_result.status.value, reflection_result.next_action.value)

    await tasks.run_turn("advance the authoritative foreground revision")
    stale = proposal(
        "memory.append_candidate",
        {"summary": "User likes coffee", "kind": "preference", "importance": 0.7},
        revision=0,
        worker="memory_extraction",
        confidence=0.91,
    )
    stale_result = await commits.commit(stale)
    print("STALE", stale_result.status.value, stale_result.next_action.value)

    assert memory_result.status is CommitStatus.COMMITTED
    assert emotion_result.status is CommitStatus.COMMITTED
    assert reflection_result.status is CommitStatus.REVIEW_REQUIRED
    assert stale_result.status is CommitStatus.STALE
    print("FOREGROUND_REVISION", tasks.revision)
    print("COMMIT_SEQUENCE", commits.commit_sequence)


if __name__ == "__main__":
    asyncio.run(main())
