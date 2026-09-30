"""Offline distributed-worker semantics demo.

No network, cloud queue, provider, renderer, or authoritative character store is used.
The in-memory broker demonstrates the contract a real deployment adapter preserves.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from ai_character_engine.distributed import (
    DistributedBrokerConfig,
    DistributedTaskCoordinator,
    DistributedWorker,
    InMemoryDistributedTaskBroker,
)
from ai_character_engine.llm.models import Message
from ai_character_engine.production import Idempotency
from ai_character_engine.tasks import TaskOutput, TaskRequest, TaskSnapshot, TaskStateSnapshot


def make_snapshot() -> TaskSnapshot:
    return TaskSnapshot(
        revision=12,
        captured_at=datetime.now(UTC),
        character_id="demo-character",
        character_name="Demo",
        state=TaskStateSnapshot(
            emotion="calm",
            energy=0.8,
            trust=0.5,
            favorability=0.5,
            relationship_stage="familiar",
        ),
        history=(Message("user", "I prefer concise answers."),),
        memory_scope_id="demo-memory",
    )


async def main() -> None:
    broker = InMemoryDistributedTaskBroker(
        config=DistributedBrokerConfig(max_attempts=3)
    )
    coordinator = DistributedTaskCoordinator(broker)
    request = TaskRequest(
        task_type="reflection",
        payload={"instruction": "extract a response-style observation"},
        source="offline-example",
        id="demo-task",
    )
    await coordinator.submit(
        request,
        make_snapshot(),
        idempotency=Idempotency.SAFE,
    )

    attempts = 0

    def reflection_handler(ctx):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("simulated transient worker dependency failure")
        return TaskOutput(
            proposals=(ctx.proposal(
                "cognition.reflection",
                {"insight": "The user may prefer concise answers."},
                confidence=0.9,
                provenance={"mode": "offline-demo"},
            ),)
        )

    worker = DistributedWorker(
        broker,
        worker_id="worker-demo",
        handlers={"reflection": reflection_handler},
    )

    await worker.run_once()  # safe transient failure -> requeue
    after_first = await broker.snapshot(request.id)
    print(f"after first attempt: {after_first.status.value}")

    await worker.run_once()  # second attempt succeeds
    final = await coordinator.wait(request.id, timeout_s=1)
    proposal = final.result.proposals[0]
    print(f"final status: {final.status.value}")
    print(f"attempts: {final.attempts}")
    print(f"proposal target: {proposal.target}")
    print(f"proposal base revision: {proposal.base_revision}")
    print("remote result remains non-authoritative")


if __name__ == "__main__":
    asyncio.run(main())
