"""Offline demonstration of safe retry versus uncertain storage effects."""
import asyncio
from datetime import UTC, datetime, timedelta

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import AutonomyController, AutonomyScheduler, ProactiveCandidate
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory.ledger import InMemoryEventLedger
from ai_character_engine.memory.manager import MemoryManager
from ai_character_engine.state.models import StatePatch
from ai_character_engine.state.policy import NoopStatePolicy


class TrustPolicy(NoopStatePolicy):
    def on_event(self, event, state):
        return StatePatch(trust_delta=5)


class FailOnceLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("synthetic inference failure")
        return LLMResponse(text="Recovered")


class AmbiguousLedger(InMemoryEventLedger):
    def append(self, entry):
        super().append(entry)
        raise OSError("synthetic acknowledgement failure")


async def main():
    clock = [datetime(2026, 9, 21, 12, tzinfo=UTC)]
    engine = CharacterRuntime(
        character=CharacterProfile("demo", "Demo", "A careful character"),
        llm=FailOnceLLM(), state_policy=TrustPolicy(),
    )
    scheduler = AutonomyScheduler(clock=lambda: clock[0])
    item = ProactiveCandidate(content="Notice", source="demo", created_at=clock[0])
    scheduler.submit(item)
    controller = AutonomyController(engine, scheduler)
    failed = await controller.run_once()
    assert failed.status == "failed" and engine.state.trust == 50
    print("inference failure: trust restored to 50; retry after 1 second")
    assert (await controller.run_once()).status == "empty"
    clock[0] += timedelta(seconds=1)
    success = await controller.run_once()
    assert success.status == "delivered" and engine.state.trust == 55
    print("retry delivered: trust=55; history messages=2")

    engine.memory_manager = MemoryManager(ledger=AmbiguousLedger())
    clock[0] += timedelta(seconds=30)
    item = ProactiveCandidate(content="Another notice", source="demo", created_at=clock[0])
    scheduler.submit(item)
    failed = await controller.run_once()
    assert failed.requires_review and failed.reason == "partial_execution"
    assert (await controller.run_once()).status == "empty"
    assert len(engine.memory_manager.ledger.list_for_character("demo")) == 1
    assert len(engine.history) == 2 and engine.state.trust == 55
    print("storage uncertainty: held for review; no automatic retry")
    # Demo host sees a recorded ledger entry and explicitly discards this job.
    # Real hosts must reconcile every affected system, not just a local counter.
    scheduler.discard(item)
    print("host reconciled the demo ledger and discarded the held candidate")


if __name__ == "__main__":
    asyncio.run(main())
