"""Offline host lifecycle example. No microphone, TTS, camera or native app.

Run: uv run --no-sync python examples/autonomy_host.py
The application, not the engine, owns tasks and user interruption policy.
"""
import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.autonomy import AutonomyController, AutonomyScheduler, ProactiveCandidate
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.llm.models import LLMResponse


@dataclass
class InactivityDetector:
    last_activity: datetime
    threshold: timedelta = timedelta(minutes=10)
    emitted: bool = False

    def touch(self, now):
        self.last_activity, self.emitted = now, False

    def poll(self, now):
        if self.emitted or now - self.last_activity < self.threshold:
            return None
        self.emitted = True
        return ProactiveCandidate(
            content="The user has been quiet for ten minutes.",
            source="host_idle", priority=30,
            dedupe_key="idle_check_in", cooldown_key="idle_check_in",
            created_at=now, expires_at=now + timedelta(minutes=1),
        )


class HostSession:
    def __init__(self, runtime, clock):
        self.clock = clock
        self.scheduler = AutonomyScheduler(clock=clock)
        self.controller = AutonomyController(runtime, self.scheduler)
        self.bridge = CharacterHostBridge(runtime)
        self.idle = InactivityDetector(clock())
        self.proactive_task = None
        self.closed = False

    def start_tick(self):
        """Call from the host timer; never create an unbounded task backlog."""
        if self.closed or (self.proactive_task and not self.proactive_task.done()):
            return None
        item = self.idle.poll(self.clock())
        if item:
            admission = self.scheduler.submit(item)
            if admission.status == "rejected":
                self.idle.emitted = False  # Retry admission on a later host tick.
        self.proactive_task = asyncio.create_task(self.controller.run_once())
        return self.proactive_task

    async def stop_proactive(self):
        if self.proactive_task:
            self.proactive_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.proactive_task
            self.proactive_task = None

    async def user_text(self, text):
        if self.closed:
            raise RuntimeError("Host session is closed.")
        await self.stop_proactive()  # Wait for rollback and release before chat.
        for item in self.scheduler.pending:
            if item.dedupe_key == "idle_check_in" and not self.scheduler.retry_state(item.id).held_reason:
                self.scheduler.discard(item)
        self.idle.touch(self.clock())
        return await self.bridge.process(text)

    async def close(self):
        self.closed = True
        await self.stop_proactive()
        await self.bridge.close()


class DemoLLM:
    def __init__(self):
        self.started = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        if "proactive_observation" in messages[-1].content:
            self.started.set()
            await asyncio.Event().wait()  # Simulate inference until user interrupts.
        return LLMResponse(text="User turn handled after proactive cancellation.")


async def main():
    clock = [datetime(2026, 9, 21, 12, tzinfo=UTC)]
    llm = DemoLLM()
    runtime = CharacterRuntime(character=CharacterProfile("demo", "Demo", "A careful character"), llm=llm)
    host = HostSession(runtime, lambda: clock[0])
    try:
        clock[0] += timedelta(minutes=10)
        task = host.start_tick()
        await llm.started.wait()
        assert host.start_tick() is None
        print("host: one proactive task; overlapping tick ignored")
        response = await host.user_text("Hello")
        assert task.cancelled()
        assert not host.scheduler.pending and len(runtime.history) == 2
        print(response.text)
        print("host: stale idle candidate removed; history messages=2")
    finally:
        await host.close()
    assert host.start_tick() is None
    print("host: closed; borrowed providers are not closed by the bridge")


if __name__ == "__main__":
    asyncio.run(main())
