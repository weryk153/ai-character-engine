from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.observability import TraceContext
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.state.models import CharacterState, CharacterStateSnapshot

from .manager import SessionManager
from .models import RelationshipSnapshot, SessionRecord, SessionRuntimeSnapshot
from .store import InMemoryRelationshipStore, RelationshipStore


def state_from_snapshot(snapshot: CharacterStateSnapshot) -> CharacterState:
    return CharacterState(
        emotion=snapshot.emotion,
        energy=snapshot.energy,
        trust=snapshot.trust,
        favorability=snapshot.favorability,
        relationship_stage=snapshot.relationship_stage,
        custom=dict(snapshot.custom),
        mood_intensity=snapshot.mood_intensity,
        mood_updated_at=snapshot.mood_updated_at,
    )


def snapshot_runtime(runtime: CharacterRuntime) -> SessionRuntimeSnapshot:
    return SessionRuntimeSnapshot(
        state=runtime.state.snapshot(),
        history=tuple(runtime.history),
    )


class ManagedCharacterSession:
    """A CharacterRuntime bound to a persisted user/session identity."""

    def __init__(
        self,
        *,
        record: SessionRecord,
        runtime: CharacterRuntime,
        session_manager: SessionManager,
        relationship_store: RelationshipStore | None = None,
    ) -> None:
        self.record = record
        self.runtime = runtime
        self.session_manager = session_manager
        self.relationship_store = relationship_store or InMemoryRelationshipStore()

    async def run_turn(self, user_message: str):
        result = await self.process_event(CharacterEvent.user_message(user_message))
        return result.response

    async def process_event(
        self,
        event: CharacterEvent,
        *,
        trace_context: TraceContext | None = None,
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CharacterRunResult:
        # Relationship data is user × character scoped and may be shared by
        # multiple conversation sessions. Refresh it just before processing.
        relationship = self.relationship_store.get(
            user_id=self.record.user_id,
            character_id=self.record.character_id,
        )
        if relationship is not None:
            self.runtime.state.trust = relationship.trust
            self.runtime.state.favorability = relationship.favorability
            self.runtime.state.relationship_stage = relationship.relationship_stage

        result = await self.runtime.process_event(
            event, trace_context=trace_context, on_text_delta=on_text_delta
        )
        self.relationship_store.put(
            RelationshipSnapshot.from_state(
                user_id=self.record.user_id,
                character_id=self.record.character_id,
                state=self.runtime.state.snapshot(),
                updated_at=datetime.now(UTC),
            )
        )
        self.record = self.session_manager.touch(
            self.record.id,
            snapshot=snapshot_runtime(self.runtime),
            metadata_updates={"last_event_id": event.id},
        )
        return result

    def close(self) -> SessionRecord:
        # Persist the latest in-memory state/history even if the host mutated
        # state between events, then mark the session closed.
        self.record = self.session_manager.touch(
            self.record.id,
            snapshot=snapshot_runtime(self.runtime),
        )
        self.record = self.session_manager.close(self.record.id)
        return self.record
