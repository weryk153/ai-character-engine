from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from collections.abc import Mapping, Sequence

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.runtime.models import CharacterRunResult

from .errors import CharacterIsolationError, SharedCognitionAccessError, UnknownCharacterError
from .models import (
    CharacterExchange,
    ExchangeStatus,
    KnowledgeVisibility,
    SharedCognitionRecord,
)
from .scheduler import FairCharacterScheduler
from .store import (
    CharacterExchangeStore,
    InMemoryCharacterExchangeStore,
    InMemorySharedCognitionStore,
    SharedCognitionStore,
)


@dataclass(slots=True, frozen=True)
class CharacterInteractionResult:
    exchange: CharacterExchange
    recipient_result: CharacterRunResult


class MultiCharacterRuntime:
    """Coordinates isolated CharacterRuntime instances without merging authority.

    Private runtime state and cognition remain owned by each CharacterRuntime.
    Cross-character information only crosses the boundary through explicit
    messages or SharedCognitionRecord delivery.
    """

    def __init__(
        self,
        runtimes: Mapping[str, CharacterRuntime],
        *,
        scheduler: FairCharacterScheduler | None = None,
        shared_cognition_store: SharedCognitionStore | None = None,
        exchange_store: CharacterExchangeStore | None = None,
        enforce_scope_isolation: bool = True,
    ) -> None:
        self._runtimes = dict(runtimes)
        if not self._runtimes:
            raise ValueError("at least one character runtime is required")
        self.scheduler = scheduler or FairCharacterScheduler()
        self.shared_cognition_store = shared_cognition_store or InMemorySharedCognitionStore()
        self.exchange_store = exchange_store or InMemoryCharacterExchangeStore()
        self.enforce_scope_isolation = enforce_scope_isolation
        self._validate_registry()

    @property
    def character_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtimes))

    def runtime_for(self, character_id: str) -> CharacterRuntime:
        try:
            return self._runtimes[character_id]
        except KeyError as exc:
            raise UnknownCharacterError(character_id) from exc

    async def process_event(self, character_id: str, event: CharacterEvent) -> CharacterRunResult:
        runtime = self.runtime_for(character_id)
        return await self.scheduler.run(character_id, lambda: runtime.process_event(event))

    async def run_turn(self, character_id: str, user_message: str) -> CharacterRunResult:
        return await self.process_event(character_id, CharacterEvent.user_message(user_message))

    def publish_shared_cognition(
        self,
        *,
        owner_character_id: str,
        kind: str,
        content: str,
        visibility: KnowledgeVisibility = KnowledgeVisibility.PRIVATE,
        audience_character_ids: Sequence[str] = (),
        source_type: str = "host",
        source_id: str | None = None,
        metadata: dict | None = None,
    ) -> SharedCognitionRecord:
        self.runtime_for(owner_character_id)
        audience = tuple(audience_character_ids)
        unknown = sorted(set(audience) - set(self._runtimes))
        if unknown:
            raise UnknownCharacterError(",".join(unknown))
        record = SharedCognitionRecord(
            owner_character_id=owner_character_id,
            kind=kind,
            content=content,
            visibility=visibility,
            audience_character_ids=audience,
            source_type=source_type,
            source_id=source_id,
            metadata=dict(metadata or {}),
        )
        self.shared_cognition_store.add(record)
        return record

    def visible_shared_cognition(self, character_id: str) -> tuple[SharedCognitionRecord, ...]:
        self.runtime_for(character_id)
        records = self.shared_cognition_store.list_visible_to(character_id)
        records.sort(key=lambda item: (item.created_at, item.id))
        return tuple(records)

    async def deliver_shared_cognition(
        self,
        record_id: str,
        *,
        recipient_character_id: str,
    ) -> CharacterRunResult:
        self.runtime_for(recipient_character_id)
        record = self.shared_cognition_store.get(record_id)
        if record is None:
            raise KeyError(record_id)
        if not record.visible_to(recipient_character_id):
            raise SharedCognitionAccessError(
                f"shared cognition {record_id} is not visible to {recipient_character_id}"
            )
        event = CharacterEvent(
            type="shared_cognition",
            source=f"character:{record.owner_character_id}",
            content=record.content,
            payload={
                "shared_cognition_id": record.id,
                "owner_character_id": record.owner_character_id,
                "kind": record.kind,
                "visibility": record.visibility.value,
                "source_type": record.source_type,
                "source_id": record.source_id,
                "memory_evidence_type": "event_observation",
                "authoritative": False,
            },
        )
        return await self.process_event(recipient_character_id, event)

    async def send_message(
        self,
        *,
        sender_character_id: str,
        recipient_character_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> CharacterInteractionResult:
        self.runtime_for(sender_character_id)
        self.runtime_for(recipient_character_id)
        exchange = CharacterExchange(
            sender_character_id=sender_character_id,
            recipient_character_id=recipient_character_id,
            content=content,
            metadata=dict(metadata or {}),
        )
        self.exchange_store.add(exchange)
        event = CharacterEvent(
            type="character_message",
            source=f"character:{sender_character_id}",
            content=content,
            payload={
                "exchange_id": exchange.id,
                "sender_character_id": sender_character_id,
                "recipient_character_id": recipient_character_id,
                "memory_evidence_type": "event_observation",
                "authoritative": False,
            },
        )
        try:
            result = await self.process_event(recipient_character_id, event)
        except BaseException as exc:
            failed = replace(
                exchange,
                status=ExchangeStatus.FAILED,
                recipient_event_id=event.id,
                error=f"{type(exc).__name__}: {exc}",
                updated_at=datetime.now(UTC),
            )
            self.exchange_store.replace(failed)
            raise
        delivered = replace(
            exchange,
            status=ExchangeStatus.DELIVERED,
            recipient_event_id=event.id,
            updated_at=datetime.now(UTC),
        )
        self.exchange_store.replace(delivered)
        return CharacterInteractionResult(delivered, result)

    async def broadcast_message(
        self,
        *,
        sender_character_id: str,
        content: str,
        recipient_character_ids: Sequence[str] | None = None,
        metadata: dict | None = None,
    ) -> tuple[CharacterInteractionResult, ...]:
        recipients = (
            tuple(recipient_character_ids)
            if recipient_character_ids is not None
            else tuple(character_id for character_id in self.character_ids if character_id != sender_character_id)
        )
        if len(set(recipients)) != len(recipients):
            raise ValueError("broadcast recipients must be unique")
        if sender_character_id in recipients:
            raise ValueError("broadcast sender must not be a recipient")
        results = []
        for recipient in recipients:
            results.append(
                await self.send_message(
                    sender_character_id=sender_character_id,
                    recipient_character_id=recipient,
                    content=content,
                    metadata=metadata,
                )
            )
        return tuple(results)

    def _validate_registry(self) -> None:
        runtime_ids: set[int] = set()
        scope_owners: dict[tuple[str, str], str] = {}
        state_owners: dict[int, str] = {}
        history_owners: dict[int, str] = {}
        for key, runtime in self._runtimes.items():
            if not key.strip():
                raise CharacterIsolationError("character registry key must not be empty")
            if runtime.character.id != key:
                raise CharacterIsolationError(
                    f"registry key {key!r} does not match runtime character {runtime.character.id!r}"
                )
            runtime_identity = id(runtime)
            if runtime_identity in runtime_ids:
                raise CharacterIsolationError("the same CharacterRuntime cannot represent two characters")
            runtime_ids.add(runtime_identity)
            for identity, owners, label in (
                (id(runtime.state), state_owners, "state"),
                (id(runtime.history), history_owners, "history"),
            ):
                previous = owners.get(identity)
                if previous is not None and previous != key:
                    raise CharacterIsolationError(f"characters {previous} and {key} share mutable {label}")
                owners[identity] = key

            if not self.enforce_scope_isolation:
                continue
            scopes = {
                "memory": runtime.memory_scope_id,
                "cognition": runtime.cognition_scope_id,
                "goal": runtime.goal_scope_id,
            }
            for scope_type, scope_id in scopes.items():
                scope_key = (scope_type, scope_id)
                previous = scope_owners.get(scope_key)
                if previous is not None and previous != key:
                    raise CharacterIsolationError(
                        f"characters {previous} and {key} share {scope_type} scope {scope_id!r}"
                    )
                scope_owners[scope_key] = key
