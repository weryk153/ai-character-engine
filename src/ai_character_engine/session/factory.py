from __future__ import annotations

from collections.abc import Callable, Mapping

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.llm.base import LLMClient
from ai_character_engine.memory.manager import MemoryManager
from ai_character_engine.long_term_cognition import LongTermCognitionManager
from ai_character_engine.goals import GoalManager
from ai_character_engine.observability import Tracer
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from ai_character_engine.state.models import CharacterState
from ai_character_engine.state.policy import CharacterStatePolicy
from ai_character_engine.tools.executor import ToolExecutor
from ai_character_engine.tools.registry import ToolRegistry

from .manager import SessionManager
from .models import SessionRecord
from .runtime import ManagedCharacterSession, state_from_snapshot
from .store import InMemoryRelationshipStore, RelationshipStore


class CharacterRuntimeFactory:
    """Creates/restores runtimes while keeping application dependencies injectable."""

    def __init__(
        self,
        *,
        characters: Mapping[str, CharacterProfile],
        llm_factory: Callable[[SessionRecord], LLMClient],
        session_manager: SessionManager,
        memory_manager: MemoryManager | None = None,
        long_term_cognition: LongTermCognitionManager | None = None,
        goal_manager: GoalManager | None = None,
        relationship_store: RelationshipStore | None = None,
        context_builder_factory: Callable[[], ContextBuilder] | None = None,
        tool_registry_factory: Callable[[], ToolRegistry] | None = None,
        tool_executor_factory: Callable[[ToolRegistry], ToolExecutor] | None = None,
        state_policy_factory: Callable[[], CharacterStatePolicy] | None = None,
        max_history_messages: int = 20,
        max_tool_rounds: int = 5,
        retrieval_limit: int = 5,
        tracer: Tracer | None = None,
    ) -> None:
        self.characters = dict(characters)
        self.llm_factory = llm_factory
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.long_term_cognition = long_term_cognition
        self.goal_manager = goal_manager
        self.relationship_store = relationship_store or InMemoryRelationshipStore()
        self.context_builder_factory = context_builder_factory or ContextBuilder
        self.tool_registry_factory = tool_registry_factory or ToolRegistry
        self.tool_executor_factory = tool_executor_factory or ToolExecutor
        self.state_policy_factory = state_policy_factory
        self.max_history_messages = max_history_messages
        self.max_tool_rounds = max_tool_rounds
        self.retrieval_limit = retrieval_limit
        self.tracer = tracer or Tracer()

    def create(
        self,
        *,
        user_id: str,
        character_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
        ttl_seconds: float | None = None,
    ) -> ManagedCharacterSession:
        record = self.session_manager.create(
            user_id=user_id,
            character_id=character_id,
            session_id=session_id,
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )
        managed = self._build(record)
        # Persist the initial snapshot so process restarts can restore an empty
        # session even before the first user event arrives.
        from .runtime import snapshot_runtime
        managed.record = self.session_manager.touch(
            record.id,
            snapshot=snapshot_runtime(managed.runtime),
        )
        return managed

    def restore(self, session_id: str) -> ManagedCharacterSession:
        return self._build(self.session_manager.require(session_id))

    def _build(self, record: SessionRecord) -> ManagedCharacterSession:
        character = self.characters.get(record.character_id)
        if character is None:
            raise KeyError(f"unknown character: {record.character_id}")

        snapshot = record.runtime_snapshot
        state = (
            state_from_snapshot(snapshot.state)
            if snapshot is not None
            else CharacterState()
        )
        relationship = self.relationship_store.get(
            user_id=record.user_id,
            character_id=record.character_id,
        )
        if relationship is not None:
            state.trust = relationship.trust
            state.favorability = relationship.favorability
            state.relationship_stage = relationship.relationship_stage

        registry = self.tool_registry_factory()
        runtime = CharacterRuntime(
            character=character,
            llm=self.llm_factory(record),
            context_builder=self.context_builder_factory(),
            tool_registry=registry,
            tool_executor=self.tool_executor_factory(registry),
            state=state,
            state_policy=(self.state_policy_factory() if self.state_policy_factory else None),
            memory_manager=self.memory_manager,
            long_term_cognition=self.long_term_cognition,
            cognition_scope_id=record.scopes.memory_scope_id,
            goal_manager=self.goal_manager,
            goal_scope_id=record.scopes.memory_scope_id,
            max_history_messages=self.max_history_messages,
            max_tool_rounds=self.max_tool_rounds,
            retrieval_limit=self.retrieval_limit,
            memory_scope_id=record.scopes.memory_scope_id,
            tracer=self.tracer,
        )
        if snapshot is not None:
            runtime.history = list(snapshot.history)

        return ManagedCharacterSession(
            record=record,
            runtime=runtime,
            session_manager=self.session_manager,
            relationship_store=self.relationship_store,
        )
