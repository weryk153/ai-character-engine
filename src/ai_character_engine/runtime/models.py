from __future__ import annotations

from dataclasses import dataclass, field

from ai_character_engine.context.budget import ContextTrace
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory.consolidation import MemoryConsolidationResult
from ai_character_engine.memory.ledger import EventLedgerEntry
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.memory.revision import MemoryRevisionResult
from ai_character_engine.memory.trace import RetrievalTrace
from ai_character_engine.state.models import CharacterStateSnapshot, StatePatch
from ai_character_engine.tools.models import ToolResult


@dataclass(slots=True, frozen=True)
class CharacterRunResult:
    """Result of processing one character event through the action loop."""

    event: CharacterEvent
    response: LLMResponse
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)
    state_updates: tuple[StatePatch, ...] = field(default_factory=tuple)
    state_before: CharacterStateSnapshot | None = None
    state_after: CharacterStateSnapshot | None = None
    retrieved_memories: tuple[RetrievedMemory, ...] = field(default_factory=tuple)
    memory_written: MemoryRecord | None = None
    ledger_entry: EventLedgerEntry | None = None
    memory_consolidation: MemoryConsolidationResult | None = None
    memory_revision: MemoryRevisionResult | None = None
    context_trace: ContextTrace | None = None
    rounds: int = 1
    retrieval_trace: RetrievalTrace | None = None

    @property
    def text(self) -> str:
        return self.response.text
