from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.evaluation import CharacterEvaluator, EvalCase, PersonaSpec
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.runtime.models import CharacterRunResult

from .models import ProductionEvalTelemetry, TraceContext
from .sink import NullObservabilitySink, ObservabilitySink


@dataclass(frozen=True, slots=True)
class ProductionEvalPolicy:
    sample_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be within 0..1")

    def should_sample(self, trace_id: str) -> bool:
        if self.sample_rate <= 0:
            return False
        if self.sample_rate >= 1:
            return True
        bucket = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < self.sample_rate


class ProductionEvalRunner:
    """Sampled read-only persona evaluation for production turns.

    The runner never mutates runtime state/history/memory. LLM-judge adapters may
    be used by the injected evaluator, but callers should usually run expensive
    judges off the latency-critical request path.
    """

    def __init__(
        self,
        *,
        evaluator: CharacterEvaluator,
        persona_resolver: Callable[[str], PersonaSpec],
        policy: ProductionEvalPolicy | None = None,
        sink: ObservabilitySink | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.persona_resolver = persona_resolver
        self.policy = policy or ProductionEvalPolicy()
        self.sink = sink or NullObservabilitySink()

    async def evaluate_turn(
        self,
        *,
        context: TraceContext,
        character: CharacterProfile,
        event: CharacterEvent,
        result: CharacterRunResult,
        history: tuple[str, ...] = (),
    ) -> ProductionEvalTelemetry | None:
        if not self.policy.should_sample(context.trace_id):
            return None
        case = EvalCase(
            case_id=f"prod:{context.trace_id}:{event.id}",
            character=character,
            response=result.text,
            persona=self.persona_resolver(character.id),
            state=result.state_after,
            user_message=event.content,
            history=history,
        )
        evaluated = await self.evaluator.evaluate(case)
        telemetry = ProductionEvalTelemetry(
            trace_id=context.trace_id,
            case_id=case.case_id,
            character_id=character.id,
            passed=evaluated.passed,
            consistency_score=evaluated.consistency_score,
            violation_count=len(evaluated.violations),
            severity=evaluated.severity.value if evaluated.severity else None,
        )
        self.sink.record_eval(telemetry)
        return telemetry
