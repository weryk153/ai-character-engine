from __future__ import annotations

from threading import Lock
from typing import Protocol

from dataclasses import dataclass

from .models import ProductionEvalTelemetry, SpanRecord, TurnTelemetry


@dataclass(frozen=True, slots=True)
class ObservabilitySummary:
    turn_count: int
    span_count: int
    error_span_count: int
    total_input_tokens: int
    total_output_tokens: int
    tool_calls: int
    avg_turn_latency_ms: float
    eval_count: int
    eval_pass_rate: float | None
    total_estimated_cost: float = 0.0
    avg_queue_wait_ms: float | None = None
    avg_ttft_ms: float | None = None
    avg_decode_tokens_per_second: float | None = None


class ObservabilitySink(Protocol):
    def record_span(self, span: SpanRecord) -> None: ...
    def record_turn(self, turn: TurnTelemetry) -> None: ...
    def record_eval(self, result: ProductionEvalTelemetry) -> None: ...


class NullObservabilitySink:
    def record_span(self, span: SpanRecord) -> None:
        return None

    def record_turn(self, turn: TurnTelemetry) -> None:
        return None

    def record_eval(self, result: ProductionEvalTelemetry) -> None:
        return None


class InMemoryObservabilitySink:
    """Thread-safe deterministic sink for tests and small local applications."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self.turns: list[TurnTelemetry] = []
        self.evals: list[ProductionEvalTelemetry] = []
        self._lock = Lock()

    def record_span(self, span: SpanRecord) -> None:
        with self._lock:
            self.spans.append(span)

    def record_turn(self, turn: TurnTelemetry) -> None:
        with self._lock:
            self.turns.append(turn)

    def record_eval(self, result: ProductionEvalTelemetry) -> None:
        with self._lock:
            self.evals.append(result)

    def summary(self) -> ObservabilitySummary:
        with self._lock:
            turns = tuple(self.turns)
            spans = tuple(self.spans)
            evals = tuple(self.evals)
        evaluated = tuple(x for x in evals if x.passed is not None)
        return ObservabilitySummary(
            turn_count=len(turns),
            span_count=len(spans),
            error_span_count=sum(1 for x in spans if x.status == "error"),
            total_input_tokens=sum(x.input_tokens or 0 for x in turns),
            total_output_tokens=sum(x.output_tokens or 0 for x in turns),
            tool_calls=sum(x.tool_count for x in turns),
            avg_turn_latency_ms=(sum(x.duration_ms for x in turns) / len(turns)) if turns else 0.0,
            eval_count=len(evals),
            eval_pass_rate=(sum(1 for x in evaluated if x.passed) / len(evaluated)) if evaluated else None,
            total_estimated_cost=sum(x.estimated_cost or 0.0 for x in turns),
            avg_queue_wait_ms=(
                sum(x.queue_wait_ms for x in turns if x.queue_wait_ms is not None)
                / sum(1 for x in turns if x.queue_wait_ms is not None)
                if any(x.queue_wait_ms is not None for x in turns) else None
            ),
            avg_ttft_ms=(
                sum(x.ttft_ms for x in turns if x.ttft_ms is not None)
                / sum(1 for x in turns if x.ttft_ms is not None)
                if any(x.ttft_ms is not None for x in turns) else None
            ),
            avg_decode_tokens_per_second=(
                sum(x.decode_tokens_per_second for x in turns if x.decode_tokens_per_second is not None)
                / sum(1 for x in turns if x.decode_tokens_per_second is not None)
                if any(x.decode_tokens_per_second is not None for x in turns) else None
            ),
        )
