from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from ai_character_engine.cognitive_evaluation import CognitiveEvalResult

from .models import (
    CircuitState,
    DegradedMode,
    EvaluationGateDecision,
    EvaluationGatePolicy,
    FailureClass,
    Idempotency,
    OperationLifecycleEvent,
    OperationResult,
    OperationStatus,
    ProductionHardeningConfig,
    ProductionHealthSnapshot,
)


FailureClassifier = Callable[[BaseException], FailureClass]
LifecycleSink = Callable[[OperationLifecycleEvent], None]


def default_failure_classifier(exc: BaseException) -> FailureClass:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return FailureClass.TRANSIENT
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return FailureClass.PERMANENT
    return FailureClass.AMBIGUOUS


@dataclass(slots=True)
class _Circuit:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: datetime | None = None
    probe_in_flight: bool = False


class ProductionHardeningRuntime:
    """Provider/host-neutral reliability envelope for bounded engine operations.

    The runtime deliberately has no cognition write APIs. It may retry safe work,
    enforce time/resource bounds, open dependency circuits, emit lifecycle telemetry,
    and enter degraded modes. Callers retain all authoritative mutation responsibility.
    """

    def __init__(
        self,
        *,
        config: ProductionHardeningConfig | None = None,
        failure_classifier: FailureClassifier | None = None,
        on_lifecycle: LifecycleSink | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config or ProductionHardeningConfig()
        self._classify = failure_classifier or default_failure_classifier
        self._on_lifecycle = on_lifecycle
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep
        self._capacity = asyncio.Semaphore(self.config.resources.max_concurrent_operations)
        self._admission_lock = asyncio.Lock()
        self._in_flight = 0
        self._waiting = 0
        self._admitted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._review_required = 0
        self._mode = DegradedMode.NORMAL
        self._circuits: dict[str, _Circuit] = {}
        self._events: deque[OperationLifecycleEvent] = deque(maxlen=self.config.lifecycle_history)

    @property
    def degraded_mode(self) -> DegradedMode:
        return self._mode

    def set_degraded_mode(self, mode: DegradedMode) -> None:
        self._mode = DegradedMode(mode)

    def lifecycle_events(self, operation_id: str | None = None) -> tuple[OperationLifecycleEvent, ...]:
        if operation_id is None:
            return tuple(self._events)
        return tuple(x for x in self._events if x.operation_id == operation_id)

    def circuit_state(self, dependency: str) -> CircuitState:
        circuit = self._circuits.get(dependency)
        if circuit is None:
            return CircuitState.CLOSED
        self._refresh_circuit(dependency, circuit)
        return circuit.state

    def reset_circuit(self, dependency: str) -> None:
        self._circuits.pop(dependency, None)

    def health_snapshot(self) -> ProductionHealthSnapshot:
        open_circuits = tuple(sorted(k for k in self._circuits if self.circuit_state(k) is CircuitState.OPEN))
        return ProductionHealthSnapshot(
            degraded_mode=self._mode,
            in_flight=self._in_flight,
            waiting=self._waiting,
            completed=self._completed,
            failed=self._failed,
            rejected=self._rejected,
            review_required=self._review_required,
            open_circuits=open_circuits,
            lifecycle_events=len(self._events),
        )

    async def run(
        self,
        operation_name: str,
        operation: Callable[[], Any | Awaitable[Any]],
        *,
        dependency: str | None = None,
        idempotency: Idempotency = Idempotency.UNKNOWN,
        timeout_s: float | None = None,
        allow_in_read_only: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> OperationResult:
        name = operation_name.strip()
        if not name:
            raise ValueError("operation_name must not be empty")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        idempotency = Idempotency(idempotency)
        op_id = uuid4().hex
        timeout = timeout_s or self.config.resources.default_timeout_s

        if self._mode is DegradedMode.READ_ONLY and not allow_in_read_only:
            return self._reject(op_id, name, dependency, "read_only_degraded_mode", metadata)
        if dependency and not self._admit_circuit(dependency):
            return self._reject(op_id, name, dependency, "circuit_open", metadata)
        if not await self._reserve_wait_slot():
            if dependency:
                circuit = self._circuits.get(dependency)
                if circuit and circuit.state is CircuitState.HALF_OPEN:
                    circuit.probe_in_flight = False
            return self._reject(op_id, name, dependency, "resource_queue_full", metadata)

        acquired = False
        try:
            await self._capacity.acquire()
            acquired = True
            self._waiting -= 1
            self._in_flight += 1
            self._emit(op_id, name, "started", 0, dependency=dependency, metadata=metadata)
            attempt = 0
            while True:
                attempt += 1
                try:
                    value = operation()
                    if inspect.isawaitable(value):
                        value = await asyncio.wait_for(value, timeout=timeout)
                    self._record_dependency_success(dependency)
                    self._completed += 1
                    self._emit(op_id, name, "succeeded", attempt, dependency=dependency, metadata=metadata)
                    return OperationResult(
                        operation_id=op_id,
                        operation_name=name,
                        status=OperationStatus.SUCCEEDED,
                        attempts=attempt,
                        value=value,
                        degraded_mode=self._mode,
                        dependency=dependency,
                    )
                except asyncio.CancelledError:
                    # Cancellation is control flow; never classify/retry it.
                    self._emit(op_id, name, "cancelled", attempt, dependency=dependency, metadata=metadata)
                    raise
                except BaseException as exc:
                    failure = FailureClass(self._classify(exc))
                    self._record_dependency_failure(dependency, failure)
                    circuit_open = bool(dependency and self.circuit_state(dependency) is CircuitState.OPEN)
                    retry = (
                        failure is FailureClass.TRANSIENT
                        and idempotency is Idempotency.SAFE
                        and attempt < self.config.retry.max_attempts
                        and not circuit_open
                    )
                    self._emit(
                        op_id, name, "retrying" if retry else "failed", attempt,
                        failure_class=failure, error_type=type(exc).__name__,
                        dependency=dependency, metadata=metadata,
                    )
                    if retry:
                        await self._sleep(self.config.retry.backoff_s(attempt))
                        continue
                    requires_review = failure is FailureClass.AMBIGUOUS or (
                        failure is FailureClass.TRANSIENT and idempotency is not Idempotency.SAFE
                    )
                    if requires_review:
                        self._review_required += 1
                        status = OperationStatus.REVIEW_REQUIRED
                    else:
                        self._failed += 1
                        status = OperationStatus.FAILED
                    return OperationResult(
                        operation_id=op_id,
                        operation_name=name,
                        status=status,
                        attempts=attempt,
                        failure_class=failure,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        retryable=retry,
                        requires_review=requires_review,
                        degraded_mode=self._mode,
                        dependency=dependency,
                    )
        finally:
            async with self._admission_lock:
                self._admitted -= 1
                if not acquired:
                    self._waiting -= 1
            if acquired:
                self._in_flight -= 1
                self._capacity.release()
            # A HALF_OPEN dependency gets one probe. If cancellation interrupted it,
            # release the probe without closing the circuit.
            if dependency:
                circuit = self._circuits.get(dependency)
                if circuit and circuit.state is CircuitState.HALF_OPEN and circuit.probe_in_flight:
                    circuit.probe_in_flight = False

    async def _reserve_wait_slot(self) -> bool:
        async with self._admission_lock:
            capacity = (
                self.config.resources.max_concurrent_operations
                + self.config.resources.max_pending_operations
            )
            if self._admitted >= capacity:
                return False
            self._admitted += 1
            self._waiting += 1
            return True

    def _reject(
        self,
        op_id: str,
        name: str,
        dependency: str | None,
        reason: str,
        metadata: dict[str, Any] | None,
    ) -> OperationResult:
        self._rejected += 1
        self._emit(op_id, name, "rejected", 0, dependency=dependency, metadata={**(metadata or {}), "reason": reason})
        return OperationResult(
            operation_id=op_id,
            operation_name=name,
            status=OperationStatus.REJECTED,
            attempts=0,
            failure_class=FailureClass.OVERLOADED if reason == "resource_queue_full" else None,
            error_message=reason,
            degraded_mode=self._mode,
            dependency=dependency,
        )

    def _refresh_circuit(self, dependency: str, circuit: _Circuit) -> None:
        if circuit.state is not CircuitState.OPEN or circuit.opened_at is None:
            return
        elapsed = (self._clock() - circuit.opened_at).total_seconds()
        if elapsed >= self.config.circuit_breaker.recovery_timeout_s:
            circuit.state = CircuitState.HALF_OPEN
            circuit.probe_in_flight = False

    def _admit_circuit(self, dependency: str) -> bool:
        circuit = self._circuits.setdefault(dependency, _Circuit())
        self._refresh_circuit(dependency, circuit)
        if circuit.state is CircuitState.OPEN:
            return False
        if circuit.state is CircuitState.HALF_OPEN:
            if circuit.probe_in_flight:
                return False
            circuit.probe_in_flight = True
        return True

    def _record_dependency_success(self, dependency: str | None) -> None:
        if dependency is None:
            return
        circuit = self._circuits.setdefault(dependency, _Circuit())
        circuit.state = CircuitState.CLOSED
        circuit.consecutive_failures = 0
        circuit.opened_at = None
        circuit.probe_in_flight = False

    def _record_dependency_failure(self, dependency: str | None, failure: FailureClass) -> None:
        if dependency is None or failure is FailureClass.PERMANENT:
            return
        circuit = self._circuits.setdefault(dependency, _Circuit())
        circuit.consecutive_failures += 1
        if circuit.state is CircuitState.HALF_OPEN or circuit.consecutive_failures >= self.config.circuit_breaker.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = self._clock()
            circuit.probe_in_flight = False

    def _emit(
        self,
        op_id: str,
        name: str,
        status: str,
        attempt: int,
        *,
        failure_class: FailureClass | None = None,
        error_type: str | None = None,
        dependency: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = OperationLifecycleEvent(
            operation_id=op_id,
            operation_name=name,
            status=status,
            attempt=attempt,
            at=self._clock(),
            failure_class=failure_class,
            error_type=error_type,
            dependency=dependency,
            degraded_mode=self._mode,
            metadata=metadata or {},
        )
        self._events.append(event)
        if self._on_lifecycle is not None:
            self._on_lifecycle(event)


def evaluation_gate(
    result: CognitiveEvalResult,
    *,
    policy: EvaluationGatePolicy | None = None,
) -> EvaluationGateDecision:
    """Translate a read-only v0.37 evaluation into an operational mode suggestion.

    This function performs no mutation and intentionally accepts only the public
    evaluation result, not Memory/Belief/Goal managers.
    """
    policy = policy or EvaluationGatePolicy()
    violations = result.violations
    severe = any(x.severity is not None and x.severity.value == "critical" for x in violations)
    if severe:
        return EvaluationGateDecision(
            mode=policy.severe_violation_mode,
            reason="critical_cognitive_evaluation_violation",
            quality_score=result.quality_score,
            violation_count=len(violations),
            should_alert=True,
        )
    if result.quality_score is not None and result.quality_score < policy.min_quality_score:
        return EvaluationGateDecision(
            mode=policy.low_quality_mode,
            reason="cognitive_quality_below_threshold",
            quality_score=result.quality_score,
            violation_count=len(violations),
            should_alert=True,
        )
    return EvaluationGateDecision(
        mode=DegradedMode.NORMAL,
        reason="cognitive_quality_within_policy",
        quality_score=result.quality_score,
        violation_count=len(violations),
        should_alert=False,
    )
