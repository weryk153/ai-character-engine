import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.cognitive_evaluation import (
    CognitiveEvalDimension,
    CognitiveEvalResult,
    CognitiveEvalTrace,
)
from ai_character_engine.evaluation.models import Severity
from ai_character_engine.production import (
    CircuitBreakerPolicy,
    CircuitState,
    DegradedMode,
    EvaluationGatePolicy,
    FailureClass,
    Idempotency,
    OperationStatus,
    ProductionHardeningConfig,
    ProductionHardeningRuntime,
    ResourceLimits,
    RetryPolicy,
    default_failure_classifier,
    evaluation_gate,
)


def runtime(**kwargs):
    return ProductionHardeningRuntime(
        config=ProductionHardeningConfig(
            retry=kwargs.pop("retry", RetryPolicy(max_attempts=3, base_backoff_s=0, max_backoff_s=1)),
            circuit_breaker=kwargs.pop("circuit", CircuitBreakerPolicy(failure_threshold=3, recovery_timeout_s=5)),
            resources=kwargs.pop("resources", ResourceLimits(max_concurrent_operations=2, max_pending_operations=2, default_timeout_s=0.1)),
            lifecycle_history=kwargs.pop("lifecycle_history", 32),
        ),
        sleep=kwargs.pop("sleep", None),
        **kwargs,
    )


def failed_eval(severity=Severity.HIGH):
    return CognitiveEvalResult(
        case_id="case",
        trace=(
            CognitiveEvalTrace(
                code="x",
                dimension=CognitiveEvalDimension.EVIDENCE_FAITHFULNESS,
                status="failed",
                message="bad",
                severity=severity,
            ),
        ),
    )


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": 0},
    {"base_backoff_s": -1},
    {"max_backoff_s": 0},
    {"base_backoff_s": 2, "max_backoff_s": 1},
])
def test_retry_policy_validation(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_retry_backoff_is_bounded_and_deterministic():
    policy = RetryPolicy(max_attempts=5, base_backoff_s=0.5, max_backoff_s=1.5)
    assert [policy.backoff_s(i) for i in range(1, 5)] == [0.5, 1.0, 1.5, 1.5]


@pytest.mark.parametrize("kwargs", [
    {"failure_threshold": 0}, {"recovery_timeout_s": 0},
])
def test_circuit_policy_validation(kwargs):
    with pytest.raises(ValueError):
        CircuitBreakerPolicy(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_concurrent_operations": 0},
    {"max_pending_operations": -1},
    {"default_timeout_s": 0},
])
def test_resource_limits_validation(kwargs):
    with pytest.raises(ValueError):
        ResourceLimits(**kwargs)


async def test_success_returns_value_and_health():
    engine = runtime()
    result = await engine.run("read", lambda: {"ok": True}, idempotency=Idempotency.SAFE)
    assert result.succeeded and result.value == {"ok": True} and result.attempts == 1
    health = engine.health_snapshot()
    assert health.completed == 1 and health.in_flight == 0 and health.failed == 0


async def test_transient_safe_operation_retries():
    calls = 0
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def op():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("offline")
        return "ok"

    engine = runtime(sleep=fake_sleep)
    result = await engine.run("safe", op, idempotency=Idempotency.SAFE)
    assert result.succeeded and result.attempts == 3 and calls == 3
    assert sleeps == [0, 0]
    assert [x.status for x in engine.lifecycle_events(result.operation_id)] == ["started", "retrying", "retrying", "succeeded"]


async def test_transient_unsafe_operation_never_retries_and_requires_review():
    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        raise ConnectionError("ack lost")

    result = await runtime().run("external-effect", op, idempotency=Idempotency.UNSAFE)
    assert result.status is OperationStatus.REVIEW_REQUIRED
    assert result.requires_review and result.attempts == 1 and calls == 1


async def test_ambiguous_failure_requires_review_even_for_safe_operation():
    result = await runtime().run("unknown", lambda: (_ for _ in ()).throw(RuntimeError("mystery")), idempotency=Idempotency.SAFE)
    assert result.status is OperationStatus.REVIEW_REQUIRED
    assert result.failure_class is FailureClass.AMBIGUOUS


async def test_permanent_failure_is_not_retried():
    result = await runtime().run("bad-input", lambda: (_ for _ in ()).throw(ValueError("bad")), idempotency=Idempotency.SAFE)
    assert result.status is OperationStatus.FAILED and result.attempts == 1
    assert result.failure_class is FailureClass.PERMANENT


async def test_timeout_is_transient_and_safe_retry_is_bounded():
    async def slow():
        await asyncio.sleep(0.02)

    engine = runtime(retry=RetryPolicy(max_attempts=2, base_backoff_s=0, max_backoff_s=1))
    result = await engine.run("slow", slow, idempotency=Idempotency.SAFE, timeout_s=0.001)
    assert result.status is OperationStatus.FAILED
    assert result.failure_class is FailureClass.TRANSIENT and result.attempts == 2


async def test_cancellation_is_never_retried_or_swallowed():
    started = asyncio.Event()

    async def op():
        started.set()
        await asyncio.Event().wait()

    engine = runtime()
    task = asyncio.create_task(engine.run("cancel", op, idempotency=Idempotency.SAFE))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.health_snapshot().in_flight == 0
    assert engine.lifecycle_events()[-1].status == "cancelled"


async def test_circuit_opens_after_threshold_and_rejects_followup():
    engine = runtime(circuit=CircuitBreakerPolicy(failure_threshold=2, recovery_timeout_s=5))

    async def fail():
        raise ConnectionError("down")

    first = await engine.run("a", fail, dependency="llm", idempotency=Idempotency.UNSAFE)
    second = await engine.run("b", fail, dependency="llm", idempotency=Idempotency.UNSAFE)
    third = await engine.run("c", lambda: "should not run", dependency="llm", idempotency=Idempotency.SAFE)
    assert first.requires_review and second.requires_review
    assert engine.circuit_state("llm") is CircuitState.OPEN
    assert third.status is OperationStatus.REJECTED and third.attempts == 0
    assert third.error_message == "circuit_open"


async def test_circuit_half_open_probe_closes_on_success():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    engine = runtime(
        circuit=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_s=5),
        clock=lambda: now[0],
    )
    await engine.run("fail", lambda: (_ for _ in ()).throw(ConnectionError()), dependency="db", idempotency=Idempotency.UNSAFE)
    assert engine.circuit_state("db") is CircuitState.OPEN
    now[0] += timedelta(seconds=5)
    assert engine.circuit_state("db") is CircuitState.HALF_OPEN
    result = await engine.run("probe", lambda: "ok", dependency="db", idempotency=Idempotency.SAFE)
    assert result.succeeded and engine.circuit_state("db") is CircuitState.CLOSED


async def test_half_open_allows_only_one_probe():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    engine = runtime(
        circuit=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_s=1),
        clock=lambda: now[0],
    )
    await engine.run("fail", lambda: (_ for _ in ()).throw(ConnectionError()), dependency="x", idempotency=Idempotency.UNSAFE)
    now[0] += timedelta(seconds=1)
    gate = asyncio.Event()
    started = asyncio.Event()

    async def probe():
        started.set(); await gate.wait(); return "ok"

    t = asyncio.create_task(engine.run("probe1", probe, dependency="x", idempotency=Idempotency.SAFE))
    await started.wait()
    other = await engine.run("probe2", lambda: "no", dependency="x", idempotency=Idempotency.SAFE)
    assert other.status is OperationStatus.REJECTED
    gate.set()
    assert (await t).succeeded


async def test_permanent_dependency_error_does_not_trip_circuit():
    engine = runtime(circuit=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_s=1))
    await engine.run("bad", lambda: (_ for _ in ()).throw(ValueError("request")), dependency="provider", idempotency=Idempotency.SAFE)
    assert engine.circuit_state("provider") is CircuitState.CLOSED


async def test_resource_limit_rejects_beyond_running_plus_pending_capacity():
    engine = runtime(resources=ResourceLimits(max_concurrent_operations=1, max_pending_operations=1, default_timeout_s=1))
    gate = asyncio.Event()
    started = asyncio.Event()

    async def blocked():
        started.set(); await gate.wait(); return "ok"

    t1 = asyncio.create_task(engine.run("one", blocked))
    await started.wait()
    t2 = asyncio.create_task(engine.run("two", blocked))
    await asyncio.sleep(0)
    third = await engine.run("three", lambda: "no")
    assert third.status is OperationStatus.REJECTED
    assert third.failure_class is FailureClass.OVERLOADED
    gate.set()
    await t1; await t2


async def test_zero_pending_still_allows_running_capacity():
    engine = runtime(resources=ResourceLimits(max_concurrent_operations=1, max_pending_operations=0, default_timeout_s=1))
    gate = asyncio.Event(); started = asyncio.Event()
    async def blocked():
        started.set(); await gate.wait(); return 1
    t = asyncio.create_task(engine.run("one", blocked))
    await started.wait()
    rejected = await engine.run("two", lambda: 2)
    assert rejected.status is OperationStatus.REJECTED
    gate.set(); assert (await t).succeeded


async def test_read_only_degraded_mode_rejects_mutating_operation_but_allows_read():
    engine = runtime()
    engine.set_degraded_mode(DegradedMode.READ_ONLY)
    blocked = await engine.run("commit", lambda: "bad", allow_in_read_only=False)
    allowed = await engine.run("evaluate", lambda: "ok", allow_in_read_only=True)
    assert blocked.status is OperationStatus.REJECTED
    assert allowed.succeeded and allowed.degraded_mode is DegradedMode.READ_ONLY


def test_evaluation_gate_pass_keeps_normal_mode():
    result = CognitiveEvalResult(case_id="ok", trace=())
    decision = evaluation_gate(result)
    assert decision.mode is DegradedMode.NORMAL and not decision.should_alert


def test_evaluation_gate_low_quality_is_conservative():
    decision = evaluation_gate(failed_eval(Severity.HIGH), policy=EvaluationGatePolicy(min_quality_score=0.5))
    assert decision.mode is DegradedMode.CONSERVATIVE
    assert decision.should_alert


def test_evaluation_gate_critical_is_read_only():
    decision = evaluation_gate(failed_eval(Severity.CRITICAL))
    assert decision.mode is DegradedMode.READ_ONLY
    assert decision.reason == "critical_cognitive_evaluation_violation"


def test_evaluation_gate_is_decision_only_and_does_not_accept_authority_managers():
    import inspect
    sig = inspect.signature(evaluation_gate)
    assert tuple(sig.parameters) == ("result", "policy")


def test_default_failure_classifier_contract():
    assert default_failure_classifier(ConnectionError()) is FailureClass.TRANSIENT
    assert default_failure_classifier(asyncio.TimeoutError()) is FailureClass.TRANSIENT
    assert default_failure_classifier(ValueError()) is FailureClass.PERMANENT
    assert default_failure_classifier(RuntimeError()) is FailureClass.AMBIGUOUS


def test_lifecycle_history_is_bounded():
    engine = runtime(lifecycle_history=2)
    async def run_many():
        for i in range(3):
            await engine.run(str(i), lambda: i)
    asyncio.run(run_many())
    assert len(engine.lifecycle_events()) == 2


async def test_custom_classifier_can_mark_overload_without_retry():
    engine = runtime(failure_classifier=lambda exc: FailureClass.OVERLOADED)
    result = await engine.run("quota", lambda: (_ for _ in ()).throw(RuntimeError("quota")), idempotency=Idempotency.SAFE)
    assert result.status is OperationStatus.FAILED
    assert result.failure_class is FailureClass.OVERLOADED and result.attempts == 1


def test_production_core_has_no_renderer_host_or_character_specific_imports():
    root = Path("src/ai_character_engine/production")
    text = "\n".join(p.read_text() for p in root.glob("*.py"))
    for forbidden in ("ai_character_engine.avatar", "ai_character_engine.host", "ai_character_engine_vrm"):
        assert forbidden not in text.casefold()


def test_production_runtime_has_no_memory_belief_goal_or_state_write_api():
    public = {x for x in dir(ProductionHardeningRuntime) if not x.startswith("_")}
    forbidden = {"remember", "commit", "write_memory", "write_belief", "write_goal", "patch_state", "repair"}
    assert not public.intersection(forbidden)


def test_v038_public_version_and_exports():
    assert ace.__version__ == "1.0.0"
    assert ace.ProductionHardeningRuntime is ProductionHardeningRuntime
    assert ace.DegradedMode is DegradedMode

async def test_half_open_probe_slot_is_released_if_resource_admission_rejects():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    engine = runtime(
        circuit=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_s=1),
        resources=ResourceLimits(max_concurrent_operations=1, max_pending_operations=0, default_timeout_s=1),
        clock=lambda: now[0],
    )
    await engine.run("fail", lambda: (_ for _ in ()).throw(ConnectionError()), dependency="x", idempotency=Idempotency.UNSAFE)
    now[0] += timedelta(seconds=1)

    gate = asyncio.Event(); started = asyncio.Event()
    async def occupy():
        started.set(); await gate.wait(); return "done"
    occupying = asyncio.create_task(engine.run("occupy", occupy))
    await started.wait()

    rejected = await engine.run("probe-rejected", lambda: "no", dependency="x", idempotency=Idempotency.SAFE)
    assert rejected.status is OperationStatus.REJECTED
    gate.set(); await occupying
    retry_probe = await engine.run("probe-retry", lambda: "ok", dependency="x", idempotency=Idempotency.SAFE)
    assert retry_probe.succeeded
    assert engine.circuit_state("x") is CircuitState.CLOSED
