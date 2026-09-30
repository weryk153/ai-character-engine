from __future__ import annotations

import asyncio
import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.distributed import (
    CompletionDisposition,
    DistributedBrokerConfig,
    DistributedCompletion,
    DistributedTaskEnvelope,
    DistributedTaskStatus,
    InMemoryDistributedTaskBroker,
)
from ai_character_engine.multi_character import FairCharacterScheduler, FairSchedulerConfig
from ai_character_engine.production import (
    Idempotency,
    ProductionHardeningConfig,
    ProductionHardeningRuntime,
    ResourceLimits,
    RetryPolicy,
)
from ai_character_engine.stability import (
    AsyncioTaskProbe,
    CallableProbe,
    CompositeProbe,
    FailureInjector,
    FailureRule,
    InjectedFailure,
    MetricInvariant,
    SoakConfig,
    SoakHarness,
    SoakReport,
    SoakSample,
    SoakStatus,
    injected_failure,
)
from ai_character_engine.tasks import TaskOutput, TaskRequest, TaskSnapshot, TaskStateSnapshot

ROOT = Path(__file__).resolve().parents[1]
STABILITY = ROOT / "src" / "ai_character_engine" / "stability"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 26, 8, 0, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def snapshot() -> TaskSnapshot:
    return TaskSnapshot(
        revision=1,
        captured_at=datetime(2026, 9, 26, 8, 0, tzinfo=UTC),
        character_id="char",
        character_name="Char",
        state=TaskStateSnapshot("calm", 1.0, 0.5, 0.5, "known", {}),
        history=(),
        memory_scope_id="memory:char",
    )


def envelope(task_id: str, *, idempotency=Idempotency.SAFE) -> DistributedTaskEnvelope:
    return DistributedTaskEnvelope(
        request=TaskRequest(task_type="soak", payload={"id": task_id}, source="v044", id=task_id),
        snapshot=snapshot(),
        idempotency=idempotency,
    )


def completion(lease, value="ok") -> DistributedCompletion:
    return DistributedCompletion(
        task_id=lease.task_id,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        fencing_token=lease.fencing_token,
        output=TaskOutput(value=value),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"concurrency": 0},
        {"sample_every": 0},
        {"max_failures": -1},
        {"max_recorded_failures": -1},
    ],
)
def test_soak_config_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        SoakConfig(**kwargs)


def test_metric_invariant_requires_limit_and_non_negative_budget():
    with pytest.raises(ValueError):
        MetricInvariant("x")
    with pytest.raises(ValueError):
        MetricInvariant("x", max_growth=-1)
    with pytest.raises(ValueError):
        MetricInvariant("x", max_final=-1)


def test_soak_sample_is_read_only_and_validates_counters():
    metrics = {"x": 1}
    sample = SoakSample(1, 1, 0, 0, metrics)
    metrics["x"] = 99
    assert sample.metrics["x"] == 1.0
    with pytest.raises(TypeError):
        sample.metrics["x"] = 2
    with pytest.raises(ValueError):
        SoakSample(1, 1, 1, 0)


def test_failure_rule_rejects_non_positive_call_numbers():
    with pytest.raises(ValueError):
        FailureRule((0,), injected_failure())


def test_failure_injector_is_deterministic_and_resettable():
    injector = FailureInjector(FailureRule((2, 4), injected_failure("boom")))
    assert injector.checkpoint() == 1
    with pytest.raises(InjectedFailure, match="call 2"):
        injector.checkpoint()
    assert injector.checkpoint() == 3
    with pytest.raises(InjectedFailure, match="call 4"):
        injector.checkpoint()
    injector.reset()
    assert injector.calls == 0 and injector.checkpoint() == 1


def test_composite_probe_rejects_duplicate_metric_names():
    probe = CompositeProbe(CallableProbe(lambda: {"x": 1}), CallableProbe(lambda: {"x": 2}))
    with pytest.raises(ValueError, match="duplicate"):
        probe.sample()


@pytest.mark.asyncio
async def test_successful_soak_is_bounded_and_reports_quiescent_task_count():
    max_active = 0
    active = 0

    async def operation(index: int):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return index

    report = await SoakHarness(
        config=SoakConfig(iterations=25, concurrency=4, sample_every=5),
        invariants=(MetricInvariant("asyncio.pending_tasks", max_growth=0),),
    ).run(operation, metadata={"case": "bounded"})
    assert report.passed
    assert report.iterations == report.succeeded == 25
    assert report.failed == report.cancelled == 0
    assert max_active <= 4
    assert report.samples[0].metrics["asyncio.pending_tasks"] == report.samples[-1].metrics["asyncio.pending_tasks"]
    assert report.metadata["case"] == "bounded"


@pytest.mark.asyncio
async def test_failure_budget_allows_expected_injected_failures():
    injector = FailureInjector(FailureRule((2, 5), injected_failure()))

    async def operation(_):
        injector.checkpoint()

    report = await SoakHarness(
        config=SoakConfig(iterations=6, max_failures=2, sample_every=2),
    ).run(operation)
    assert report.passed
    assert report.failed == 2 and report.succeeded == 4


@pytest.mark.asyncio
async def test_failure_budget_exceeded_is_a_failed_report():
    async def operation(_):
        raise ConnectionError("offline")

    report = await SoakHarness(config=SoakConfig(iterations=3, max_failures=1)).run(operation)
    assert not report.passed
    assert report.status is SoakStatus.FAILED
    assert {item.code for item in report.violations} == {"failure_budget_exceeded"}


@pytest.mark.asyncio
async def test_recorded_failures_are_bounded_even_when_many_iterations_fail():
    async def operation(_):
        raise RuntimeError("x")

    report = await SoakHarness(
        config=SoakConfig(iterations=20, max_failures=20, max_recorded_failures=3),
    ).run(operation)
    assert report.passed and report.failed == 20
    assert len(report.failures) == 3


@pytest.mark.asyncio
async def test_fail_fast_stops_after_failure_budget_is_exceeded():
    async def operation(_):
        raise RuntimeError("x")

    report = await SoakHarness(
        config=SoakConfig(iterations=50, concurrency=2, max_failures=0, fail_fast=True),
    ).run(operation)
    assert not report.passed
    assert report.iterations == 2
    assert "fail_fast" in {item.code for item in report.violations}


@pytest.mark.asyncio
async def test_cancellation_is_observed_not_swallowed():
    async def operation(index: int):
        if index == 1:
            raise asyncio.CancelledError()
        await asyncio.sleep(0)

    report = await SoakHarness(config=SoakConfig(iterations=3, max_failures=1)).run(operation)
    assert report.passed
    assert report.cancelled == 1
    assert report.failures[0].cancelled


@pytest.mark.asyncio
async def test_metric_growth_budget_detects_leak_signal():
    values = iter((1, 1, 3, 3))
    probe = CallableProbe(lambda: {"leak": next(values, 3)})
    report = await SoakHarness(
        config=SoakConfig(iterations=1),
        probe=probe,
        invariants=(MetricInvariant("leak", max_growth=1),),
    ).run(lambda _: None)
    assert not report.passed
    assert report.violations[0].code == "metric_growth_exceeded"


@pytest.mark.asyncio
async def test_metric_final_budget_detects_non_quiescent_state():
    report = await SoakHarness(
        config=SoakConfig(iterations=1),
        probe=CallableProbe(lambda: {"pending": 2}),
        invariants=(MetricInvariant("pending", max_final=0),),
    ).run(lambda _: None)
    assert not report.passed
    assert report.violations[0].code == "metric_final_exceeded"


@pytest.mark.asyncio
async def test_missing_metric_is_fail_closed():
    report = await SoakHarness(
        config=SoakConfig(iterations=1),
        probe=CallableProbe(lambda: {"other": 0}),
        invariants=(MetricInvariant("expected", max_growth=0),),
    ).run(lambda _: None)
    assert not report.passed
    assert report.violations[0].code == "metric_missing"


@pytest.mark.asyncio
async def test_asyncio_probe_can_detect_and_then_cleanup_background_task_leak():
    stop = asyncio.Event()
    leaked: list[asyncio.Task] = []

    async def operation(_):
        leaked.append(asyncio.create_task(stop.wait()))

    report = await SoakHarness(
        config=SoakConfig(iterations=1),
        probe=AsyncioTaskProbe(),
        invariants=(MetricInvariant("asyncio.pending_tasks", max_growth=0),),
    ).run(operation)
    try:
        assert not report.passed
        assert any(item.code == "metric_growth_exceeded" for item in report.violations)
    finally:
        stop.set()
        await asyncio.gather(*leaked)


@pytest.mark.asyncio
async def test_production_runtime_soak_returns_resource_counters_to_zero_after_failures():
    runtime = ProductionHardeningRuntime(
        config=ProductionHardeningConfig(
            retry=RetryPolicy(max_attempts=1, base_backoff_s=0, max_backoff_s=1),
            resources=ResourceLimits(max_concurrent_operations=3, max_pending_operations=3, default_timeout_s=0.1),
            lifecycle_history=16,
        )
    )

    async def operation(index: int):
        async def inner():
            if index % 4 == 0:
                raise ValueError("bad input")
            await asyncio.sleep(0)
            return index
        return await runtime.run("soak", inner, idempotency=Idempotency.SAFE)

    report = await SoakHarness(config=SoakConfig(iterations=40, concurrency=3)).run(operation)
    assert report.passed
    health = runtime.health_snapshot()
    assert health.in_flight == health.waiting == 0
    assert health.completed + health.failed == 40
    assert health.lifecycle_events <= 16


@pytest.mark.asyncio
async def test_production_runtime_cancellation_storm_does_not_leak_capacity():
    runtime = ProductionHardeningRuntime(
        config=ProductionHardeningConfig(resources=ResourceLimits(4, 8, 1), lifecycle_history=32)
    )
    started = asyncio.Event()

    async def blocked():
        started.set()
        await asyncio.Event().wait()

    tasks = [asyncio.create_task(runtime.run("cancel", blocked, idempotency=Idempotency.SAFE)) for _ in range(4)]
    await started.wait()
    await asyncio.sleep(0)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    health = runtime.health_snapshot()
    assert health.in_flight == 0 and health.waiting == 0


@pytest.mark.asyncio
async def test_fair_character_scheduler_cancellation_storm_returns_to_empty_snapshot():
    scheduler = FairCharacterScheduler(FairSchedulerConfig(2, 10, 20))
    gate = asyncio.Event()

    async def blocked():
        await gate.wait()

    tasks = [asyncio.create_task(scheduler.run("a" if i % 2 == 0 else "b", blocked)) for i in range(12)]
    await asyncio.sleep(0)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    state = scheduler.snapshot()
    assert state.running_character_ids == () and state.total_pending == 0


@pytest.mark.asyncio
async def test_distributed_safe_worker_loss_requeues_and_accepts_new_lease_once():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1, max_attempts=3), clock=clock)
    await broker.enqueue(envelope("safe"))
    old = await broker.lease("worker-a", ("soak",))
    assert old is not None
    clock.advance(2)
    assert await broker.reap_expired() == 1
    new = await broker.lease("worker-b", ("soak",))
    assert new is not None and new.fencing_token > old.fencing_token
    receipt = await broker.complete(completion(new))
    assert receipt.disposition is CompletionDisposition.ACCEPTED
    assert (await broker.snapshot("safe")).status is DistributedTaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_distributed_zombie_completion_is_fenced_after_requeue():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope("zombie"))
    old = await broker.lease("worker-a", ("soak",))
    assert old is not None
    clock.advance(2)
    await broker.reap_expired()
    new = await broker.lease("worker-b", ("soak",))
    assert new is not None
    stale = await broker.complete(completion(old, "stale"))
    accepted = await broker.complete(completion(new, "fresh"))
    assert stale.disposition is CompletionDisposition.STALE_LEASE
    assert accepted.disposition is CompletionDisposition.ACCEPTED
    assert (await broker.snapshot("zombie")).result.value == "fresh"


@pytest.mark.asyncio
async def test_distributed_unsafe_worker_loss_goes_to_review_not_replay():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope("unsafe", idempotency=Idempotency.UNSAFE))
    assert await broker.lease("worker-a", ("soak",)) is not None
    clock.advance(2)
    await broker.reap_expired()
    state = await broker.snapshot("unsafe")
    assert state.status is DistributedTaskStatus.REVIEW_REQUIRED
    assert await broker.lease("worker-b", ("soak",)) is None


@pytest.mark.asyncio
async def test_distributed_repeated_safe_lease_loss_is_bounded_by_dead_letter():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1, max_attempts=2), clock=clock)
    await broker.enqueue(envelope("dead"))
    first = await broker.lease("a", ("soak",))
    assert first is not None
    clock.advance(2)
    await broker.reap_expired()
    second = await broker.lease("b", ("soak",))
    assert second is not None
    clock.advance(2)
    await broker.reap_expired()
    assert (await broker.snapshot("dead")).status is DistributedTaskStatus.DEAD_LETTER


def test_stability_package_has_no_cognition_world_or_renderer_authority_imports():
    forbidden = (
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
        "ai_character_engine_vrm",
    )
    offenders = []
    for path in STABILITY.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        offenders.append((path.name, alias.name))
            if module.startswith(forbidden):
                offenders.append((path.name, module))
    assert offenders == []


def test_stability_core_has_no_product_renderer_or_vendor_vocabulary():
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in STABILITY.rglob("*.py"))
    for token in ("vrm", "live2d", "redis", "kafka", "sqs", "openai", "anthropic"):
        assert token not in text


def test_v044_version_and_public_stability_exports():
    assert ace.__version__ == "1.0.0"
    for name in (
        "SoakHarness",
        "SoakConfig",
        "SoakReport",
        "MetricInvariant",
        "FailureInjector",
        "FailureRule",
        "AsyncioTaskProbe",
    ):
        assert name in ace.__all__
        assert getattr(ace, name) is not None


def test_soak_report_to_dict_is_json_ready_and_does_not_expose_mapping_proxy():
    report = SoakReport(
        SoakStatus.PASSED,
        1,
        1,
        0,
        0,
        (SoakSample(0, 0, 0, 0, {"x": 0}), SoakSample(1, 1, 0, 0, {"x": 0})),
        (),
        (),
        {"suite": "v044"},
    )
    payload = report.to_dict()
    assert payload["metadata"] == {"suite": "v044"}
    assert payload["samples"][0]["metrics"] == {"x": 0.0}


def test_python_runtime_probe_exposes_gc_metric_without_forcing_memory_tracing():
    from ai_character_engine.stability import PythonRuntimeProbe

    probe = PythonRuntimeProbe(trace_memory=False)
    metrics = probe.sample()
    assert metrics["python.gc_objects"] >= 0
    assert "python.traced_current_bytes" not in metrics


def test_soak_report_metadata_is_defensively_copied_and_read_only():
    metadata = {"suite": "v044"}
    report = SoakReport(SoakStatus.PASSED, 0, 0, 0, 0, (), (), (), metadata)
    metadata["suite"] = "changed"
    assert report.metadata["suite"] == "v044"
    with pytest.raises(TypeError):
        report.metadata["x"] = 1


def test_stability_cli_smoke_passes_and_emits_json(capsys):
    from ai_character_engine.stability.__main__ import main

    assert main(["--iterations", "8", "--concurrency", "2", "--fail-on-violations"]) == 0
    output = capsys.readouterr().out
    assert '"status": "passed"' in output
    assert '"iterations": 8' in output


def test_stability_cli_can_write_machine_readable_report(tmp_path):
    import json
    from ai_character_engine.stability.__main__ import main

    path = tmp_path / "soak.json"
    assert main(["--iterations", "5", "--output", str(path), "--fail-on-violations"]) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["metadata"]["scenario"] == "offline_noop"
