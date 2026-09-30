from __future__ import annotations

import asyncio
import ast
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.distributed import (
    DISTRIBUTED_PROTOCOL_VERSION,
    CompletionDisposition,
    DistributedBrokerConfig,
    DistributedCompletion,
    DistributedQueueFullError,
    DistributedTaskCoordinator,
    DistributedTaskEnvelope,
    DistributedTaskNotFoundError,
    DistributedTaskStatus,
    DistributedWorker,
    DistributedWorkerConfig,
    DuplicateDistributedTaskError,
    InMemoryDistributedTaskBroker,
)
from ai_character_engine.llm.models import Message
from ai_character_engine.production import FailureClass, Idempotency
from ai_character_engine.tasks import (
    TaskOutput,
    TaskPriority,
    TaskProposal,
    TaskRequest,
    TaskSnapshot,
    TaskStateSnapshot,
)
from ai_character_engine.tools.models import ToolCall, ToolResult

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 26, 6, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def snapshot(*, revision: int = 7, character_id: str = "char-a") -> TaskSnapshot:
    return TaskSnapshot(
        revision=revision,
        captured_at=datetime(2026, 9, 26, 5, 59, tzinfo=UTC),
        character_id=character_id,
        character_name="Alice",
        state=TaskStateSnapshot(
            emotion="calm",
            energy=0.8,
            trust=0.7,
            favorability=0.6,
            relationship_stage="familiar",
            custom={"room": "library"},
        ),
        history=(Message("user", "remember this"),),
        memory_scope_id=f"memory:{character_id}",
    )


def request(
    *, task_id: str = "task-1", task_type: str = "memory", priority=TaskPriority.NORMAL,
    payload=None,
) -> TaskRequest:
    return TaskRequest(
        task_type=task_type,
        payload=payload or {"topic": "books"},
        priority=priority,
        source="test",
        id=task_id,
        created_at=datetime(2026, 9, 26, 6, 0, tzinfo=UTC),
    )


def envelope(
    *, task_id: str = "task-1", task_type: str = "memory",
    idempotency=Idempotency.SAFE, priority=TaskPriority.NORMAL, payload=None,
) -> DistributedTaskEnvelope:
    return DistributedTaskEnvelope(
        request=request(task_id=task_id, task_type=task_type, priority=priority, payload=payload),
        snapshot=snapshot(),
        idempotency=idempotency,
        metadata={"origin": "test"},
        submitted_at=datetime(2026, 9, 26, 6, 0, tzinfo=UTC),
    )


def success(lease, *, value=None, proposals=()):
    return DistributedCompletion(
        task_id=lease.task_id,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        fencing_token=lease.fencing_token,
        output=TaskOutput(value=value, proposals=proposals),
    )


def failure(lease, failure_class, message="boom"):
    return DistributedCompletion(
        task_id=lease.task_id,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        fencing_token=lease.fencing_token,
        error=message,
        failure_class=failure_class,
    )


def test_protocol_version_is_explicit_and_stable():
    assert DISTRIBUTED_PROTOCOL_VERSION == 1


def test_envelope_rejects_wrong_protocol_version():
    with pytest.raises(ValueError, match="unsupported distributed protocol"):
        DistributedTaskEnvelope(request=request(), snapshot=snapshot(), protocol_version=99)


def test_envelope_metadata_is_defensively_copied_and_read_only():
    meta = {"nested": {"x": 1}}
    item = DistributedTaskEnvelope(request=request(), snapshot=snapshot(), metadata=meta)
    meta["nested"]["x"] = 2
    assert item.metadata["nested"]["x"] == 1
    with pytest.raises(TypeError):
        item.metadata["x"] = 2


def test_envelope_wire_roundtrip_preserves_snapshot_history_and_tool_data():
    snap = replace(
        snapshot(),
        history=(
            Message("assistant", "calling", tool_calls=(ToolCall("c1", "lookup", {"q": "x"}),)),
            Message("tool", "", tool_result=ToolResult("c1", "lookup", "done")),
        ),
    )
    item = DistributedTaskEnvelope(request=request(), snapshot=snap, idempotency=Idempotency.SAFE)
    restored = DistributedTaskEnvelope.from_dict(item.to_dict())
    assert restored.request == item.request
    assert restored.snapshot == item.snapshot
    assert restored.idempotency is Idempotency.SAFE


def test_envelope_wire_rejects_non_json_payload_before_queueing():
    item = envelope(payload={"bad": object()})
    with pytest.raises(ValueError, match="JSON-safe"):
        item.to_dict()


def test_completion_wire_roundtrip_preserves_proposals():
    proposal = TaskProposal(
        target="memory.add",
        payload={"text": "x"},
        base_revision=7,
        source_task_id="task-1",
        confidence=0.9,
    )
    item = DistributedCompletion(
        task_id="task-1",
        lease_id="lease",
        worker_id="worker",
        fencing_token=2,
        output=TaskOutput(value={"ok": True}, proposals=(proposal,), metadata={"remote": True}),
    )
    restored = DistributedCompletion.from_dict(item.to_dict())
    assert restored.output == item.output
    assert restored.fencing_token == 2


def test_completion_requires_exactly_output_or_error():
    with pytest.raises(ValueError, match="requires output or error"):
        DistributedCompletion("t", "l", "w", 1)
    with pytest.raises(ValueError, match="both output and error"):
        DistributedCompletion("t", "l", "w", 1, output=TaskOutput(), error="x")


@pytest.mark.asyncio
async def test_enqueue_and_snapshot_are_queued():
    broker = InMemoryDistributedTaskBroker()
    assert await broker.enqueue(envelope()) is True
    state = await broker.snapshot("task-1")
    assert state.status is DistributedTaskStatus.QUEUED
    assert state.attempts == 0


@pytest.mark.asyncio
async def test_same_envelope_retransmission_is_idempotent():
    broker = InMemoryDistributedTaskBroker()
    item = envelope()
    assert await broker.enqueue(item) is True
    assert await broker.enqueue(item) is False


@pytest.mark.asyncio
async def test_same_task_id_same_semantics_different_submit_time_is_idempotent():
    broker = InMemoryDistributedTaskBroker()
    first = envelope()
    second = replace(first, submitted_at=first.submitted_at + timedelta(seconds=4))
    assert await broker.enqueue(first)
    assert await broker.enqueue(second) is False


@pytest.mark.asyncio
async def test_same_task_id_different_payload_is_conflict():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(payload={"x": 1}))
    with pytest.raises(DuplicateDistributedTaskError):
        await broker.enqueue(envelope(payload={"x": 2}))


@pytest.mark.asyncio
async def test_bounded_pending_queue_rejects_overload():
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(max_pending_tasks=1))
    await broker.enqueue(envelope(task_id="a"))
    with pytest.raises(DistributedQueueFullError):
        await broker.enqueue(envelope(task_id="b"))


@pytest.mark.asyncio
async def test_lease_filters_by_worker_capability():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(task_type="vision"))
    assert await broker.lease("w", ("memory",)) is None
    lease = await broker.lease("w", ("vision",))
    assert lease is not None and lease.envelope.request.task_type == "vision"


@pytest.mark.asyncio
async def test_priority_is_preserved_across_remote_queue():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(task_id="low", priority=TaskPriority.LOW))
    await broker.enqueue(envelope(task_id="high", priority=TaskPriority.HIGH))
    lease = await broker.lease("w", ("memory",))
    assert lease is not None and lease.task_id == "high"


@pytest.mark.asyncio
async def test_lease_increments_attempt_and_fencing_token():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope())
    one = await broker.lease("w1", ("memory",))
    assert one is not None and (one.attempt, one.fencing_token) == (1, 1)
    clock.advance(2)
    assert await broker.reap_expired() == 1
    two = await broker.lease("w2", ("memory",))
    assert two is not None and (two.attempt, two.fencing_token) == (2, 2)


@pytest.mark.asyncio
async def test_heartbeat_extends_current_lease():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=5), clock=clock)
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    clock.advance(4)
    refreshed = await broker.heartbeat(lease)
    assert refreshed is not None
    assert refreshed.expires_at > lease.expires_at
    clock.advance(2)
    assert await broker.reap_expired() == 0


@pytest.mark.asyncio
async def test_heartbeat_rejects_wrong_worker_or_fencing_token():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    wrong_worker = replace(lease, worker_id="other")
    wrong_token = replace(lease, fencing_token=lease.fencing_token + 1)
    assert await broker.heartbeat(wrong_worker) is None
    assert await broker.heartbeat(wrong_token) is None


@pytest.mark.asyncio
async def test_successful_completion_is_accepted_once():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    receipt = await broker.complete(success(lease, value={"summary": "x"}))
    assert receipt.disposition is CompletionDisposition.ACCEPTED
    assert receipt.status is DistributedTaskStatus.SUCCEEDED
    state = await broker.snapshot("task-1")
    assert state.result.value == {"summary": "x"}


@pytest.mark.asyncio
async def test_duplicate_completion_is_idempotently_detected():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    item = success(lease, value="x")
    assert (await broker.complete(item)).disposition is CompletionDisposition.ACCEPTED
    assert (await broker.complete(item)).disposition is CompletionDisposition.DUPLICATE


@pytest.mark.asyncio
async def test_old_lease_cannot_overwrite_newer_attempt():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope())
    old = await broker.lease("w1", ("memory",))
    clock.advance(2)
    await broker.reap_expired()
    new = await broker.lease("w2", ("memory",))
    stale = await broker.complete(success(old, value="stale"))
    assert stale.disposition is CompletionDisposition.STALE_LEASE
    accepted = await broker.complete(success(new, value="fresh"))
    assert accepted.status is DistributedTaskStatus.SUCCEEDED
    assert (await broker.snapshot("task-1")).result.value == "fresh"


@pytest.mark.asyncio
async def test_safe_transient_failure_requeues():
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(max_attempts=3))
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    lease = await broker.lease("w", ("memory",))
    receipt = await broker.complete(failure(lease, FailureClass.TRANSIENT))
    assert receipt.disposition is CompletionDisposition.REQUEUED
    assert receipt.status is DistributedTaskStatus.QUEUED


@pytest.mark.asyncio
@pytest.mark.parametrize("idempotency", [Idempotency.UNKNOWN, Idempotency.UNSAFE])
async def test_non_safe_transient_failure_requires_review(idempotency):
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(idempotency=idempotency))
    lease = await broker.lease("w", ("memory",))
    receipt = await broker.complete(failure(lease, FailureClass.TRANSIENT))
    assert receipt.status is DistributedTaskStatus.REVIEW_REQUIRED


@pytest.mark.asyncio
async def test_ambiguous_failure_never_auto_retries():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    lease = await broker.lease("w", ("memory",))
    receipt = await broker.complete(failure(lease, FailureClass.AMBIGUOUS))
    assert receipt.status is DistributedTaskStatus.REVIEW_REQUIRED


@pytest.mark.asyncio
async def test_permanent_failure_is_terminal_failed():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    receipt = await broker.complete(failure(lease, FailureClass.PERMANENT))
    assert receipt.status is DistributedTaskStatus.FAILED


@pytest.mark.asyncio
async def test_safe_transient_failure_hits_dead_letter_after_max_attempts():
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(max_attempts=2))
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    first = await broker.lease("w1", ("memory",))
    assert (await broker.complete(failure(first, FailureClass.TRANSIENT))).disposition is CompletionDisposition.REQUEUED
    second = await broker.lease("w2", ("memory",))
    receipt = await broker.complete(failure(second, FailureClass.TRANSIENT))
    assert receipt.status is DistributedTaskStatus.DEAD_LETTER


@pytest.mark.asyncio
async def test_safe_lease_expiry_requeues_worker_loss():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    await broker.lease("dead-worker", ("memory",))
    clock.advance(2)
    assert await broker.reap_expired() == 1
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.QUEUED


@pytest.mark.asyncio
async def test_unknown_lease_expiry_requires_review_not_reexecution():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope(idempotency=Idempotency.UNKNOWN))
    await broker.lease("dead-worker", ("memory",))
    clock.advance(2)
    await broker.reap_expired()
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.REVIEW_REQUIRED


@pytest.mark.asyncio
async def test_safe_repeated_lease_expiry_dead_letters_at_max_attempts():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1, max_attempts=2), clock=clock)
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    await broker.lease("w1", ("memory",))
    clock.advance(2); await broker.reap_expired()
    await broker.lease("w2", ("memory",))
    clock.advance(2); await broker.reap_expired()
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.DEAD_LETTER


@pytest.mark.asyncio
async def test_cancel_invalidates_active_lease_and_late_result():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    assert await broker.cancel("task-1") is True
    receipt = await broker.complete(success(lease, value="late"))
    assert receipt.disposition is CompletionDisposition.REJECTED
    assert receipt.status is DistributedTaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_terminal_task_is_noop():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("w", ("memory",))
    await broker.complete(success(lease))
    assert await broker.cancel("task-1") is False


@pytest.mark.asyncio
async def test_unknown_snapshot_and_completion_are_explicit_errors():
    broker = InMemoryDistributedTaskBroker()
    with pytest.raises(DistributedTaskNotFoundError):
        await broker.snapshot("missing")
    with pytest.raises(DistributedTaskNotFoundError):
        await broker.complete(DistributedCompletion("missing", "l", "w", 1, output=TaskOutput()))


@pytest.mark.asyncio
async def test_lifecycle_events_keep_delivery_provenance():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    lease = await broker.lease("worker-a", ("memory",))
    await broker.complete(success(lease))
    events = await broker.lifecycle_events(task_id="task-1")
    assert [x.status for x in events] == ["queued", "leased", "succeeded"]
    assert events[-1].worker_id == "worker-a"
    assert events[-1].fencing_token == 1


@pytest.mark.asyncio
async def test_worker_executes_existing_task_context_and_returns_non_authoritative_proposal():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())

    def handler(ctx):
        assert ctx.snapshot.revision == 7
        return TaskOutput(proposals=(ctx.proposal("memory.add", {"text": "candidate"}),))

    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": handler})
    assert await worker.run_once() is True
    state = await broker.snapshot("task-1")
    assert state.status is DistributedTaskStatus.SUCCEEDED
    assert state.result.proposals[0].target == "memory.add"
    assert state.result.proposals[0].base_revision == 7


@pytest.mark.asyncio
async def test_worker_returns_false_when_no_matching_work():
    broker = InMemoryDistributedTaskBroker()
    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": lambda ctx: None})
    assert await worker.run_once() is False


@pytest.mark.asyncio
async def test_worker_transient_exception_requeues_safe_task():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))

    def handler(ctx):
        raise ConnectionError("network")

    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": handler})
    await worker.run_once()
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.QUEUED


@pytest.mark.asyncio
async def test_worker_unknown_exception_uses_ambiguous_review_boundary():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))

    def handler(ctx):
        raise RuntimeError("unknown outcome")

    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": handler})
    await worker.run_once()
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.REVIEW_REQUIRED


@pytest.mark.asyncio
async def test_worker_cancellation_abandons_lease_for_broker_recovery():
    clock = Clock()
    broker = InMemoryDistributedTaskBroker(config=DistributedBrokerConfig(lease_ttl_s=1), clock=clock)
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    started = asyncio.Event()

    async def handler(ctx):
        started.set()
        await asyncio.Event().wait()

    worker = DistributedWorker(
        broker, worker_id="w", handlers={"memory": handler},
        config=DistributedWorkerConfig(lease_heartbeat_interval_s=100),
    )
    task = asyncio.create_task(worker.run_once())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.LEASED
    clock.advance(2)
    await broker.reap_expired()
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.QUEUED


@pytest.mark.asyncio
async def test_coordinator_submission_is_transport_facade_only():
    broker = InMemoryDistributedTaskBroker()
    coordinator = DistributedTaskCoordinator(broker)
    item = await coordinator.submit(
        request(), snapshot(), idempotency=Idempotency.SAFE, metadata={"trace": "x"}
    )
    assert item.task_id == "task-1"
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.QUEUED


def test_distributed_package_does_not_import_authority_managers_or_character_runtime():
    package = ROOT / "src" / "ai_character_engine" / "distributed"
    forbidden = (
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.runtime",
        "ai_character_engine.multi_character",
    )
    offenders = []
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        offenders.append((path.name, alias.name))
            if module and module.startswith(forbidden):
                offenders.append((path.name, module))
    assert offenders == []


def test_distributed_package_has_no_cloud_queue_vendor_dependency_tokens():
    package = ROOT / "src" / "ai_character_engine" / "distributed"
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in package.glob("*.py"))
    for token in ("redis", "kafka", "rabbitmq", "sqs", "pubsub", "celery", "ray.io"):
        assert token not in text


def test_public_api_exports_v042_distributed_runtime():
    assert ace.__version__ == "1.0.0"
    assert ace.DISTRIBUTED_PROTOCOL_VERSION == 1
    assert ace.DistributedWorker is DistributedWorker
    assert ace.InMemoryDistributedTaskBroker is InMemoryDistributedTaskBroker


def test_both_package_versions_match_v042():
    import tomllib

    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    vrm = tomllib.loads((ROOT / "packages/renderer-vrm/pyproject.toml").read_text())["project"]["version"]
    assert (core, vrm) == ("1.0.0", "1.0.0")

@pytest.mark.asyncio
async def test_worker_honors_remote_task_timeout_and_requeues_safe_timeout():
    broker = InMemoryDistributedTaskBroker()
    timed = envelope(idempotency=Idempotency.SAFE)
    timed = replace(timed, request=replace(timed.request, timeout_s=0.01))
    await broker.enqueue(timed)

    async def handler(ctx):
        await asyncio.sleep(1)

    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": handler})
    await worker.run_once()
    state = await broker.snapshot("task-1")
    assert state.status is DistributedTaskStatus.QUEUED
    assert "TimeoutError" in (state.error or "")


@pytest.mark.asyncio
async def test_run_forever_processes_work_until_stop_signal():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope())
    stop = asyncio.Event()

    async def handler(ctx):
        stop.set()
        return "done"

    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": handler})
    await asyncio.wait_for(worker.run_forever(poll_interval_s=0.01, stop_event=stop), timeout=1)
    assert (await broker.snapshot("task-1")).status is DistributedTaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_coordinator_wait_returns_only_after_terminal_status():
    broker = InMemoryDistributedTaskBroker()
    coordinator = DistributedTaskCoordinator(broker)
    await coordinator.submit(request(), snapshot(), idempotency=Idempotency.SAFE)
    worker = DistributedWorker(broker, worker_id="w", handlers={"memory": lambda ctx: "ok"})
    runner = asyncio.create_task(worker.run_once())
    state = await coordinator.wait("task-1", timeout_s=1)
    await runner
    assert state.status is DistributedTaskStatus.SUCCEEDED
    assert state.result.value == "ok"


def test_worker_config_rejects_invalid_heartbeat_and_timeout():
    with pytest.raises(ValueError, match="lease_heartbeat"):
        DistributedWorkerConfig(lease_heartbeat_interval_s=0)
    with pytest.raises(ValueError, match="default_task_timeout"):
        DistributedWorkerConfig(default_task_timeout_s=0)


def test_v042_docs_lock_authority_delivery_and_next_scope():
    guide = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    assert "Remote execution is not remote authority" in guide
    assert "At-least-once execution" in guide
    assert "fencing" in guide




def test_v042_offline_example_runs_without_network_or_provider():
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests/scenarios/distributed_workers.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "after first attempt: queued" in proc.stdout
    assert "final status: succeeded" in proc.stdout
    assert "attempts: 2" in proc.stdout
    assert "remote result remains non-authoritative" in proc.stdout

@pytest.mark.asyncio
async def test_duplicate_retryable_failure_completion_is_detected_after_requeue():
    broker = InMemoryDistributedTaskBroker()
    await broker.enqueue(envelope(idempotency=Idempotency.SAFE))
    lease = await broker.lease("w", ("memory",))
    item = failure(lease, FailureClass.TRANSIENT)
    assert (await broker.complete(item)).disposition is CompletionDisposition.REQUEUED
    duplicate = await broker.complete(item)
    assert duplicate.disposition is CompletionDisposition.DUPLICATE
    assert duplicate.status is DistributedTaskStatus.QUEUED
