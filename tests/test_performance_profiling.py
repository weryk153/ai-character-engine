from __future__ import annotations

from ai_character_engine._version import VERSION
import ast
import asyncio
import json
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder
from ai_character_engine.performance import (
    PERFORMANCE_REPORT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkEnvironment,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSample,
    PerformanceBudget,
    PerformanceStatus,
    capture_benchmark_environment,
    evaluate_performance_budget,
    load_benchmark_result,
    save_benchmark_result,
)
from ai_character_engine.performance.__main__ import main as performance_main

ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE = ROOT / "src" / "ai_character_engine" / "performance"


def env(profile_id: str = "ci-linux") -> BenchmarkEnvironment:
    return BenchmarkEnvironment(profile_id, "3.11", "CPython", "Linux", "x86_64", 4, {"runner": "test"})


def metrics(
    *,
    mean=10.0,
    p95=12.0,
    throughput=100.0,
    peak=1000,
    count=100,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(count, mean, mean, p95, p95, mean, p95, throughput, 1000.0, peak)


def result(
    name="hotpath",
    *,
    environment=None,
    benchmark_metrics=None,
    failures=0,
    max_failures=0,
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        environment=environment or env(),
        config=BenchmarkConfig(iterations=100, warmup_iterations=10, max_failures=max_failures),
        metrics=benchmark_metrics or metrics(),
        failures=failures,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"warmup_iterations": -1},
        {"concurrency": 0},
        {"max_failures": -1},
    ],
)
def test_benchmark_config_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_environment_is_immutable_and_validates_profile():
    metadata = {"x": 1}
    item = env()
    metadata["x"] = 2
    assert item.metadata["runner"] == "test"
    with pytest.raises(TypeError):
        item.metadata["x"] = 2
    with pytest.raises(ValueError):
        BenchmarkEnvironment("", "3.11", "CPython", "Linux", "x", 1)
    with pytest.raises(ValueError):
        BenchmarkEnvironment("ci", "3.11", "CPython", "Linux", "x", 0)


def test_sample_validation():
    with pytest.raises(ValueError):
        BenchmarkSample(-1, 1)
    with pytest.raises(ValueError):
        BenchmarkSample(0, -1)
    with pytest.raises(ValueError):
        BenchmarkSample(0, 1, True, "Error")
    with pytest.raises(ValueError):
        BenchmarkSample(0, 1, False, None)


def test_metrics_validation():
    with pytest.raises(ValueError):
        BenchmarkMetrics(-1, 0, 0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        BenchmarkMetrics(1, -1, 0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        BenchmarkMetrics(1, 0, 0, 0, 0, 0, 0, 0, 0, -1)


def test_result_schema_and_failure_budget():
    assert PERFORMANCE_REPORT_SCHEMA_VERSION == 1
    assert result(failures=0).passed
    assert result(failures=1, max_failures=1).passed
    assert not result(failures=1).passed
    with pytest.raises(ValueError):
        BenchmarkResult("x", env(), BenchmarkConfig(), metrics(), schema_version=99)


def test_result_roundtrip_and_samples(tmp_path):
    original = BenchmarkResult(
        "x",
        env(),
        BenchmarkConfig(iterations=1, record_samples=True),
        metrics(count=1),
        samples=(BenchmarkSample(0, 1000),),
        metadata={"case": "roundtrip"},
    )
    path = tmp_path / "result.json"
    save_benchmark_result(original, path)
    loaded = load_benchmark_result(path)
    assert loaded.to_dict() == original.to_dict()


def test_load_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_result(path)


def test_capture_environment_has_explicit_profile():
    captured = capture_benchmark_environment("mac-m4", metadata={"power": "ac"})
    assert captured.profile_id == "mac-m4"
    assert captured.python_version
    assert captured.metadata["power"] == "ac"


@pytest.mark.asyncio
async def test_runner_warmup_is_excluded_from_measured_count():
    calls: list[int] = []

    async def operation(index: int):
        calls.append(index)
        await asyncio.sleep(0)

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=5, warmup_iterations=2),
        environment=env(),
    ).run("warmup", operation)
    assert report.metrics.count == 5
    assert calls[:2] == [-1, -2]
    assert calls[2:] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_runner_concurrency_is_bounded():
    active = max_active = 0

    async def operation(_: int):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.002)
        active -= 1

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=12, warmup_iterations=0, concurrency=3),
        environment=env(),
    ).run("bounded", operation)
    assert report.passed
    assert max_active <= 3


@pytest.mark.asyncio
async def test_runner_records_failures_without_treating_them_as_latency_samples():
    async def operation(index: int):
        if index in {1, 3}:
            raise ConnectionError("offline")

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=5, warmup_iterations=0, max_failures=2, record_samples=True),
        environment=env(),
    ).run("failure", operation)
    assert report.passed
    assert report.failures == 2
    assert report.metrics.count == 3
    assert len(report.samples) == 5
    assert [s.error_type for s in report.samples if not s.success] == ["ConnectionError", "ConnectionError"]


@pytest.mark.asyncio
async def test_runner_failure_budget_can_fail_report():
    async def operation(_: int):
        raise RuntimeError("x")

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=2, warmup_iterations=0), environment=env()
    ).run("failure", operation)
    assert not report.passed and report.metrics.count == 0


@pytest.mark.asyncio
async def test_runner_cancellation_is_observed_as_failed_sample():
    async def operation(_: int):
        raise asyncio.CancelledError()

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=1, warmup_iterations=0, max_failures=1, record_samples=True),
        environment=env(),
    ).run("cancel", operation)
    assert report.failures == 1
    assert report.samples[0].error_type == "CancelledError"


@pytest.mark.asyncio
async def test_runner_can_measure_peak_python_memory():
    holder = []

    def operation(_: int):
        holder.append(bytearray(1024))

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=4, warmup_iterations=0),
        environment=env(),
        measure_peak_memory=True,
    ).run("memory", operation)
    assert report.metrics.peak_memory_bytes is not None
    assert report.metrics.peak_memory_bytes > 0


@pytest.mark.asyncio
async def test_runner_does_not_retain_samples_by_default():
    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=3, warmup_iterations=0), environment=env()
    ).run("small", lambda _: None)
    assert report.samples == ()


@pytest.mark.asyncio
async def test_runner_can_profile_context_builder_hot_path():
    character = CharacterProfile(id="perf", name="Perf", description="benchmark character")
    builder = ContextBuilder()

    def operation(_: int):
        builder.build(character=character, history=(), user_message="hello")

    report = await BenchmarkRunner(
        config=BenchmarkConfig(iterations=5, warmup_iterations=1), environment=env()
    ).run("context_builder", operation, metadata={"hot_path": "context"})
    assert report.passed and report.metrics.count == 5
    assert report.metadata["hot_path"] == "context"


def test_budget_requires_environment_profile_for_absolute_thresholds():
    with pytest.raises(ValueError):
        PerformanceBudget(absolute_max_p95_ms=10)
    with pytest.raises(ValueError):
        PerformanceBudget(absolute_min_throughput_ops_s=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_p95_ratio": 0},
        {"max_mean_ratio": -1},
        {"min_throughput_ratio": 0},
        {"max_peak_memory_ratio": -1},
        {"absolute_max_p95_ms": -1, "environment_profile_id": "x"},
        {"absolute_min_throughput_ops_s": -1, "environment_profile_id": "x"},
    ],
)
def test_budget_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        PerformanceBudget(**kwargs)


def test_same_profile_baseline_within_budget_passes():
    evaluation = evaluate_performance_budget(
        result(benchmark_metrics=metrics(mean=10, p95=12, throughput=100, peak=1000)),
        result(benchmark_metrics=metrics(mean=10, p95=12, throughput=100, peak=1000)),
    )
    assert evaluation.passed and evaluation.comparable


def test_name_mismatch_is_not_comparable():
    evaluation = evaluate_performance_budget(result("a"), result("b"))
    assert evaluation.status is PerformanceStatus.NOT_COMPARABLE
    assert evaluation.violations[0].code == "benchmark_name_mismatch"


def test_environment_profile_mismatch_is_not_comparable():
    evaluation = evaluate_performance_budget(result(environment=env("a")), result(environment=env("b")))
    assert not evaluation.comparable
    assert evaluation.violations[0].code == "environment_profile_mismatch"


def test_failed_benchmark_is_not_comparable():
    evaluation = evaluate_performance_budget(result(failures=1), result())
    assert not evaluation.comparable
    assert evaluation.violations[0].code == "benchmark_failed"


def test_p95_regression_is_reported():
    current = result(benchmark_metrics=metrics(p95=15))
    baseline = result(benchmark_metrics=metrics(p95=10))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=1.2, max_mean_ratio=None, min_throughput_ratio=None, max_peak_memory_ratio=None))
    assert not evaluation.passed
    assert {v.code for v in evaluation.violations} == {"p95_regression"}


def test_mean_regression_is_reported():
    current = result(benchmark_metrics=metrics(mean=13))
    baseline = result(benchmark_metrics=metrics(mean=10))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=None, max_mean_ratio=1.2, min_throughput_ratio=None, max_peak_memory_ratio=None))
    assert {v.code for v in evaluation.violations} == {"mean_regression"}


def test_throughput_regression_is_reported():
    current = result(benchmark_metrics=metrics(throughput=70))
    baseline = result(benchmark_metrics=metrics(throughput=100))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=None, max_mean_ratio=None, min_throughput_ratio=0.8, max_peak_memory_ratio=None))
    assert {v.code for v in evaluation.violations} == {"throughput_regression"}


def test_peak_memory_regression_is_reported_when_both_reports_measured_memory():
    current = result(benchmark_metrics=metrics(peak=1400))
    baseline = result(benchmark_metrics=metrics(peak=1000))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=None, max_mean_ratio=None, min_throughput_ratio=None, max_peak_memory_ratio=1.25))
    assert {v.code for v in evaluation.violations} == {"peak_memory_regression"}


def test_peak_memory_budget_is_skipped_when_measurement_missing():
    cur = metrics(peak=1000)
    base = BenchmarkMetrics(100, 10, 10, 12, 12, 10, 12, 100, 1000, None)
    evaluation = evaluate_performance_budget(result(benchmark_metrics=cur), result(benchmark_metrics=base))
    assert evaluation.passed


def test_absolute_budget_requires_matching_calibrated_environment():
    evaluation = evaluate_performance_budget(
        result(environment=env("ci")),
        result(environment=env("ci")),
        budget=PerformanceBudget(environment_profile_id="mac-m4", absolute_max_p95_ms=20),
    )
    assert not evaluation.comparable
    assert evaluation.violations[0].code == "absolute_budget_environment_mismatch"


def test_absolute_latency_and_throughput_budgets_are_environment_scoped():
    budget = PerformanceBudget(
        max_p95_ratio=None,
        max_mean_ratio=None,
        min_throughput_ratio=None,
        max_peak_memory_ratio=None,
        environment_profile_id="ci-linux",
        absolute_max_p95_ms=11,
        absolute_min_throughput_ops_s=110,
    )
    evaluation = evaluate_performance_budget(
        result(benchmark_metrics=metrics(p95=12, throughput=100)), result(), budget=budget
    )
    assert {v.code for v in evaluation.violations} == {"absolute_p95_exceeded", "absolute_throughput_below_minimum"}


def test_zero_latency_baseline_fails_closed_for_positive_current():
    baseline = result(benchmark_metrics=metrics(mean=0, p95=0))
    current = result(benchmark_metrics=metrics(mean=1, p95=1))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=1.2, max_mean_ratio=1.2, min_throughput_ratio=None, max_peak_memory_ratio=None))
    assert {v.code for v in evaluation.violations} == {"p95_regression", "mean_regression"}


def test_zero_throughput_baseline_does_not_create_fake_ratio_failure():
    baseline = result(benchmark_metrics=metrics(throughput=0))
    current = result(benchmark_metrics=metrics(throughput=0))
    evaluation = evaluate_performance_budget(current, baseline, budget=PerformanceBudget(max_p95_ratio=None, max_mean_ratio=None, min_throughput_ratio=0.8, max_peak_memory_ratio=None))
    assert evaluation.passed


def test_cli_compare_passes_and_can_fail_on_regression(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    save_benchmark_result(result(benchmark_metrics=metrics(p95=10)), baseline_path)
    save_benchmark_result(result(benchmark_metrics=metrics(p95=20)), current_path)
    assert performance_main(["compare", str(baseline_path), str(current_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert performance_main(["compare", str(baseline_path), str(current_path), "--fail-on-regression"]) == 1


def test_cli_run_writes_machine_readable_report(tmp_path):
    output = tmp_path / "perf.json"
    assert performance_main(["run", "--iterations", "5", "--warmup", "1", "--profile-id", "test", "--output", str(output)]) == 0
    loaded = load_benchmark_result(output)
    assert loaded.name == "offline_noop" and loaded.environment.profile_id == "test"
    assert loaded.metadata["portable_correctness_gate"] is False


def test_performance_package_has_no_engine_authority_imports():
    forbidden = {
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
    }
    observed: set[str] = set()
    for path in PERFORMANCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                observed.add(node.module)
            elif isinstance(node, ast.Import):
                observed.update(alias.name for alias in node.names)
    assert forbidden.isdisjoint(observed)


def test_performance_public_api_is_exported_from_root():
    expected = {
        "PERFORMANCE_REPORT_SCHEMA_VERSION",
        "BenchmarkConfig",
        "BenchmarkEnvironment",
        "BenchmarkMetrics",
        "BenchmarkResult",
        "BenchmarkRunner",
        "BenchmarkSample",
        "PerformanceBudget",
        "PerformanceEvaluation",
        "PerformanceStatus",
        "PerformanceViolation",
        "capture_benchmark_environment",
        "evaluate_performance_budget",
        "load_benchmark_result",
        "save_benchmark_result",
    }
    assert expected.issubset(set(ace.__all__))
    for name in expected:
        assert hasattr(ace, name)


def test_package_and_contract_versions_are_independent():
    from ai_character_engine.distributed import DISTRIBUTED_PROTOCOL_VERSION
    from ai_character_engine.extensions import EXTENSION_API_VERSION
    from ai_character_engine.compatibility import PUBLIC_API_CONTRACT_VERSION
    from ai_character_engine_vrm import __version__ as vrm_version

    assert (ace.__version__, vrm_version) == (VERSION, VERSION)
    assert (PUBLIC_API_CONTRACT_VERSION, EXTENSION_API_VERSION, DISTRIBUTED_PROTOCOL_VERSION) == (1, 1, 1)


def test_sealed_api_upgrade_only_adds_symbols():
    from ai_character_engine.compatibility import build_public_api_manifest, compare_public_api_manifests, load_public_api_manifest

    baseline = load_public_api_manifest(ROOT / "tests/fixtures/api" / "public_api_v0.44_sealed.json")
    sealed_v045 = load_public_api_manifest(ROOT / "tests/fixtures/api" / "public_api_v0.45_sealed.json")
    report = compare_public_api_manifests(baseline, sealed_v045)
    assert sealed_v045.engine_version == "0.45.0"
    assert report.compatible and report.breaking == ()
    assert len(report.issues) == 15
    assert {issue.code for issue in report.issues} == {"symbol_added"}
