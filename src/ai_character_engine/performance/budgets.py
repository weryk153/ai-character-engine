from __future__ import annotations

from .models import (
    BenchmarkResult,
    PerformanceBudget,
    PerformanceEvaluation,
    PerformanceStatus,
    PerformanceViolation,
)


def evaluate_performance_budget(
    current: BenchmarkResult,
    baseline: BenchmarkResult,
    *,
    budget: PerformanceBudget | None = None,
) -> PerformanceEvaluation:
    policy = budget or PerformanceBudget()
    if current.name != baseline.name:
        return PerformanceEvaluation(
            PerformanceStatus.NOT_COMPARABLE,
            current.name,
            baseline.name,
            (PerformanceViolation("benchmark_name_mismatch", "current and baseline benchmark names differ"),),
        )
    if current.environment.profile_id != baseline.environment.profile_id:
        return PerformanceEvaluation(
            PerformanceStatus.NOT_COMPARABLE,
            current.name,
            baseline.name,
            (
                PerformanceViolation(
                    "environment_profile_mismatch",
                    "performance regression comparison requires the same environment profile id",
                ),
            ),
        )
    if not current.passed or not baseline.passed:
        return PerformanceEvaluation(
            PerformanceStatus.NOT_COMPARABLE,
            current.name,
            baseline.name,
            (PerformanceViolation("benchmark_failed", "current or baseline benchmark exceeded its failure budget"),),
        )

    violations: list[PerformanceViolation] = []
    _max_ratio(violations, "p95_regression", current.metrics.p95_ms, baseline.metrics.p95_ms, policy.max_p95_ratio)
    _max_ratio(violations, "mean_regression", current.metrics.mean_ms, baseline.metrics.mean_ms, policy.max_mean_ratio)
    _min_ratio(
        violations,
        "throughput_regression",
        current.metrics.throughput_ops_s,
        baseline.metrics.throughput_ops_s,
        policy.min_throughput_ratio,
    )
    if (
        policy.max_peak_memory_ratio is not None
        and current.metrics.peak_memory_bytes is not None
        and baseline.metrics.peak_memory_bytes is not None
    ):
        _max_ratio(
            violations,
            "peak_memory_regression",
            float(current.metrics.peak_memory_bytes),
            float(baseline.metrics.peak_memory_bytes),
            policy.max_peak_memory_ratio,
        )

    if policy.environment_profile_id is not None:
        if current.environment.profile_id != policy.environment_profile_id:
            return PerformanceEvaluation(
                PerformanceStatus.NOT_COMPARABLE,
                current.name,
                baseline.name,
                (
                    PerformanceViolation(
                        "absolute_budget_environment_mismatch",
                        f"absolute thresholds are calibrated for {policy.environment_profile_id!r}",
                    ),
                ),
            )
        if policy.absolute_max_p95_ms is not None and current.metrics.p95_ms > policy.absolute_max_p95_ms:
            violations.append(
                PerformanceViolation(
                    "absolute_p95_exceeded",
                    "current p95 latency exceeds the environment-calibrated absolute budget",
                    current.metrics.p95_ms,
                    policy.absolute_max_p95_ms,
                )
            )
        if (
            policy.absolute_min_throughput_ops_s is not None
            and current.metrics.throughput_ops_s < policy.absolute_min_throughput_ops_s
        ):
            violations.append(
                PerformanceViolation(
                    "absolute_throughput_below_minimum",
                    "current throughput is below the environment-calibrated absolute budget",
                    current.metrics.throughput_ops_s,
                    policy.absolute_min_throughput_ops_s,
                )
            )

    return PerformanceEvaluation(
        PerformanceStatus.PASSED if not violations else PerformanceStatus.FAILED,
        current.name,
        baseline.name,
        tuple(violations),
    )


def _max_ratio(
    violations: list[PerformanceViolation],
    code: str,
    current: float,
    baseline: float,
    limit: float | None,
) -> None:
    if limit is None:
        return
    if baseline == 0:
        if current > 0:
            violations.append(PerformanceViolation(code, "baseline is zero; any positive current value exceeds ratio budget"))
        return
    ratio = current / baseline
    if ratio > limit:
        violations.append(PerformanceViolation(code, f"observed ratio {ratio:.4f} exceeds limit {limit:.4f}", ratio, limit))


def _min_ratio(
    violations: list[PerformanceViolation],
    code: str,
    current: float,
    baseline: float,
    limit: float | None,
) -> None:
    if limit is None:
        return
    if baseline == 0:
        return
    ratio = current / baseline
    if ratio < limit:
        violations.append(PerformanceViolation(code, f"observed ratio {ratio:.4f} is below minimum {limit:.4f}", ratio, limit))
