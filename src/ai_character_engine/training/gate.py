from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdapterEvaluationMetrics:
    persona_consistency: float
    task_success: float
    regression_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("persona_consistency", "task_success", "regression_rate"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdapterAcceptanceResult:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdapterAcceptanceGate:
    min_persona_consistency: float = 0.9
    min_task_success: float = 0.8
    max_regression_rate: float = 0.05
    max_persona_drop_from_baseline: float = 0.02

    def evaluate(
        self,
        candidate: AdapterEvaluationMetrics,
        *,
        baseline: AdapterEvaluationMetrics | None = None,
    ) -> AdapterAcceptanceResult:
        reasons: list[str] = []
        if candidate.persona_consistency < self.min_persona_consistency:
            reasons.append("persona consistency below threshold")
        if candidate.task_success < self.min_task_success:
            reasons.append("task success below threshold")
        if candidate.regression_rate > self.max_regression_rate:
            reasons.append("regression rate above threshold")
        if baseline is not None and candidate.persona_consistency < baseline.persona_consistency - self.max_persona_drop_from_baseline:
            reasons.append("persona consistency regressed versus baseline")
        return AdapterAcceptanceResult(accepted=not reasons, reasons=tuple(reasons))
