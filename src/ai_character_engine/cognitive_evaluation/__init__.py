"""v0.37 read-only cognition-wide evaluation contracts."""
from .models import (
    CognitiveEvalCase, CognitiveEvalDimension, CognitiveEvalResult, CognitiveEvalSource,
    CognitiveEvalTrace, CognitiveTimelineFrame,
)
from .evaluator import (
    CallableCognitiveJudgeAdapter, CognitiveEvalPolicy, CognitiveEvaluator,
    CognitiveJudgeAdapter, CognitiveJudgeVerdict,
)
from .dataset import (
    CognitiveAggregateMetrics, CognitiveEvalDataset, CognitiveEvalReport,
    aggregate_cognitive_metrics, evaluate_cognitive_dataset,
)
__all__ = [
    "CognitiveEvalCase", "CognitiveEvalDimension", "CognitiveEvalResult",
    "CognitiveEvalSource", "CognitiveEvalTrace", "CognitiveTimelineFrame",
    "CallableCognitiveJudgeAdapter", "CognitiveEvalPolicy", "CognitiveEvaluator",
    "CognitiveJudgeAdapter", "CognitiveJudgeVerdict", "CognitiveAggregateMetrics",
    "CognitiveEvalDataset", "CognitiveEvalReport", "aggregate_cognitive_metrics",
    "evaluate_cognitive_dataset",
]
