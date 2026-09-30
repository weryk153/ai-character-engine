"""v0.12 persona consistency evaluation; opt-in and provider-neutral."""
from .dataset import (
    AggregateMetrics, CharacterEvalDataset, CharacterEvalReport, aggregate_metrics, evaluate_dataset,
)
from .evaluator import (
    CallableLLMJudgeAdapter, CharacterEvaluator, JudgeVerdict, LLMJudgeAdapter,
    PersonaEvaluator, RuleBasedCharacterEvaluator,
)
from .models import (
    EvalCase, EvalDimension, EvalResult, EvalTrace, PersonaFact, PersonaRule,
    PersonaSpec, RelationshipBoundary, Severity, StateCondition,
)

__all__ = [
    "AggregateMetrics", "CharacterEvalDataset", "CharacterEvalReport", "aggregate_metrics", "evaluate_dataset",
    "CallableLLMJudgeAdapter", "CharacterEvaluator", "JudgeVerdict", "LLMJudgeAdapter",
    "PersonaEvaluator", "RuleBasedCharacterEvaluator", "EvalCase", "EvalDimension", "EvalResult", "EvalTrace",
    "PersonaFact", "PersonaRule", "PersonaSpec", "RelationshipBoundary", "Severity", "StateCondition",
]

_COGNITIVE_EXPORTS = {
    "CognitiveEvalCase", "CognitiveEvalDimension", "CognitiveEvalResult",
    "CognitiveEvalSource", "CognitiveEvalTrace", "CognitiveTimelineFrame",
    "CallableCognitiveJudgeAdapter", "CognitiveEvalPolicy", "CognitiveEvaluator",
    "CognitiveJudgeAdapter", "CognitiveJudgeVerdict", "CognitiveAggregateMetrics",
    "CognitiveEvalDataset", "CognitiveEvalReport", "aggregate_cognitive_metrics",
    "evaluate_cognitive_dataset",
}
__all__ += sorted(_COGNITIVE_EXPORTS)

def __getattr__(name):
    if name in _COGNITIVE_EXPORTS:
        from ai_character_engine import cognitive_evaluation as module
        return getattr(module, name)
    raise AttributeError(name)
