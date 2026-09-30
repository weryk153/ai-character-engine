from __future__ import annotations

import copy
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Protocol, Sequence

from .models import (
    EvalCase, EvalDimension, EvalResult, EvalTrace, PersonaRule, Severity, StateCondition,
)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _condition(case: EvalCase, condition: StateCondition) -> tuple[bool | None, object]:
    if case.state is None:
        return None, None
    if condition.field.startswith("custom."):
        key = condition.field[7:]
        if key not in case.state.custom:
            return None, None
        actual = case.state.custom[key]
    else:
        actual = getattr(case.state, condition.field)
    value = condition.value
    if condition.op in {"lt", "le", "gt", "ge"}:
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise ValueError(f"{case.case_id}: nonnumeric state field {condition.field}")
    comparisons = {
        "eq": lambda: actual == value, "ne": lambda: actual != value,
        "lt": lambda: actual < value, "le": lambda: actual <= value,
        "gt": lambda: actual > value, "ge": lambda: actual >= value,
        "in": lambda: actual in value, "not_in": lambda: actual not in value,
    }
    return comparisons[condition.op](), actual


def _applies(case: EvalCase, rule: PersonaRule) -> tuple[bool | None, dict]:
    evaluated = [(c, *_condition(case, c)) for c in rule.when]
    observed = {c.field: actual for c, _, actual in evaluated}
    if any(result is False for _, result, _ in evaluated):
        return False, observed
    if any(result is None for _, result, _ in evaluated):
        return None, observed
    return True, observed


# Explicit name assertions only: “I am happy” and “我是店長” (“I am the shop
# manager”) are not names.
# Pattern captures a whole assertion up to punctuation, avoiding prefix matches.
_NAME_PATTERN = re.compile(
    r"(?:\bmy name is\s+|\bi(?:'m| am) named\s+|我的名字是\s*|我叫\s*)(?P<value>[^\n,.!?，。！？;；:：]+)",
    re.IGNORECASE,
)


class CharacterEvaluator(Protocol):
    async def evaluate(self, case: EvalCase) -> EvalResult: ...


class RuleBasedCharacterEvaluator:
    """Deterministic baseline. Natural-language rules must be configured explicitly."""

    def evaluate_sync(self, case: EvalCase) -> EvalResult:
        trace: list[EvalTrace] = []
        if case.persona.check_name:
            values = tuple(m.group("value").strip() for m in _NAME_PATTERN.finditer(case.response))
            allowed = (case.character.name, *case.persona.name_aliases)
            incorrect = tuple(x for x in values if _normalize(x) not in {_normalize(a) for a in allowed})
            trace.append(EvalTrace(
                "identity.name", EvalDimension.IDENTITY,
                "failed" if incorrect else "passed" if values else "skipped", "rule_based",
                "Explicit self-name assertion" if values else "No explicit self-name assertion matched",
                Severity.CRITICAL if incorrect else None, incorrect, allowed, values,
            ))
        for fact in case.persona.facts:
            matches = tuple(re.finditer(fact.claim_pattern, case.response, re.IGNORECASE))
            values = tuple(m.group("value") for m in matches)
            if any(value is None or not value.strip() for value in values):
                raise ValueError(f"{fact.fact_id}: claim pattern produced an empty value")
            allowed = {_normalize(x) for x in fact.allowed_values}
            bad = tuple(m.group(0) for m in matches if _normalize(m.group("value")) not in allowed)
            trace.append(EvalTrace(
                fact.fact_id, fact.dimension, "failed" if bad else "passed" if matches else "skipped",
                "rule_based", fact.statement if matches else "No explicit fact claim matched",
                fact.severity if bad else None, bad, fact.allowed_values, values,
            ))
        for rule in case.persona.rules:
            applicable, observed = _applies(case, rule)
            if applicable is not True:
                trace.append(EvalTrace(
                    rule.rule_id, rule.dimension, "skipped", "rule_based",
                    "State condition is false" if applicable is False else "Required state is missing",
                    expected=[{"field": c.field, "op": c.op, "value": c.value} for c in rule.when],
                    observed=observed,
                ))
                continue
            evidence: tuple[str, ...] = ()
            if rule.mode == "max_length":
                failed = len(case.response) > rule.max_chars
                expected = {"max_chars": rule.max_chars}
                observed["response_chars"] = len(case.response)
                if failed:
                    evidence = (case.response,)
            else:
                matches = tuple(m.group(0) for m in re.finditer(rule.pattern, case.response, re.IGNORECASE))
                failed = bool(matches) if rule.mode == "forbidden" else not matches
                evidence = matches if failed else ()
                expected = {"mode": rule.mode, "pattern": rule.pattern}
            expected["when"] = [{"field": c.field, "op": c.op, "value": c.value} for c in rule.when]
            trace.append(EvalTrace(
                rule.rule_id, rule.dimension, "failed" if failed else "passed", "rule_based",
                rule.description, rule.severity if failed else None, evidence, expected, observed,
            ))
        for boundary in case.persona.relationships:
            matches = tuple(m.group(0) for m in re.finditer(boundary.pattern, case.response, re.IGNORECASE))
            stage = case.state.relationship_stage if case.state else None
            failed = stage is not None and bool(matches) and stage not in boundary.allowed_stages
            trace.append(EvalTrace(
                boundary.boundary_id, EvalDimension.RELATIONSHIP,
                "skipped" if stage is None else "failed" if failed else "passed", "rule_based",
                "Required state is missing" if stage is None else boundary.description,
                boundary.severity if failed else None, matches if failed else (), boundary.allowed_stages, stage,
            ))
        return EvalResult(case.case_id, tuple(trace))

    async def evaluate(self, case: EvalCase) -> EvalResult:
        return self.evaluate_sync(case)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    constraint_id: str
    status: str
    rationale: str
    evidence: tuple[str, ...] = ()
    severity: Severity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_id, str) or not self.constraint_id.strip():
            raise ValueError("judge constraint_id must be nonempty")
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("invalid judge status")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("judge rationale must be nonempty")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if any(not isinstance(x, str) or not x.strip() for x in self.evidence):
            raise ValueError("judge evidence must contain nonempty strings")
        if self.severity is not None:
            object.__setattr__(self, "severity", Severity(self.severity))
            if self.status != "failed":
                raise ValueError("only failed verdicts may specify severity")


class LLMJudgeAdapter(Protocol):
    """Return one verdict per requested ID. No provider dependency in this contract.

    Implementations own prompts, authentication, timeout, retries and structured
    output parsing. Treat case response/history as untrusted material to assess.
    """

    async def judge(self, case: EvalCase, *, constraint_ids: tuple[str, ...]) -> Sequence[JudgeVerdict]: ...


class CallableLLMJudgeAdapter:
    def __init__(self, callback: Callable[..., Awaitable[Sequence[JudgeVerdict]]]) -> None:
        self.callback = callback

    async def judge(self, case: EvalCase, *, constraint_ids: tuple[str, ...]) -> Sequence[JudgeVerdict]:
        return await self.callback(case, constraint_ids=constraint_ids)


class PersonaEvaluator:
    """Compose a rule-based baseline and an opt-in LLM judge; failures are unioned.

    Judge errors propagate (a failed/incomplete judge run never becomes a pass).
    A judge pass cannot erase a deterministic failure. Evaluation never alters
    runtime state, memory, history or generated dialogue.
    """

    def __init__(self, *, judge: LLMJudgeAdapter | None = None, llm_judge_enabled: bool = False) -> None:
        if type(llm_judge_enabled) is not bool:
            raise ValueError("llm_judge_enabled must be a boolean")
        if llm_judge_enabled and judge is None:
            raise ValueError("llm_judge_enabled requires a judge adapter")
        self.judge = judge
        self.llm_judge_enabled = llm_judge_enabled
        self.baseline = RuleBasedCharacterEvaluator()

    async def evaluate(self, case: EvalCase) -> EvalResult:
        baseline = self.baseline.evaluate_sync(case)
        if not self.llm_judge_enabled:
            return baseline
        catalog = {}
        if case.persona.check_name:
            catalog["identity.name"] = (EvalDimension.IDENTITY, Severity.CRITICAL, True)
        for fact in case.persona.facts:
            catalog[fact.fact_id] = (fact.dimension, fact.severity, True)
        for rule in case.persona.rules:
            if _applies(case, rule)[0] is True:
                catalog[rule.rule_id] = (rule.dimension, rule.severity, rule.mode != "required")
        for boundary in case.persona.relationships:
            if case.state is not None:
                catalog[boundary.boundary_id] = (EvalDimension.RELATIONSHIP, boundary.severity, True)
        if not catalog:
            return baseline
        # Ground-truth labels are never shown to the judge. Protect the input
        # from accidental adapter mutation as well as mutable runtime objects.
        judge_case = replace(copy.deepcopy(case), expected_failures=None)
        verdicts = tuple(await self.judge.judge(judge_case, constraint_ids=tuple(catalog)))
        if any(not isinstance(v, JudgeVerdict) for v in verdicts):
            raise ValueError("judge must return JudgeVerdict objects")
        ids = [v.constraint_id for v in verdicts]
        if len(ids) != len(set(ids)) or set(ids) != set(catalog):
            raise ValueError("judge must return exactly one verdict for every requested constraint ID")
        traces = list(baseline.trace)
        baseline_by_id = {t.constraint_id: t for t in baseline.trace}
        for verdict in verdicts:
            dimension, default_severity, evidence_required = catalog[verdict.constraint_id]
            if any(e not in case.response for e in verdict.evidence):
                raise ValueError("judge evidence must be verbatim response text")
            if verdict.status == "failed" and evidence_required and not verdict.evidence:
                raise ValueError("judge failure requires response evidence")
            traces.append(EvalTrace(
                verdict.constraint_id, dimension, verdict.status, "llm_judge", verdict.rationale,
                (verdict.severity or default_severity) if verdict.status == "failed" else None,
                verdict.evidence, baseline_by_id[verdict.constraint_id].expected,
                baseline_by_id[verdict.constraint_id].observed,
            ))
        return EvalResult(case.case_id, tuple(traces), llm_judge_executed=True)
