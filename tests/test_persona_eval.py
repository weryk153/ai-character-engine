from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ai_character_engine import CharacterProfile, CharacterState, StatePatch
from ai_character_engine.evaluation import (
    CallableLLMJudgeAdapter, CharacterEvalDataset, EvalCase, EvalDimension,
    JudgeVerdict, PersonaEvaluator, PersonaFact, PersonaRule, PersonaSpec,
    RelationshipBoundary, RuleBasedCharacterEvaluator, Severity, StateCondition,
    aggregate_metrics, evaluate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'examples/data/character_eval.jsonl'


@pytest.fixture
def cases():
    return CharacterEvalDataset.from_jsonl(FIXTURE).cases


@pytest.mark.parametrize('index', range(8))
def test_positive_and_negative_dimensions_and_trace(cases, index):
    case = cases[index]
    result = RuleBasedCharacterEvaluator().evaluate_sync(case)
    assert {v.dimension for v in result.violations} == set(case.expected_failures)
    assert result.passed == (index == 0)
    assert set(result.evaluated_dimensions) == set(EvalDimension)
    for trace in result.violations:
        assert trace.constraint_id
        assert trace.evidence and all(e in case.response for e in trace.evidence)
        assert trace.expected is not None
        assert trace.severity is not None
    if index == 4:
        assert result.violations[0].observed['energy'] == 10
    if index == 5:
        assert result.violations[0].observed == 'stranger'


def test_same_romantic_line_allowed_at_partner_stage(cases):
    case = replace(cases[5], state=CharacterState(energy=10, relationship_stage='partner'))
    assert RuleBasedCharacterEvaluator().evaluate_sync(case).passed is True


def test_unknown_relationship_stage_is_not_implicitly_allowed(cases):
    case = replace(cases[5], state=CharacterState(energy=10, relationship_stage='unknown'))
    assert {v.dimension for v in RuleBasedCharacterEvaluator().evaluate_sync(case).violations} == {EvalDimension.RELATIONSHIP}


def test_same_energy_line_not_rejected_when_energy_is_high(cases):
    case = replace(cases[4], state=CharacterState(energy=90))
    result = RuleBasedCharacterEvaluator().evaluate_sync(case)
    assert result.passed is True
    assert next(t for t in result.trace if t.constraint_id == 'state.low_energy').status == 'skipped'


def test_missing_state_skips_state_constraints(cases):
    result = RuleBasedCharacterEvaluator().evaluate_sync(replace(cases[5], state=None))
    assert EvalDimension.STATE not in result.evaluated_dimensions
    assert EvalDimension.RELATIONSHIP not in result.evaluated_dimensions
    assert [t.message for t in result.trace if t.dimension == EvalDimension.STATE] == ['Required state is missing']


def test_name_alias_casefold_and_no_generic_i_am_claim(cases):
    evaluator = RuleBasedCharacterEvaluator()
    assert evaluator.evaluate_sync(replace(cases[0], response='My name is akari.')).passed is True
    result = evaluator.evaluate_sync(replace(cases[0], response='I am happy. 我是店長。'))
    assert not any(v.dimension == EvalDimension.IDENTITY for v in result.violations)
    assert next(t for t in result.trace if t.constraint_id == 'identity.name').status == 'skipped'


def test_all_claims_checked_and_longer_name_not_prefix_match(cases):
    result = RuleBasedCharacterEvaluator().evaluate_sync(replace(cases[0], response='我叫燈。我叫燈光。我的故鄉是京都。我的故鄉是東京。'))
    assert {v.constraint_id for v in result.violations} == {'identity.name', 'fact.hometown'}


def test_unassessed_case_is_not_perfect_pass(cases):
    case = replace(cases[0], persona=PersonaSpec(check_name=False))
    result = RuleBasedCharacterEvaluator().evaluate_sync(case)
    assert result.passed is None and result.consistency_score is None
    metrics = aggregate_metrics([result])
    assert metrics.pass_rate is None and metrics.consistency_score is None
    assert metrics.unassessed_cases == 1
    assert set(metrics.per_dimension_failure_rate.values()) == {None}


def test_empty_aggregate_and_dataset():
    metrics = aggregate_metrics([])
    assert metrics.total_cases == 0 and metrics.pass_rate is None
    assert set(metrics.per_dimension_coverage.values()) == {0.0}
    assert asyncio.run(evaluate_dataset(CharacterEvalDataset(()))).metrics.label_accuracy is None


async def test_metrics_exact_formulas_and_labels(cases):
    report = await evaluate_dataset(CharacterEvalDataset(cases))
    m = report.metrics
    assert m.pass_rate == 1 / 8
    assert m.per_dimension_failure_rate['identity'] == 2 / 8
    assert m.per_dimension_failure_rate['state'] == 1 / 8
    assert set(m.per_dimension_coverage.values()) == {1.0}
    assert m.mean_severity == pytest.approx(5.25 / 8)
    assert m.consistency_score == pytest.approx(1 - 5.25 / (8 * 6))
    assert m.severity_counts == {'low': 0, 'medium': 1, 'high': 5, 'critical': 1, 'none': 1}
    assert m.label_accuracy == 1.0 and m.labeled_cases == 8


async def test_no_ground_truth_leakage_into_baseline(cases):
    before = await PersonaEvaluator().evaluate(cases[6])
    after = await PersonaEvaluator().evaluate(replace(cases[6], expected_failures=()))
    assert before == after
    report = await evaluate_dataset(CharacterEvalDataset((replace(cases[6], expected_failures=()),)))
    assert report.metrics.label_accuracy == 0


def test_required_pattern_and_length_boundary(cases):
    persona = PersonaSpec(check_name=False, rules=(
        PersonaRule('style.limit', 'Length limit', 'style', mode='max_length', max_chars=3),
        PersonaRule('style.marker', 'Required marker', 'style', pattern='hi', mode='required'),
    ))
    evaluator = RuleBasedCharacterEvaluator()
    assert evaluator.evaluate_sync(replace(cases[0], persona=persona, response='hi!')).passed
    result = evaluator.evaluate_sync(replace(cases[0], persona=persona, response='xxxx'))
    assert len(result.violations) == 2
    assert next(t for t in result.violations if t.constraint_id == 'style.marker').evidence == ()
    assert result.consistency_score == 0.5


@pytest.mark.parametrize('op,value,applies', [('eq','rain',True), ('ne','rain',False), ('in',['rain','snow'],True), ('not_in',['rain'],False)])
def test_custom_state_conditions(cases, op, value, applies):
    persona = PersonaSpec(check_name=False, rules=(PersonaRule('weather', 'Weather', 'state', 'sunny', when=(StateCondition('custom.weather',op,value),)),))
    case = replace(cases[0], persona=persona, response='sunny', state=CharacterState(custom={'weather':'rain'}))
    assert RuleBasedCharacterEvaluator().evaluate_sync(case).passed is (False if applies else None)
    assert RuleBasedCharacterEvaluator().evaluate_sync(replace(case,state=CharacterState())).passed is None


@pytest.mark.parametrize('op', ['lt','le','gt','ge'])
def test_numeric_condition_boundaries(cases,op):
    persona=PersonaSpec(check_name=False,rules=(PersonaRule('energy', 'Energy', 'state', 'tired', when=(StateCondition('energy',op,20),)),))
    case=replace(cases[0],persona=persona,response='tired',state=CharacterState(energy=20))
    assert RuleBasedCharacterEvaluator().evaluate_sync(case).passed is (False if op in {'le','ge'} else None)


def test_case_captures_runtime_state_without_mutation(cases):
    state = CharacterState(energy=10, custom={'nested': {'a': 1}})
    profile = CharacterProfile('id','燈','description')
    case = replace(cases[0], state=state, character=profile)
    state.apply(StatePatch(energy_delta=80))
    state.custom['nested']['a'] = 2
    profile.name = 'Changed'
    assert case.state.energy == 10 and case.state.custom['nested']['a'] == 1
    assert case.character.name == '燈'
    before = case.to_dict()
    RuleBasedCharacterEvaluator().evaluate_sync(case)
    assert case.to_dict() == before


async def test_disabled_judge_never_accesses_adapter(cases):
    class Bomb:
        @property
        def judge(self):
            raise AssertionError('Disabled branch touched the provider')
    report = await evaluate_dataset(CharacterEvalDataset(cases), PersonaEvaluator(judge=Bomb(), llm_judge_enabled=False))
    assert not any(r.llm_judge_executed for r in report.results)
    assert report.metrics.label_accuracy == 1


async def test_judge_adapter_no_labels_no_mutation_and_union(cases):
    seen = []
    async def judge(case, *, constraint_ids):
        assert case.expected_failures is None
        seen.append(constraint_ids)
        case.character.name = 'Changed inside judge'
        return [JudgeVerdict(cid,'passed','No additional semantic issue') for cid in constraint_ids]
    result = await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge), llm_judge_enabled=True).evaluate(cases[6])
    assert seen and result.llm_judge_executed
    assert not result.passed
    assert result.severity == Severity.CRITICAL
    assert cases[6].character.name == '燈'


async def test_judge_deduplicates_metrics(cases):
    async def judge(case, *, constraint_ids):
        return [JudgeVerdict(cid, 'failed', 'Disrespectful phrase', ('你是笨蛋',), Severity.HIGH)
                if cid == 'rule.respect' else JudgeVerdict(cid,'passed','OK') for cid in constraint_ids]
    result = await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge), llm_judge_enabled=True).evaluate(cases[2])
    assert len(result.violations) == 2
    assert aggregate_metrics([result]).severity_counts['high'] == 1
    assert aggregate_metrics([result]).per_dimension_failure_rate['personality'] == 1
    assert result.consistency_score == pytest.approx(1 - .75/6)


async def test_judge_can_find_semantic_issue_baseline_misses(cases):
    case = replace(cases[0], response=cases[0].response+'你的智力實在令人擔憂。')
    assert (await PersonaEvaluator().evaluate(case)).passed
    async def judge(case, *, constraint_ids):
        return [JudgeVerdict(cid,'failed','Indirect insult',('你的智力實在令人擔憂',))
                if cid == 'rule.respect' else JudgeVerdict(cid,'passed','OK') for cid in constraint_ids]
    result = await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge),llm_judge_enabled=True).evaluate(case)
    assert result.violations[0].constraint_id == 'rule.respect'
    assert result.violations[0].source == 'llm_judge'


@pytest.mark.parametrize('bad', ['empty','unknown','duplicate','evidence','no_evidence','raw'])
async def test_invalid_judge_output_fails_closed(cases, bad):
    async def judge(case, *, constraint_ids):
        verdicts=[JudgeVerdict(cid,'passed','OK') for cid in constraint_ids]
        if bad == 'empty': return []
        if bad == 'unknown': verdicts[0]=JudgeVerdict('invented','passed','OK')
        if bad == 'duplicate': verdicts.append(verdicts[0])
        if bad == 'evidence': verdicts[0]=JudgeVerdict(constraint_ids[0],'failed','Bad',('not in response',))
        if bad == 'no_evidence': verdicts[0]=JudgeVerdict(constraint_ids[0],'failed','Bad')
        if bad == 'raw': return [{}]
        return verdicts
    with pytest.raises(ValueError):
        await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge),llm_judge_enabled=True).evaluate(cases[0])


async def test_judge_error_propagates(cases):
    async def judge(case, **kwargs):
        raise TimeoutError('provider unavailable')
    with pytest.raises(TimeoutError):
        await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge),llm_judge_enabled=True).evaluate(cases[0])


async def test_judge_does_not_assess_inactive_constraints(cases):
    async def judge(case, *, constraint_ids):
        assert 'state.low_energy' not in constraint_ids
        assert 'relationship.romance' not in constraint_ids
        return [JudgeVerdict(cid,'skipped','Insufficient context') for cid in constraint_ids]
    await PersonaEvaluator(judge=CallableLLMJudgeAdapter(judge),llm_judge_enabled=True).evaluate(replace(cases[0],state=None))


def test_judge_requires_explicit_valid_configuration():
    with pytest.raises(ValueError): PersonaEvaluator(llm_judge_enabled=True)
    with pytest.raises(ValueError): PersonaEvaluator(llm_judge_enabled='false')


def test_jsonl_roundtrip_and_report_json(cases, tmp_path):
    path=tmp_path/'dataset.jsonl'
    CharacterEvalDataset(cases).to_jsonl(path)
    loaded=CharacterEvalDataset.from_jsonl(path)
    assert loaded.cases == cases
    report=asyncio.run(evaluate_dataset(loaded))
    report.to_json(tmp_path/'report.json')
    assert json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))['results'][6]['severity'] == 'critical'


@pytest.mark.parametrize('row', ['{bad json', '[]', '{}', '{"case_id":"x","unexpected":42}'])
def test_jsonl_errors_include_path_and_line(tmp_path,row):
    path=tmp_path/'bad.jsonl'
    path.write_text('\n'+row+'\n')
    with pytest.raises(ValueError,match=r'bad.jsonl:2:'):
        CharacterEvalDataset.from_jsonl(path)


def test_duplicate_ids_rejected(cases,tmp_path):
    with pytest.raises(ValueError,match='unique'):
        CharacterEvalDataset((cases[0],cases[0]))
    path=tmp_path/'duplicate.jsonl'
    path.write_text((json.dumps(cases[0].to_dict())+'\n')*2)
    with pytest.raises(ValueError,match=r'duplicate.jsonl:2:'):
        CharacterEvalDataset.from_jsonl(path)
    with pytest.raises(ValueError,match='unique'):
        PersonaSpec(rules=(cases[0].persona.rules[0],)*2)


@pytest.mark.parametrize('build', [
    lambda: PersonaFact('f','Fact','no group',('a',)),
    lambda: PersonaFact('f','Fact','(?P<value>.+)',()),
    lambda: PersonaRule('r','Rule','style','['),
    lambda: PersonaRule('r','Rule','style',mode='max_length',max_chars=0),
    lambda: PersonaRule('identity.name','Rule','identity','x'),
    lambda: PersonaRule('r','Rule','unknown','x'),
    lambda: StateCondition('badfield','eq',0),
    lambda: StateCondition('energy','lt',float('nan')),
    lambda: StateCondition('energy','in','abc'),
    lambda: RelationshipBoundary('b','Boundary','x',()),
])
def test_invalid_spec_fails_early(build):
    with pytest.raises((ValueError, __import__('re').error)): build()


def test_cli_exit_codes_output_and_input_protection(tmp_path):
    output=tmp_path/'report.json'
    cmd=[sys.executable,'-m','ai_character_engine.evaluation',str(FIXTURE),'--output',str(output)]
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT / "src"))
    assert subprocess.run(cmd,cwd=ROOT,capture_output=True,env=env).returncode == 0
    assert json.loads(output.read_text(encoding='utf-8'))['metrics']['failed_cases'] == 7
    assert subprocess.run(cmd+['--fail-on-violations'],cwd=ROOT,capture_output=True,env=env).returncode == 1
    bad=tmp_path/'bad.jsonl'; bad.write_text('{')
    assert subprocess.run(cmd[:3]+[str(bad)],cwd=ROOT,capture_output=True,env=env).returncode == 2
    assert subprocess.run(cmd[:3]+[str(bad),'--output',str(bad)],cwd=ROOT,capture_output=True,env=env).returncode == 2
    assert bad.read_text() == '{'


def test_retrieval_evaluation_api_is_exported():
    from ai_character_engine.memory import (
        RetrievalEvalCase, RetrievalEvalDataset, RetrievalEvaluator,
        RetrievalComparisonReport, QueryRewriter, IdentityQueryRewriter,
        ContextAppendingQueryRewriter, CallableQueryRewriter,
        AsyncMemoryReranker, CallableAsyncMemoryReranker, compare_retrievers,
    )
    from ai_character_engine import CharacterRuntime, __version__
    assert __version__ == '1.0.0'
    assert RetrievalEvalDataset((RetrievalEvalCase('c','query',('memory',)),)).cases[0].character_id == 'c'
