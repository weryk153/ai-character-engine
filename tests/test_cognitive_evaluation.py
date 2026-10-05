from __future__ import annotations

from ai_character_engine._version import VERSION
import asyncio
import json
from dataclasses import replace

import pytest

from ai_character_engine import __version__
from ai_character_engine.collaboration import (
    CollaborationResult,
    CognitiveWorkItem,
    CognitiveWorkPlan,
    SpecialistFinding,
    SpecialistKind,
    SpecialistRunStatus,
    VerificationDecision,
    VerificationIssue,
    VerificationReport,
)
from ai_character_engine.evaluation import (
    CallableCognitiveJudgeAdapter,
    CognitiveEvalCase,
    CognitiveEvalDataset,
    CognitiveEvalDimension,
    CognitiveEvalPolicy,
    CognitiveEvalSource,
    CognitiveEvaluator,
    CognitiveJudgeVerdict,
    CognitiveTimelineFrame,
    aggregate_cognitive_metrics,
    evaluate_cognitive_dataset,
)
from ai_character_engine.events import CharacterEvent
from ai_character_engine.goals import GoalEvidenceRef, GoalHorizon, GoalRecord, GoalStatus, MotivationKind, MotivationSignal
from ai_character_engine.long_term_cognition import BeliefClaim, BeliefRecord, BeliefStatus, CognitionEvidenceRef, ReflectionRecord
from ai_character_engine.memory import MemoryRecord
from ai_character_engine.state import CharacterStateSnapshot
from ai_character_engine.evaluation.models import Severity


def ev(source_id: str, text: str, evidence_type: str = "asserted_fact") -> CognitionEvidenceRef:
    return CognitionEvidenceRef("event", source_id, evidence_type, text, 0.9)


def signal(kind: MotivationKind, source_type: str, source_id: str, excerpt: str = "evidence") -> MotivationSignal:
    return MotivationSignal(kind, 0.8, GoalEvidenceRef(source_type, source_id, excerpt), "grounded reason")


def collaboration(*, finding_status=SpecialistRunStatus.SUCCEEDED, accepted=True, evidence_ids=("src1",), specialist_id="memory") -> CollaborationResult:
    item = CognitiveWorkItem("w1", "memory", "inspect memory evidence", ("src1",))
    plan = CognitiveWorkPlan("understand prior preference", "memory may matter", (item,), 3)
    finding = SpecialistFinding(
        "w1", specialist_id, SpecialistKind.MEMORY, finding_status,
        summary="Relevant prior preference." if finding_status is SpecialistRunStatus.SUCCEEDED else None,
        confidence=0.9 if finding_status is SpecialistRunStatus.SUCCEEDED else None,
        evidence_source_ids=evidence_ids if finding_status is SpecialistRunStatus.SUCCEEDED else (),
        error=None if finding_status is SpecialistRunStatus.SUCCEEDED else "failed",
    )
    decision = VerificationDecision.ACCEPT if accepted else VerificationDecision.REJECT
    verification = VerificationReport(decision, ("w1",) if accepted else (), (), "checked")
    return CollaborationResult("understand prior preference", plan, (finding,), verification, 3, "c1")


def good_case() -> CognitiveEvalCase:
    e1 = CharacterEvent("user_message", "user", "Please keep answers concise", id="e1")
    e2 = CharacterEvent("user_message", "user", "Short replies are easier for me", id="e2")
    memory = MemoryRecord("char", "User asked for concise answers", source_event_id="e1", id="m1")
    r1 = ReflectionRecord("char", "User may prefer concise answers", 0.9, (ev("e1", "Please keep answers concise"),), BeliefClaim("user", "prefers_response_style", "concise"), id="r1")
    r2 = ReflectionRecord("char", "Concise output seems preferred", 0.9, (ev("e2", "Short replies are easier for me"),), BeliefClaim("user", "prefers_response_style", "concise"), id="r2")
    belief = BeliefRecord("char", BeliefClaim("user", "prefers_response_style", "concise"), 0.9, (ev("e1", "Please keep answers concise"), ev("e2", "Short replies are easier for me")), ("r1", "r2"), 2, id="b1")
    goal = GoalRecord(
        "char", "Answer the user concisely", GoalHorizon.LONG_TERM, 0.5, 0.9,
        (signal(MotivationKind.EXPLICIT_REQUEST, "event", "e1", "Please keep answers concise"), signal(MotivationKind.BELIEF_ALIGNMENT, "belief", "b1", "user prefers concise")),
        id="g1",
    )
    completed = replace(goal, status=GoalStatus.COMPLETED)
    frame1 = CognitiveTimelineFrame(2, (belief,), (goal,))
    frame2 = CognitiveTimelineFrame(3, (belief,), (completed,))
    return CognitiveEvalCase.from_artifacts(
        case_id="good", character_id="char", revision=3,
        state=CharacterStateSnapshot("neutral", 80, 60, 60, "friend", {}),
        events=(e1, e2), memories=(memory,), reflections=(r1, r2), beliefs=(belief,), goals=(goal,),
        collaborations=(collaboration(),), extra_sources=(CognitiveEvalSource("collaboration", "src1", "prior preference evidence"),),
        timeline=(frame1, frame2),
    )


@pytest.mark.asyncio
async def test_good_case_passes_rule_based_evaluation():
    result = await CognitiveEvaluator().evaluate(good_case())
    assert result.passed is True
    assert result.quality_score == 1.0
    assert not result.violations


def test_from_artifacts_builds_canonical_source_catalog():
    case = good_case()
    keys = case.source_map()
    assert ("event", "e1") in keys
    assert ("memory", "m1") in keys
    assert ("belief", "b1") in keys
    assert ("state", "emotion") in keys


@pytest.mark.asyncio
async def test_cross_character_artifact_is_critical_scope_failure():
    case = good_case()
    leaked = replace(case.memories[0], character_id="other")
    result = await CognitiveEvaluator().evaluate(replace(case, memories=(leaked,)))
    assert any(v.code == "evidence.character_scope" and v.severity is Severity.CRITICAL for v in result.violations)


@pytest.mark.asyncio
async def test_missing_reflection_evidence_source_is_high_failure():
    case = good_case()
    bad = replace(case.reflections[0], evidence=(ev("missing", "x"),))
    result = await CognitiveEvaluator().evaluate(replace(case, reflections=(bad, case.reflections[1])))
    assert any(v.code == "evidence.reflection_source_resolves" and v.severity is Severity.HIGH for v in result.violations)


@pytest.mark.asyncio
async def test_memory_source_event_must_resolve():
    case = good_case()
    bad = replace(case.memories[0], source_event_id="missing")
    result = await CognitiveEvaluator().evaluate(replace(case, memories=(bad,)))
    assert any(v.code == "evidence.memory_source_resolves" for v in result.violations)


@pytest.mark.asyncio
async def test_active_belief_requires_independent_support():
    case = good_case()
    one = BeliefRecord("char", BeliefClaim("user", "likes", "tea"), 0.9, (ev("e1", "tea"),), ("r1",), 1, id="weak")
    result = await CognitiveEvaluator().evaluate(replace(case, beliefs=(one,)))
    assert any(v.code == "belief.independent_support" and v.artifact_id == "weak" for v in result.violations)


@pytest.mark.asyncio
async def test_unsafe_belief_evidence_is_critical():
    case = good_case()
    unsafe = CognitionEvidenceRef("event", "e1", "quoted_reference", "quote", 0.9)
    belief = BeliefRecord("char", BeliefClaim("user", "likes", "tea"), 0.9, (unsafe, ev("e2", "tea")), ("r1", "r2"), 2, id="unsafe")
    result = await CognitiveEvaluator().evaluate(replace(case, beliefs=(belief,)))
    assert any(v.code == "belief.safe_evidence" and v.severity is Severity.CRITICAL for v in result.violations)


@pytest.mark.asyncio
async def test_conflicting_active_beliefs_are_critical():
    case = good_case()
    b1 = case.beliefs[0]
    b2 = BeliefRecord("char", BeliefClaim("user", "prefers_response_style", "detailed"), 0.9, (ev("e1", "x"), ev("e2", "y")), ("r1", "r2"), 2, id="b2")
    result = await CognitiveEvaluator().evaluate(replace(case, beliefs=(b1, b2)))
    assert any(v.code == "belief.active_conflict" and v.severity is Severity.CRITICAL for v in result.violations)


@pytest.mark.asyncio
async def test_duplicate_active_belief_value_is_detected():
    case = good_case()
    duplicate = replace(case.beliefs[0], id="b2")
    result = await CognitiveEvaluator().evaluate(replace(case, beliefs=(case.beliefs[0], duplicate)))
    assert any(v.code == "belief.duplicate_active_value" for v in result.violations)


@pytest.mark.asyncio
async def test_goal_motivation_kind_source_compatibility():
    case = good_case()
    bad = replace(case.goals[0], motivation_signals=(signal(MotivationKind.BELIEF_ALIGNMENT, "event", "e1"),))
    result = await CognitiveEvaluator().evaluate(replace(case, goals=(bad,)))
    assert any(v.code == "goal.motivation_source_compatible" for v in result.violations)


@pytest.mark.asyncio
async def test_goal_cannot_ground_on_inactive_memory_source():
    case = good_case()
    forgotten = replace(case.memories[0], status="forgotten")
    goal = replace(case.goals[0], motivation_signals=(signal(MotivationKind.COMMITMENT, "memory", "m1"),))
    sources = tuple(replace(x, status="forgotten") if x.key == ("memory", "m1") else x for x in case.sources)
    result = await CognitiveEvaluator().evaluate(replace(case, memories=(forgotten,), goals=(goal,), sources=sources))
    assert any(v.code == "goal.source_is_active" for v in result.violations)


@pytest.mark.asyncio
async def test_goal_cannot_ground_on_contested_belief_source():
    case = good_case()
    contested = replace(case.beliefs[0], status=BeliefStatus.CONTESTED)
    goal = replace(case.goals[0], motivation_signals=(signal(MotivationKind.BELIEF_ALIGNMENT, "belief", "b1"),))
    sources = tuple(replace(x, status="contested") if x.key == ("belief", "b1") else x for x in case.sources)
    result = await CognitiveEvaluator().evaluate(replace(case, beliefs=(contested,), goals=(goal,), sources=sources))
    assert any(v.code == "goal.source_is_active" for v in result.violations)


@pytest.mark.asyncio
async def test_unreconciled_active_goal_conflict_is_high_failure():
    case = good_case()
    g1 = replace(case.goals[0], conflict_key="reply-style")
    g2 = GoalRecord("char", "Answer in exhaustive detail", GoalHorizon.LONG_TERM, 0.5, 0.8, (signal(MotivationKind.EXPLICIT_REQUEST, "event", "e2"),), conflict_key="reply-style", id="g2")
    result = await CognitiveEvaluator().evaluate(replace(case, goals=(g1, g2)))
    assert any(v.code == "goal.unreconciled_active_conflict" and v.severity is Severity.HIGH for v in result.violations)


@pytest.mark.asyncio
async def test_specialist_finding_must_match_planned_specialist():
    case = good_case()
    bad = collaboration(specialist_id="vision")
    result = await CognitiveEvaluator().evaluate(replace(case, collaborations=(bad,)))
    assert any(v.code == "specialist.work_item_alignment" for v in result.violations)


@pytest.mark.asyncio
async def test_specialist_cannot_cite_outside_source_slice():
    case = good_case()
    bad = collaboration(evidence_ids=("outside",))
    result = await CognitiveEvaluator().evaluate(replace(case, collaborations=(bad,)))
    assert any(v.code == "specialist.evidence_slice" and v.severity is Severity.CRITICAL for v in result.violations)


@pytest.mark.asyncio
async def test_verifier_cannot_accept_failed_finding():
    case = good_case()
    bad = collaboration(finding_status=SpecialistRunStatus.FAILED, accepted=True)
    result = await CognitiveEvaluator().evaluate(replace(case, collaborations=(bad,)))
    assert any(v.code == "verifier.accepts_only_successful" for v in result.violations)


@pytest.mark.asyncio
async def test_verifier_partial_accepting_all_is_incoherent():
    case = good_case()
    collab = collaboration()
    verification = replace(collab.verification, decision=VerificationDecision.PARTIAL)
    result = await CognitiveEvaluator().evaluate(replace(case, collaborations=(replace(collab, verification=verification),)))
    assert any(v.code == "verifier.decision_coherence" for v in result.violations)


@pytest.mark.asyncio
async def test_verifier_issue_must_reference_known_work_item():
    case = good_case()
    collab = collaboration()
    verification = replace(collab.verification, issues=(VerificationIssue("bad", "unknown item", "missing"),))
    result = await CognitiveEvaluator().evaluate(replace(case, collaborations=(replace(collab, verification=verification),)))
    assert any(v.code == "verifier.issue_references" for v in result.violations)


@pytest.mark.asyncio
async def test_timeline_revisions_must_increase():
    case = good_case()
    result = await CognitiveEvaluator().evaluate(replace(case, timeline=(case.timeline[1], case.timeline[0])))
    assert any(v.code == "timeline.revision_monotonic" for v in result.violations)


@pytest.mark.asyncio
async def test_belief_id_cannot_change_claim_across_timeline():
    case = good_case()
    b = case.beliefs[0]
    changed = replace(b, claim=BeliefClaim("user", "prefers_response_style", "detailed"))
    frames = (CognitiveTimelineFrame(1, (b,), ()), CognitiveTimelineFrame(2, (changed,), ()))
    result = await CognitiveEvaluator().evaluate(replace(case, timeline=frames))
    assert any(v.code == "timeline.belief_identity_stable" for v in result.violations)


@pytest.mark.asyncio
async def test_retired_belief_cannot_reactivate():
    case = good_case(); b = case.beliefs[0]
    frames = (CognitiveTimelineFrame(1, (replace(b, status=BeliefStatus.RETIRED),), ()), CognitiveTimelineFrame(2, (b,), ()))
    result = await CognitiveEvaluator().evaluate(replace(case, timeline=frames))
    assert any(v.code == "timeline.retired_belief_terminal" for v in result.violations)


@pytest.mark.asyncio
async def test_goal_id_cannot_change_semantic_identity_across_timeline():
    case = good_case(); g = case.goals[0]
    changed = replace(g, objective="Different objective")
    frames = (CognitiveTimelineFrame(1, (), (g,)), CognitiveTimelineFrame(2, (), (changed,)))
    result = await CognitiveEvaluator().evaluate(replace(case, timeline=frames))
    assert any(v.code == "timeline.goal_identity_stable" for v in result.violations)


@pytest.mark.asyncio
async def test_completed_goal_cannot_reactivate():
    case = good_case(); g = case.goals[0]
    frames = (CognitiveTimelineFrame(1, (), (replace(g, status=GoalStatus.COMPLETED),)), CognitiveTimelineFrame(2, (), (g,)))
    result = await CognitiveEvaluator().evaluate(replace(case, timeline=frames))
    assert any(v.code == "timeline.terminal_goal_not_reactivated" for v in result.violations)


@pytest.mark.asyncio
async def test_optional_semantic_judge_is_provider_neutral_and_non_authoritative():
    case = good_case()
    original_beliefs = case.beliefs
    async def judge(received, *, dimensions):
        assert received is case
        assert CognitiveEvalDimension.SPECIALIST_RELEVANCE in dimensions
        return [CognitiveJudgeVerdict(CognitiveEvalDimension.SPECIALIST_RELEVANCE, "passed", "finding answers the objective")]
    result = await CognitiveEvaluator(judge=CallableCognitiveJudgeAdapter(judge), semantic_judge_enabled=True).evaluate(case)
    assert result.semantic_judge_executed is True
    assert case.beliefs == original_beliefs
    assert any(t.source == "semantic_judge" for t in result.trace)


def test_semantic_judge_requires_explicit_adapter():
    with pytest.raises(ValueError):
        CognitiveEvaluator(semantic_judge_enabled=True)
    with pytest.raises(ValueError):
        CognitiveEvaluator(semantic_judge_enabled="yes")


@pytest.mark.asyncio
async def test_semantic_judge_rejects_duplicate_dimension_verdicts():
    async def judge(case, *, dimensions):
        v = CognitiveJudgeVerdict(CognitiveEvalDimension.EVIDENCE_FAITHFULNESS, "passed", "ok")
        return [v, v]
    with pytest.raises(ValueError, match="duplicate"):
        await CognitiveEvaluator(judge=CallableCognitiveJudgeAdapter(judge), semantic_judge_enabled=True).evaluate(good_case())


def test_policy_requires_positive_minimum_belief_support():
    with pytest.raises(ValueError):
        CognitiveEvalPolicy(min_active_belief_support=0)


def test_jsonl_roundtrip_preserves_cognitive_case(tmp_path):
    case = good_case(); path = tmp_path / "cognitive.jsonl"
    CognitiveEvalDataset((case,)).to_jsonl(path)
    loaded = CognitiveEvalDataset.from_jsonl(path)
    assert loaded.cases[0].case_id == case.case_id
    assert loaded.cases[0].beliefs[0].claim == case.beliefs[0].claim
    assert loaded.cases[0].collaborations[0].verification.decision is VerificationDecision.ACCEPT


def test_a_case_keeps_her_mood_intensity_and_when_it_was_set():
    state = CharacterStateSnapshot(
        emotion="sad", energy=90.0, trust=60.0, favorability=55.0,
        relationship_stage="friend", mood_intensity=0.7, mood_updated_at=1000.0,
    )
    case = CognitiveEvalCase("mood", "char", state=state)
    loaded = CognitiveEvalCase.from_dict(json.loads(json.dumps(case.to_dict())))
    assert loaded.state == state


def test_dataset_duplicate_ids_rejected():
    case = good_case()
    with pytest.raises(ValueError, match="unique"):
        CognitiveEvalDataset((case, case))


def test_jsonl_error_includes_path_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"; path.write_text("\n{bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2:"):
        CognitiveEvalDataset.from_jsonl(path)


@pytest.mark.asyncio
async def test_dataset_report_and_labeled_accuracy(tmp_path):
    case = replace(good_case(), expected_failures=())
    report = await evaluate_cognitive_dataset(CognitiveEvalDataset((case,)))
    assert report.metrics.pass_rate == 1.0
    assert report.metrics.label_accuracy == 1.0
    out = tmp_path / "report.json"; report.to_json(out)
    assert json.loads(out.read_text())["schema_version"] == "0.37"


@pytest.mark.asyncio
async def test_aggregate_metrics_track_dimension_coverage():
    good = await CognitiveEvaluator().evaluate(good_case())
    metrics = aggregate_cognitive_metrics((good,))
    assert metrics.total_cases == 1
    assert metrics.per_dimension_coverage["belief_consistency"] == 1.0
    assert metrics.mean_quality_score == 1.0


@pytest.mark.asyncio
async def test_empty_case_is_unassessed_not_false_pass():
    case = CognitiveEvalCase("empty", "char")
    result = await CognitiveEvaluator().evaluate(case)
    assert result.passed is None
    assert result.quality_score is None


def test_public_api_and_version():
    from ai_character_engine import CognitiveEvaluator as PublicEvaluator, CognitiveEvalCase as PublicCase
    assert PublicEvaluator is CognitiveEvaluator
    assert PublicCase is CognitiveEvalCase
    assert __version__ == VERSION


def test_cognitive_evaluation_core_has_no_runtime_authority_imports():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "ai_character_engine" / "cognitive_evaluation"
    forbidden = {
        "ai_character_engine.commit.coordinator",
        "ai_character_engine.goals.manager",
        "ai_character_engine.goals.store",
        "ai_character_engine.long_term_cognition.manager",
        "ai_character_engine.long_term_cognition.store",
        "ai_character_engine.memory.manager",
        "ai_character_engine.memory.store",
        "ai_character_engine.runtime.character_runtime",
        "ai_character_engine.tools.executor",
    }
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "") in forbidden:
                offenders.append((path.name, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        offenders.append((path.name, alias.name))
    assert offenders == []


def test_cognitive_cli_success_and_output(tmp_path):
    import subprocess, sys
    from pathlib import Path
    dataset = tmp_path / "cases.jsonl"
    output = tmp_path / "report.json"
    CognitiveEvalDataset((replace(good_case(), timeline=()),)).to_jsonl(dataset)
    proc = subprocess.run(
        [sys.executable, "-m", "ai_character_engine.cognitive_evaluation", str(dataset), "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env=dict(__import__("os").environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )
    assert proc.returncode == 0
    assert json.loads(output.read_text())["schema_version"] == "0.37"


def test_cognitive_cli_fail_on_violations(tmp_path):
    import subprocess, sys
    from pathlib import Path
    case = good_case()
    weak = BeliefRecord("char", BeliefClaim("user", "likes", "tea"), 0.9, (ev("e1", "tea"),), ("r1",), 1, id="weak-cli")
    dataset = tmp_path / "bad.jsonl"
    CognitiveEvalDataset((replace(case, beliefs=(weak,), timeline=()),)).to_jsonl(dataset)
    proc = subprocess.run(
        [sys.executable, "-m", "ai_character_engine.cognitive_evaluation", str(dataset), "--fail-on-violations"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env=dict(__import__("os").environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )
    assert proc.returncode == 1


def test_cognitive_cli_protects_input_file(tmp_path):
    import subprocess, sys
    from pathlib import Path
    dataset = tmp_path / "cases.jsonl"
    CognitiveEvalDataset((good_case(),)).to_jsonl(dataset)
    before = dataset.read_text()
    proc = subprocess.run(
        [sys.executable, "-m", "ai_character_engine.cognitive_evaluation", str(dataset), "--output", str(dataset)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        env=dict(__import__("os").environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )
    assert proc.returncode == 2
    assert dataset.read_text() == before
