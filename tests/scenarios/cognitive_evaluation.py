from __future__ import annotations

import asyncio

from ai_character_engine.collaboration import (
    CollaborationResult, CognitiveWorkItem, CognitiveWorkPlan, SpecialistFinding, SpecialistKind,
    SpecialistRunStatus, VerificationDecision, VerificationReport,
)
from ai_character_engine.cognitive_evaluation import (
    CognitiveEvalCase,
    CognitiveEvalDataset,
    CognitiveEvaluator,
    CognitiveTimelineFrame,
    evaluate_cognitive_dataset,
)
from ai_character_engine.events import CharacterEvent
from ai_character_engine.goals import GoalEvidenceRef, GoalHorizon, GoalRecord, GoalStatus, MotivationKind, MotivationSignal
from ai_character_engine.long_term_cognition import BeliefClaim, BeliefRecord, CognitionEvidenceRef, ReflectionRecord
from ai_character_engine.memory import MemoryRecord
from ai_character_engine.state import CharacterStateSnapshot


def build_case() -> CognitiveEvalCase:
    first = CharacterEvent("user_message", "user", "Please keep answers concise", id="event-1")
    second = CharacterEvent("user_message", "user", "Short replies are easier for me", id="event-2")
    e1 = CognitionEvidenceRef("event", first.id, "asserted_fact", first.content, 0.92)
    e2 = CognitionEvidenceRef("event", second.id, "asserted_fact", second.content, 0.91)
    r1 = ReflectionRecord("demo-character", "The user may prefer concise answers.", 0.9, (e1,), BeliefClaim("user", "prefers_response_style", "concise"), id="reflection-1")
    r2 = ReflectionRecord("demo-character", "Concise replies seem easier for the user.", 0.9, (e2,), BeliefClaim("user", "prefers_response_style", "concise"), id="reflection-2")
    belief = BeliefRecord("demo-character", BeliefClaim("user", "prefers_response_style", "concise"), 0.91, (e1, e2), (r1.id, r2.id), 2, id="belief-1")
    memory = MemoryRecord("demo-character", "The user asked for concise answers.", source_event_id=first.id, id="memory-1")
    motivation = MotivationSignal(
        MotivationKind.BELIEF_ALIGNMENT,
        0.8,
        GoalEvidenceRef("belief", belief.id, "user prefers concise response style"),
        "Align future response style with the active long-term belief.",
    )
    goal = GoalRecord("demo-character", "Keep future replies concise", GoalHorizon.LONG_TERM, 0.4, 0.9, (motivation,), id="goal-1")
    completed = GoalRecord(
        goal.character_id, goal.objective, goal.horizon, goal.urgency, goal.confidence,
        goal.motivation_signals, status=GoalStatus.COMPLETED, id=goal.id,
    )
    item = CognitiveWorkItem("work-1", "memory", "Check prior response-style evidence", ("memory-1",))
    plan = CognitiveWorkPlan("Check useful context", "Prior preference may matter", (item,), 2)
    finding = SpecialistFinding(
        "work-1", "memory", SpecialistKind.MEMORY, SpecialistRunStatus.SUCCEEDED,
        summary="The stored memory is relevant to response style.", confidence=0.9,
        evidence_source_ids=("memory-1",),
    )
    verification = VerificationReport(VerificationDecision.ACCEPT, ("work-1",), (), "Finding is grounded in the assigned source.")
    collaboration = CollaborationResult("Check useful context", plan, (finding,), verification, 2, "collab-1")
    return CognitiveEvalCase.from_artifacts(
        case_id="demo-cognition",
        character_id="demo-character",
        revision=2,
        state=CharacterStateSnapshot("neutral", 80, 60, 60, "acquaintance", {}),
        events=(first, second),
        memories=(memory,),
        reflections=(r1, r2),
        beliefs=(belief,),
        goals=(goal,),
        collaborations=(collaboration,),
        timeline=(CognitiveTimelineFrame(1, (belief,), (goal,)), CognitiveTimelineFrame(2, (belief,), (completed,))),
        expected_failures=(),
    )


async def main() -> None:
    case = build_case()
    result = await CognitiveEvaluator().evaluate(case)
    print("case:", result.case_id)
    print("passed:", result.passed)
    print("quality_score:", result.quality_score)
    print("violations:", [trace.code for trace in result.violations])

    report = await evaluate_cognitive_dataset(CognitiveEvalDataset((case,)))
    print("dataset pass_rate:", report.metrics.pass_rate)
    print("dimension coverage:", report.metrics.per_dimension_coverage)


if __name__ == "__main__":
    asyncio.run(main())
