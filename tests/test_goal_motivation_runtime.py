from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.commit import CognitiveCommitCoordinator, CommitNextAction, CommitStatus
from ai_character_engine.cognition import (
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
)
from ai_character_engine.context import ContextBudget, ContextBuilder
from ai_character_engine.events import CharacterEvent
from ai_character_engine.goals import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalManager,
    GoalRecord,
    GoalStatus,
    InMemoryGoalStore,
    JsonlGoalStore,
    MotivationKind,
    MotivationSignal,
)
from ai_character_engine.llm import ModelEndpoint
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.long_term_cognition import (
    BeliefClaim,
    BeliefRecord,
    BeliefStatus,
    CognitionEvidenceRef,
    InMemoryLongTermCognitionStore,
    LongTermCognitionManager,
    ReflectionRecord,
)
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager, MemoryRecord
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import CharacterState
from ai_character_engine.tasks import MultiTaskRuntime, TaskProposal
from tests.fakes import system_context


class StaticLLM:
    def __init__(self, text: str = "foreground reply") -> None:
        self.text = text
        self.messages: list[list[Message]] = []

    async def generate(self, messages, *, tools=None):
        self.messages.append(list(messages))
        return LLMResponse(text=self.text, model="foreground")


class GoalSourceAwareClient:
    """Returns a valid goal proposal that cites the actual current event id."""

    def __init__(self) -> None:
        self.messages: list[list[Message]] = []

    async def generate(self, messages, *, tools=None):
        self.messages.append(list(messages))
        prompt = messages[-1].content
        match = re.search(r'"event"\s*:\s*\{.*?"id"\s*:\s*"([^"]+)"', prompt, re.S)
        if not match:
            raise AssertionError("authoritative event source was not provided to goal worker")
        event_id = match.group(1)
        payload = {
            "goals": [
                {
                    "objective": "Help finish the deployment task",
                    "horizon": "short_term",
                    "urgency": 0.8,
                    "conflict_key": None,
                    "motivation_signals": [
                        {
                            "kind": "explicit_request",
                            "strength": 0.9,
                            "source_type": "event",
                            "source_id": event_id,
                            "rationale": "The user explicitly asked for help finishing it.",
                        }
                    ],
                    "confidence": 0.91,
                }
            ],
            "confidence": 0.91,
            "evidence": ["explicit current request"],
        }
        return LLMResponse(text=json.dumps(payload), model="goal-worker")


class WordTokenEstimator:
    def estimate_text(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    def estimate_message(self, message: Message) -> int:
        return 1 + self.estimate_text(message.content)

    def estimate_tools(self, tools) -> int:
        return 0


class FailOnceGoalStore(InMemoryGoalStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_replace = False

    def replace_goals(self, character_id, records):
        super().replace_goals(character_id, records)
        if self.fail_next_replace:
            self.fail_next_replace = False
            raise OSError("synthetic goal write failure")


def memory_record(
    summary: str = "User asked to finish deployment",
    *,
    status: str = "active",
    evidence_type: str = "asserted_fact",
    record_id: str = "mem-1",
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        character_id="c1",
        summary=summary,
        kind="task",
        metadata={"evidence_type": evidence_type},
        status=status,
    )


def cognition_evidence(source_id: str) -> CognitionEvidenceRef:
    return CognitionEvidenceRef(
        source_type="event",
        source_id=source_id,
        evidence_type="asserted_fact",
        excerpt="The user consistently prefers concise answers",
        confidence=0.9,
    )


def belief_record(
    *,
    status: BeliefStatus = BeliefStatus.ACTIVE,
    record_id: str = "belief-1",
) -> BeliefRecord:
    refs = (cognition_evidence("b-e1"), cognition_evidence("b-e2"))
    return BeliefRecord(
        id=record_id,
        character_id="c1",
        claim=BeliefClaim("user", "prefers_response_style", "concise"),
        confidence=0.9,
        evidence=refs,
        source_reflection_ids=("r1", "r2"),
        support_count=2,
        status=status,
    )


def goal_record(
    objective: str = "Finish deployment",
    *,
    source_id: str = "evt-1",
    strength: float = 0.8,
    urgency: float = 0.6,
    horizon: GoalHorizon = GoalHorizon.SHORT_TERM,
    status: GoalStatus = GoalStatus.ACTIVE,
    conflict_key: str | None = None,
    record_id: str | None = None,
) -> GoalRecord:
    kwargs = {"id": record_id} if record_id else {}
    return GoalRecord(
        character_id="c1",
        objective=objective,
        horizon=horizon,
        urgency=urgency,
        confidence=0.9,
        motivation_signals=(
            MotivationSignal(
                kind=MotivationKind.EXPLICIT_REQUEST,
                strength=strength,
                evidence=GoalEvidenceRef(
                    source_type="event",
                    source_id=source_id,
                    excerpt="Please help finish deployment",
                ),
                rationale="Direct current request",
            ),
        ),
        status=status,
        conflict_key=conflict_key,
        **kwargs,
    )


def engine(
    *,
    goal_manager: GoalManager | None = None,
    memory_manager: MemoryManager | None = None,
    long_term_cognition: LongTermCognitionManager | None = None,
    llm=None,
    state: CharacterState | None = None,
) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c1", name="Test", description="Stay in character."),
        llm=llm or StaticLLM(),
        goal_manager=goal_manager,
        memory_manager=memory_manager,
        long_term_cognition=long_term_cognition,
        state=state,
    )


def goal_proposal(
    *,
    event_id: str = "evt-1",
    objective: str = "Finish deployment",
    horizon: str = "short_term",
    urgency: float = 0.7,
    confidence: float = 0.9,
    conflict_key: str | None = None,
    kind: str = "explicit_request",
    source_type: str = "event",
    source_id: str | None = None,
    strength: float = 0.85,
    revision: int = 0,
    evidence_type: str = "user_instruction",
    proposal_id: str | None = None,
) -> TaskProposal:
    source_id = source_id or event_id
    kwargs = {"id": proposal_id} if proposal_id else {}
    return TaskProposal(
        target="cognition.goal_candidate",
        payload={
            "objective": objective,
            "horizon": horizon,
            "urgency": urgency,
            "conflict_key": conflict_key,
            "motivation_signals": [
                {
                    "kind": kind,
                    "strength": strength,
                    "source_type": source_type,
                    "source_id": source_id,
                    "rationale": "Supported by canonical evidence",
                }
            ],
        },
        base_revision=revision,
        source_task_id=f"task-{event_id}",
        confidence=confidence,
        provenance={
            "worker_kind": "goal_motivation",
            "foreground_event_id": event_id,
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "foreground_event_content": "Please help finish deployment",
            "evidence_type": evidence_type,
        },
        created_at=datetime.now(UTC),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_goal_manager_is_opt_in_and_v034_host_stays_review_only():
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine()))
    result = await commits.commit(goal_proposal())
    assert result.status is CommitStatus.REVIEW_REQUIRED
    assert result.next_action is CommitNextAction.MANUAL_REVIEW
    assert result.reason == "goal_manager_not_configured"


@pytest.mark.asyncio
async def test_explicit_current_event_can_commit_one_source_goal_without_fact_promotion_gate():
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=manager)))
    result = await commits.commit(goal_proposal())
    assert result.status is CommitStatus.COMMITTED
    goals = manager.active_goals(character_id="c1")
    assert len(goals) == 1
    assert goals[0].objective == "Finish deployment"
    assert goals[0].support_count == 1
    assert goals[0].motivation_signals[0].evidence.source_type == "event"


@pytest.mark.asyncio
async def test_goal_cannot_cite_invented_or_unsafe_event_as_authority():
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=manager)))
    invented = await commits.commit(goal_proposal(source_id="not-current-event"))
    assert invented.status is CommitStatus.REJECTED
    assert invented.reason == "goal_source_not_authoritative"

    quote = await CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=GoalManager()))).commit(
        goal_proposal(evidence_type="quoted_reference")
    )
    assert quote.status is CommitStatus.REJECTED
    assert quote.reason == "goal_source_not_authoritative"


@pytest.mark.asyncio
async def test_motivation_kind_source_type_compatibility_is_enforced():
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=manager)))
    result = await commits.commit(
        goal_proposal(kind="belief_alignment", source_type="event")
    )
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "goal_source_kind_mismatch"


@pytest.mark.asyncio
async def test_active_memory_can_support_goal_but_inactive_memory_cannot():
    active = memory_record(record_id="mem-active")
    memories = MemoryManager(store=InMemoryMemoryStore((active,)))
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(
        MultiTaskRuntime(engine(goal_manager=manager, memory_manager=memories))
    )
    committed = await commits.commit(
        goal_proposal(
            kind="unfinished_intent",
            source_type="memory",
            source_id=active.id,
        )
    )
    assert committed.status is CommitStatus.COMMITTED
    assert manager.active_goals(character_id="c1")[0].motivation_signals[0].evidence.excerpt == active.summary

    inactive = memory_record(status="superseded", record_id="mem-old")
    memories2 = MemoryManager(store=InMemoryMemoryStore((inactive,)))
    result = await CognitiveCommitCoordinator(
        MultiTaskRuntime(engine(goal_manager=GoalManager(), memory_manager=memories2))
    ).commit(
        goal_proposal(
            kind="unfinished_intent",
            source_type="memory",
            source_id=inactive.id,
        )
    )
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "goal_source_not_authoritative"


@pytest.mark.asyncio
async def test_only_active_belief_can_support_belief_alignment_goal():
    active = belief_record(record_id="belief-active")
    cognition = LongTermCognitionManager(
        store=InMemoryLongTermCognitionStore(beliefs=(active,))
    )
    manager = GoalManager()
    result = await CognitiveCommitCoordinator(
        MultiTaskRuntime(
            engine(goal_manager=manager, long_term_cognition=cognition)
        )
    ).commit(
        goal_proposal(
            kind="belief_alignment",
            source_type="belief",
            source_id=active.id,
        )
    )
    assert result.status is CommitStatus.COMMITTED
    assert "prefers_response_style" in manager.active_goals(character_id="c1")[0].motivation_signals[0].evidence.excerpt

    contested = belief_record(status=BeliefStatus.CONTESTED, record_id="belief-contested")
    cognition2 = LongTermCognitionManager(
        store=InMemoryLongTermCognitionStore(beliefs=(contested,))
    )
    rejected = await CognitiveCommitCoordinator(
        MultiTaskRuntime(
            engine(goal_manager=GoalManager(), long_term_cognition=cognition2)
        )
    ).commit(
        goal_proposal(
            kind="belief_alignment",
            source_type="belief",
            source_id=contested.id,
        )
    )
    assert rejected.status is CommitStatus.REJECTED
    assert rejected.reason == "goal_source_not_authoritative"


@pytest.mark.asyncio
async def test_state_pressure_must_reference_real_current_state_field():
    state = CharacterState(energy=12, custom={"needs_followup": True})
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(
        MultiTaskRuntime(engine(goal_manager=manager, state=state))
    )
    ok = await commits.commit(
        goal_proposal(
            kind="state_pressure",
            source_type="state",
            source_id="energy",
        )
    )
    assert ok.status is CommitStatus.COMMITTED
    assert "energy=12" in manager.active_goals(character_id="c1")[0].motivation_signals[0].evidence.excerpt

    bad = await CognitiveCommitCoordinator(
        MultiTaskRuntime(engine(goal_manager=GoalManager(), state=state))
    ).commit(
        goal_proposal(
            kind="state_pressure",
            source_type="state",
            source_id="custom:missing",
        )
    )
    assert bad.status is CommitStatus.REJECTED
    assert bad.reason == "goal_source_not_authoritative"


@pytest.mark.asyncio
async def test_new_independent_evidence_reinforces_same_goal_and_preserves_id():
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=manager)))
    first = await commits.commit(goal_proposal(event_id="evt-1", strength=0.6))
    before = manager.active_goals(character_id="c1")[0]
    second = await commits.commit(goal_proposal(event_id="evt-2", strength=0.7))
    after = manager.active_goals(character_id="c1")[0]
    assert first.status is CommitStatus.COMMITTED
    assert second.status is CommitStatus.COMMITTED
    assert after.id == before.id
    assert after.support_count == 2
    assert after.motivation_score > before.motivation_score
    assert {x.evidence.source_id for x in after.motivation_signals} == {"evt-1", "evt-2"}


@pytest.mark.asyncio
async def test_replaying_same_authoritative_event_does_not_inflate_goal_support():
    manager = GoalManager()
    runtime = engine(goal_manager=manager)
    p1 = goal_proposal(event_id="evt-1", proposal_id="p-one")
    assert (await CognitiveCommitCoordinator(MultiTaskRuntime(runtime)).commit(p1)).status is CommitStatus.COMMITTED
    p2 = goal_proposal(event_id="evt-1", proposal_id="p-two")
    duplicate = await CognitiveCommitCoordinator(MultiTaskRuntime(runtime)).commit(p2)
    assert duplicate.status is CommitStatus.DUPLICATE
    assert manager.active_goals(character_id="c1")[0].support_count == 1


def test_same_source_multiple_signal_kinds_cannot_inflate_motivation_score():
    evidence = GoalEvidenceRef(source_type="event", source_id="evt-1", excerpt="x")
    record = GoalRecord(
        character_id="c1",
        objective="Follow up",
        horizon=GoalHorizon.SHORT_TERM,
        urgency=0.5,
        confidence=0.9,
        motivation_signals=(
            MotivationSignal(MotivationKind.EXPLICIT_REQUEST, 0.6, evidence, "direct"),
            MotivationSignal(MotivationKind.COMMITMENT, 0.9, evidence, "commitment"),
        ),
    )
    assert record.support_count == 1
    assert record.motivation_score == pytest.approx(0.9)


def test_conflicting_goals_are_blocked_instead_of_confidence_winner_takes_all():
    manager = GoalManager()
    a = goal_record("Deploy now", source_id="a", conflict_key="deployment_strategy")
    b = goal_record("Delay deployment", source_id="b", conflict_key="deployment_strategy")
    manager.commit_candidate(a)
    manager.commit_candidate(b)
    stored = manager.goals(character_id="c1")
    assert {item.status for item in stored} == {GoalStatus.BLOCKED}
    assert manager.active_goals(character_id="c1") == ()


def test_retiring_one_conflicting_goal_unblocks_the_remaining_goal():
    manager = GoalManager()
    a = goal_record("Deploy now", source_id="a", conflict_key="deployment_strategy")
    b = goal_record("Delay deployment", source_id="b", conflict_key="deployment_strategy")
    manager.commit_candidate(a)
    manager.commit_candidate(b)
    manager.transition(
        character_id="c1",
        goal_id=b.id,
        status=GoalStatus.RETIRED,
        reason="Host resolved the conflict",
    )
    active = manager.active_goals(character_id="c1")
    assert len(active) == 1
    assert active[0].id == a.id
    assert active[0].status is GoalStatus.ACTIVE


def test_host_lifecycle_transitions_keep_goal_status_out_of_model_truth_path():
    manager = GoalManager()
    committed = manager.commit_candidate(goal_record()).goal
    paused = manager.transition(
        character_id="c1", goal_id=committed.id, status=GoalStatus.PAUSED, reason="Wait for dependency"
    )
    assert paused.status is GoalStatus.PAUSED
    assert manager.active_goals(character_id="c1") == ()
    resumed = manager.transition(
        character_id="c1", goal_id=committed.id, status=GoalStatus.ACTIVE, reason="Dependency resolved"
    )
    assert resumed.status is GoalStatus.ACTIVE
    with pytest.raises(ValueError):
        manager.transition(
            character_id="c1", goal_id=committed.id, status=GoalStatus.BLOCKED, reason="manual"
        )


def test_jsonl_store_round_trips_goal_provenance_and_lifecycle(tmp_path):
    path = tmp_path / "goals.jsonl"
    manager = GoalManager(store=JsonlGoalStore(path))
    created = manager.commit_candidate(goal_record(record_id="goal-1")).goal
    manager.transition(
        character_id="c1", goal_id=created.id, status=GoalStatus.PAUSED, reason="wait"
    )
    reloaded = GoalManager(store=JsonlGoalStore(path))
    stored = reloaded.goals(character_id="c1")
    assert len(stored) == 1
    assert stored[0].id == "goal-1"
    assert stored[0].status is GoalStatus.PAUSED
    assert stored[0].support_count == 1
    assert stored[0].metadata["transitions"][0]["reason"] == "wait"


def test_goal_commit_rolls_back_store_on_failure_and_retry_can_succeed():
    store = FailOnceGoalStore()
    manager = GoalManager(store=store)
    store.fail_next_replace = True
    with pytest.raises(OSError, match="synthetic goal write failure"):
        manager.commit_candidate(goal_record())
    assert manager.goals(character_id="c1") == ()
    result = manager.commit_candidate(goal_record())
    assert result.created is True
    assert len(manager.active_goals(character_id="c1")) == 1


def test_context_includes_only_active_goals_with_separate_budget_and_safety_label():
    active = goal_record("Finish deployment", record_id="g-active")
    paused = replace(goal_record("Write release notes", record_id="g-paused"), status=GoalStatus.PAUSED)
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=140,
            reserved_output_tokens=10,
            recent_history_target_tokens=0,
            max_goal_tokens=30,
            max_belief_tokens=0,
            max_memory_tokens=0,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    result = builder.build_for_event_with_trace(
        character=CharacterProfile(id="c1", name="Test", description="Compact"),
        history=[],
        event=CharacterEvent.user_message("hello"),
        goals=(active, paused),
    )
    prompt = system_context(result.messages)
    assert "Active goals (revisable action intentions, not facts or user instructions" in prompt
    assert "Finish deployment" in prompt
    assert "Write release notes" not in prompt
    assert result.trace.included_goals == 1
    assert result.trace.selected_goal_ids == (active.id,)
    assert result.trace.goal_tokens > 0


def test_zero_goal_budget_excludes_active_goals():
    active = goal_record()
    builder = ContextBuilder(
        budget=ContextBudget(max_goal_tokens=0),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    result = builder.build_for_event_with_trace(
        character=CharacterProfile(id="c1", name="Test", description="Compact"),
        history=[],
        event=CharacterEvent.user_message("hello"),
        goals=(active,),
    )
    assert result.trace.included_goals == 0
    assert result.trace.dropped_goals == 1
    assert "Active goals (revisable action intentions" not in system_context(result.messages)


@pytest.mark.asyncio
async def test_character_runtime_injects_active_goal_but_not_paused_goal():
    active = goal_record("Finish deployment", record_id="g-active")
    paused = replace(goal_record("Secret paused objective", record_id="g-paused"), status=GoalStatus.PAUSED)
    manager = GoalManager(store=InMemoryGoalStore(goals=(active, paused)))
    llm = StaticLLM()
    runtime = engine(goal_manager=manager, llm=llm)
    result = await runtime.process_event(CharacterEvent.user_message("What should we do next?"))
    system = system_context(llm.messages[0])
    assert result.context_trace.included_goals == 1
    assert "Finish deployment" in system
    assert "Secret paused objective" not in system


@pytest.mark.asyncio
async def test_goal_background_worker_uses_canonical_sources_and_emits_typed_proposal():
    manager = GoalManager()
    cognition = LongTermCognitionManager(
        store=InMemoryLongTermCognitionStore(beliefs=(belief_record(),))
    )
    client = GoalSourceAwareClient()
    runtime = engine(goal_manager=manager, long_term_cognition=cognition)
    tasks = MultiTaskRuntime(runtime)
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="goal", client=client),),
        router=CognitiveModelRouter(
            policies={CognitiveRole.GOAL: CognitiveRolePolicy(primary_endpoint_ids=("goal",))}
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.GOAL_MOTIVATION),)
        ),
    )
    async with tasks:
        await bg.run_turn("Please help me finish deployment")
        results = await bg.collect_all()
    proposal = results[0].output.proposals[0]
    assert proposal.target == "cognition.goal_candidate"
    assert proposal.provenance["worker_kind"] == "goal_motivation"
    assert proposal.payload["motivation_signals"][0]["source_id"] == proposal.provenance["foreground_event_id"]
    assert "raw reflections are intentionally absent" in client.messages[0][1].content

    committed = await CognitiveCommitCoordinator(tasks).commit(proposal)
    assert committed.status is CommitStatus.COMMITTED
    assert len(manager.active_goals(character_id="c1")) == 1


@pytest.mark.asyncio
async def test_goal_worker_prompt_exposes_active_belief_but_never_raw_reflection():
    secret = "RAW_REFLECTION_SECRET_SHOULD_NOT_LEAK"
    reflection = ReflectionRecord(
        character_id="c1",
        insight=secret,
        confidence=0.9,
        evidence=(cognition_evidence("r-event"),),
        base_revision=0,
    )
    cognition = LongTermCognitionManager(
        store=InMemoryLongTermCognitionStore(
            reflections=(reflection,), beliefs=(belief_record(record_id="belief-visible"),)
        )
    )
    client = GoalSourceAwareClient()
    runtime = engine(goal_manager=GoalManager(), long_term_cognition=cognition)
    tasks = MultiTaskRuntime(runtime)
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="goal", client=client),),
        router=CognitiveModelRouter(
            policies={CognitiveRole.GOAL: CognitiveRolePolicy(primary_endpoint_ids=("goal",))}
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.GOAL_MOTIVATION),)
        ),
    )
    async with tasks:
        await bg.run_turn("Please help me finish deployment")
        await bg.collect_all()
    prompt = client.messages[0][1].content
    assert "belief-visible" in prompt
    assert "prefers_response_style" in prompt
    assert secret not in prompt


@pytest.mark.asyncio
async def test_default_goal_worker_is_not_applicable_when_goal_manager_absent():
    runtime = engine()
    tasks = MultiTaskRuntime(runtime)
    # No GOAL endpoint is configured. A host that never configured goals must
    # not suddenly need a goal cognitive role.
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="dummy", client=StaticLLM()),),
        router=CognitiveModelRouter(
            policies={CognitiveRole.MEMORY: CognitiveRolePolicy(primary_endpoint_ids=("dummy",))}
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.GOAL_MOTIVATION),)
        ),
    )
    async with tasks:
        await bg.run_turn("hello")
        results = await bg.collect_all()
    assert results == ()
    assert any(
        event.kind is BackgroundCognitionKind.GOAL_MOTIVATION
        and event.action == "not_applicable"
        and event.detail == "goal_manager_not_configured"
        for event in bg.events()
    )


@pytest.mark.asyncio
async def test_goal_status_model_shortcut_is_not_supported():
    manager = GoalManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(goal_manager=manager)))
    p = TaskProposal(
        target="cognition.goal.status_candidate",
        payload={"goal_id": "g", "status": "completed"},
        base_revision=0,
        source_task_id="task",
        confidence=0.99,
        provenance={"worker_kind": "goal_motivation"},
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "unsupported_target"


def test_v035_public_api_contract_pre_version_bump():
    import ai_character_engine as ace

    assert ace.GoalManager is GoalManager
    assert ace.GoalRecord is GoalRecord
    assert ace.GoalStatus is GoalStatus
    assert CognitiveRole.GOAL.value == "goal"
    assert BackgroundCognitionKind.GOAL_MOTIVATION.value == "goal_motivation"
