from __future__ import annotations

from ai_character_engine._version import VERSION
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.commit import (
    CognitiveCommitCoordinator,
    CommitNextAction,
    CommitStatus,
    StalePolicy,
)
from ai_character_engine.cognition import (
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
    CognitiveModelRouter,
)
from ai_character_engine.llm import ModelEndpoint
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state.mood import MOOD_TURN_KEY
from ai_character_engine.state.models import StatePatch
from ai_character_engine.tasks import MultiTaskRuntime, TaskProposal


class StaticLLM:
    def __init__(self, text: str = "foreground reply") -> None:
        self.text = text

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=self.text, model="foreground")


class JSONClient:
    def __init__(self, text: str, model: str = "worker") -> None:
        self.text = text
        self.model = model

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=self.text, model=self.model)


class GateLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.started.set()
        await self.release.wait()
        return LLMResponse(text="later foreground", model="gate")


class FailOnceStore(InMemoryMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def add(self, record):
        super().add(record)
        if not self.failed:
            self.failed = True
            raise OSError("synthetic write failure")


def runtime(*, llm=None, memory_manager=None) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c1", name="Mei", description="Stay in character."),
        llm=llm or StaticLLM(),
        memory_manager=memory_manager or MemoryManager(),
    )


def proposal(
    target: str,
    payload: dict,
    *,
    revision: int = 0,
    confidence: float = 0.9,
    worker_kind: str | None = None,
    evidence_type: str = "asserted_fact",
    event_id: str = "evt-1",
    created_at: datetime | None = None,
) -> TaskProposal:
    return TaskProposal(
        target=target,
        payload=payload,
        base_revision=revision,
        source_task_id="task-1",
        confidence=confidence,
        provenance={
            "worker_kind": worker_kind or {
                "memory.append_candidate": "memory_extraction",
                "state.emotion_candidate": "emotion_analysis",
                "memory.conversation_summary_candidate": "conversation_summary",
                "cognition.reflection_candidate": "reflection",
                "context.vision_interpretation_candidate": "vision_interpretation",
            }.get(target, "test"),
            "foreground_event_id": event_id,
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "evidence_type": evidence_type,
        },
        created_at=created_at or datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_fresh_asserted_memory_candidate_commits_with_audit_metadata():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    p = proposal(
        "memory.append_candidate",
        {"summary": "User prefers tea", "kind": "preference", "importance": 0.8},
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.COMMITTED
    assert result.commit_sequence == 1
    assert tasks.revision == 0
    records = engine.memory_manager.store.list_for_character(engine.memory_scope_id)
    assert len(records) == 1
    assert records[0].summary == "User prefers tea"
    assert records[0].metadata["proposal_id"] == p.id
    assert records[0].metadata["evidence_type"] == "asserted_fact"


@pytest.mark.asyncio
async def test_memory_candidate_requires_direct_asserted_fact_provenance():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal(
        "memory.append_candidate",
        {"summary": "Quoted name is Bob"},
        evidence_type="quoted_reference",
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "memory_candidate_requires_asserted_fact"
    assert engine.memory_manager.store.list_for_character(engine.memory_scope_id) == []


@pytest.mark.asyncio
async def test_what_the_character_said_about_herself_needs_her_own_statement():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    payload = {"summary": "Mei drinks jasmine tea", "kind": "preference"}
    said = proposal(
        "memory.self_candidate",
        payload,
        worker_kind="self_memory_extraction",
        evidence_type="character_statement",
    )
    from_the_user = proposal(
        "memory.self_candidate",
        {**payload, "summary": "Mei has a cat"},
        worker_kind="self_memory_extraction",
        evidence_type="asserted_fact",
    )
    again = proposal(
        "memory.self_candidate",
        payload,
        worker_kind="self_memory_extraction",
        evidence_type="character_statement",
        event_id="evt-2",
    )
    results = [await commits.commit(item) for item in (said, from_the_user, again)]
    assert [result.status for result in results] == [
        CommitStatus.COMMITTED,
        CommitStatus.REJECTED,
        CommitStatus.DUPLICATE,
    ]
    assert results[1].reason == "self_memory_candidate_requires_character_statement"
    records = engine.memory_manager.store.list_for_character(engine.memory_scope_id)
    assert [record.summary for record in records] == ["Mei drinks jasmine tea"]
    assert records[0].tags == ("background_cognition", "self_memory_extraction")


@pytest.mark.asyncio
async def test_low_confidence_proposal_is_rejected_before_mutation():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal("memory.append_candidate", {"summary": "weak"}, confidence=0.2)
    result = await commits.commit(p)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "confidence_below_threshold"


@pytest.mark.asyncio
async def test_stale_memory_proposal_requests_explicit_rebase_then_can_commit():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    await tasks.run_turn("advance")
    commits = CognitiveCommitCoordinator(tasks)
    old = proposal("memory.append_candidate", {"summary": "User prefers tea"}, revision=0)
    stale = await commits.commit(old)
    assert stale.status is CommitStatus.STALE
    assert stale.next_action is CommitNextAction.REBASE
    rebased = commits.rebase(old, reason="host confirmed the direct assertion is still current")
    assert rebased.base_revision == 1
    assert rebased.id != old.id
    applied = await commits.commit(rebased)
    assert applied.status is CommitStatus.COMMITTED


@pytest.mark.asyncio
async def test_stale_emotion_proposal_requests_rerun_not_rebase():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    await tasks.run_turn("advance")
    commits = CognitiveCommitCoordinator(tasks)
    p = proposal(
        "state.emotion_candidate",
        {"emotion": "frustrated", "intensity": 0.8},
        revision=0,
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.STALE
    assert result.next_action is CommitNextAction.RERUN
    with pytest.raises(ValueError):
        commits.rebase(p, reason="unsafe")


@pytest.mark.asyncio
async def test_emotion_analysis_updates_observed_user_emotion_not_character_emotion():
    engine = runtime()
    original = engine.state.emotion
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal(
        "state.emotion_candidate",
        {"emotion": "frustrated", "intensity": 0.8},
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.COMMITTED
    assert engine.state.emotion == original
    assert engine.state.custom["observed_user_emotion"]["emotion"] == "frustrated"
    assert engine.state.custom["observed_user_emotion"]["intensity"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_conversation_summary_commits_as_typed_memory():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal(
        "memory.conversation_summary_candidate",
        {"summary": "They discussed the project and tea."},
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.COMMITTED
    record = engine.memory_manager.store.list_for_character(engine.memory_scope_id)[0]
    assert record.kind == "conversation_summary"
    assert record.metadata["evidence_type"] == "conversation_summary"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,payload",
    [
        ("cognition.reflection_candidate", {"insight": "Be cautious."}),
        ("context.vision_interpretation_candidate", {"interpretation": "A cup is visible."}),
    ],
)
async def test_interpretation_only_targets_require_review_and_do_not_mutate(target, payload):
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    before = engine.state.snapshot()
    result = await commits.commit(proposal(target, payload))
    assert result.status is CommitStatus.REVIEW_REQUIRED
    assert result.next_action is CommitNextAction.MANUAL_REVIEW
    assert engine.state.snapshot() == before
    assert engine.memory_manager.store.list_for_character(engine.memory_scope_id) == []


@pytest.mark.asyncio
async def test_same_proposal_is_exactly_once_idempotent():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal("memory.append_candidate", {"summary": "User prefers tea"})
    first = await commits.commit(p)
    second = await commits.commit(p)
    assert second == first
    assert len(engine.memory_manager.store.list_for_character(engine.memory_scope_id)) == 1
    assert commits.commit_sequence == 1


@pytest.mark.asyncio
async def test_semantic_duplicate_with_new_proposal_id_is_not_written_twice():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    first = proposal("memory.append_candidate", {"summary": "User prefers tea"})
    second = proposal("memory.append_candidate", {"summary": "  user PREFERS   tea  "})
    assert (await commits.commit(first)).status is CommitStatus.COMMITTED
    duplicate = await commits.commit(second)
    assert duplicate.status is CommitStatus.DUPLICATE
    assert len(engine.memory_manager.store.list_for_character(engine.memory_scope_id)) == 1


@pytest.mark.asyncio
async def test_same_revision_different_summaries_conflict_first_write_wins():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    a = proposal("memory.conversation_summary_candidate", {"summary": "Summary A"})
    b = proposal("memory.conversation_summary_candidate", {"summary": "Summary B"})
    assert (await commits.commit(a)).status is CommitStatus.COMMITTED
    conflict = await commits.commit(b)
    assert conflict.status is CommitStatus.CONFLICT
    assert conflict.next_action is CommitNextAction.MANUAL_REVIEW
    assert len(engine.memory_manager.store.list_for_character(engine.memory_scope_id)) == 1


@pytest.mark.asyncio
async def test_retryable_memory_failure_rolls_back_and_retry_commits_once():
    store = FailOnceStore()
    engine = runtime(memory_manager=MemoryManager(store=store))
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal("memory.append_candidate", {"summary": "User prefers tea"})
    failed = await commits.commit(p)
    assert failed.status is CommitStatus.RETRYABLE_ERROR
    assert failed.next_action is CommitNextAction.RETRY
    assert store.list_for_character(engine.memory_scope_id) == []
    applied = await commits.retry(p)
    assert applied.status is CommitStatus.COMMITTED
    assert len(store.list_for_character(engine.memory_scope_id)) == 1


@pytest.mark.asyncio
async def test_foreground_and_commit_share_authority_lock_and_recheck_freshness():
    gate = GateLLM()
    engine = runtime(llm=gate)
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    p = proposal("memory.append_candidate", {"summary": "User prefers tea"}, revision=0)

    foreground = asyncio.create_task(tasks.run_turn("later foreground"))
    await gate.started.wait()
    commit_task = asyncio.create_task(commits.commit(p))
    await asyncio.sleep(0)
    assert commit_task.done() is False
    gate.release.set()
    await foreground
    result = await commit_task
    assert tasks.revision == 1
    assert result.status is CommitStatus.STALE
    records = engine.memory_manager.store.list_for_character(engine.memory_scope_id)
    assert all(record.summary != "User prefers tea" for record in records)


@pytest.mark.asyncio
async def test_old_proposal_expires_even_without_revision_change():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal(
        "memory.append_candidate",
        {"summary": "old fact"},
        created_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.STALE
    assert result.reason == "proposal_expired"


@pytest.mark.asyncio
async def test_unexpected_worker_kind_and_unsupported_target_are_rejected():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    wrong_worker = proposal(
        "memory.append_candidate",
        {"summary": "fact"},
        worker_kind="reflection",
    )
    assert (await commits.commit(wrong_worker)).reason == "unexpected_worker_kind"
    unsupported = proposal("state.magic_candidate", {"x": 1}, worker_kind="test")
    result = await commits.commit(unsupported)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "unsupported_target"


@pytest.mark.asyncio
async def test_two_safe_sibling_commits_share_foreground_revision_but_get_commit_sequence():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    memory = proposal("memory.append_candidate", {"summary": "User prefers tea"})
    emotion = proposal(
        "state.emotion_candidate", {"emotion": "happy", "intensity": 0.6}, event_id="evt-2"
    )
    a = await commits.commit(memory)
    b = await commits.commit(emotion)
    assert a.commit_sequence == 1
    assert b.commit_sequence == 2
    assert tasks.revision == 0


@pytest.mark.asyncio
async def test_real_background_memory_proposal_stays_non_authoritative_until_coordinator_commit():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    client = JSONClient(
        '{"items":[{"summary":"User prefers tea","kind":"preference","importance":0.8,"confidence":0.93,'
        '"evidence":"I prefer tea"}],"confidence":0.93,"evidence":["I prefer tea"]}'
    )
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="memory", client=client),),
        router=CognitiveModelRouter(
            policies={
                CognitiveRole.MEMORY: CognitiveRolePolicy(primary_endpoint_ids=("memory",))
            }
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),)
        ),
    )
    commits = CognitiveCommitCoordinator(tasks)
    async with tasks:
        await bg.run_turn("I prefer tea")
        results = await bg.collect_all()
        before_records = engine.memory_manager.store.list_for_character(engine.memory_scope_id)
        assert all(record.summary != "User prefers tea" for record in before_records)
        decisions = await commits.commit_task_results(results)
    assert len(decisions) == 1
    assert decisions[0].status is CommitStatus.CONFLICT
    assert decisions[0].reason == "foreground_memory_already_exists_for_source_event"
    after_records = engine.memory_manager.store.list_for_character(engine.memory_scope_id)
    assert len(after_records) == len(before_records)
    assert all(record.summary != "User prefers tea" for record in after_records)


@pytest.mark.asyncio
async def test_new_coordinator_instance_detects_existing_memory_duplicate_from_authority_store():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    first = CognitiveCommitCoordinator(tasks)
    p1 = proposal("memory.append_candidate", {"summary": "User prefers tea"}, event_id="event-z")
    assert (await first.commit(p1)).status is CommitStatus.COMMITTED
    second = CognitiveCommitCoordinator(tasks)
    p2 = proposal("memory.append_candidate", {"summary": "user prefers tea"}, event_id="event-other")
    result = await second.commit(p2)
    assert result.status is CommitStatus.DUPLICATE
    assert len(engine.memory_manager.store.list_for_character(engine.memory_scope_id)) == 1


@pytest.mark.asyncio
async def test_new_coordinator_instance_detects_state_conflict_from_authoritative_state():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    first = CognitiveCommitCoordinator(tasks)
    p1 = proposal("state.emotion_candidate", {"emotion": "happy", "intensity": 0.6})
    assert (await first.commit(p1)).status is CommitStatus.COMMITTED
    second = CognitiveCommitCoordinator(tasks)
    p2 = proposal("state.emotion_candidate", {"emotion": "sad", "intensity": 0.8})
    result = await second.commit(p2)
    assert result.status is CommitStatus.CONFLICT
    assert engine.state.custom["observed_user_emotion"]["emotion"] == "happy"


@pytest.mark.asyncio
async def test_emotion_candidate_rejects_quoted_reference_evidence():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal(
        "state.emotion_candidate",
        {"emotion": "angry", "intensity": 0.9},
        evidence_type="quoted_reference",
    )
    result = await commits.commit(p)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "emotion_candidate_has_unsafe_evidence_type"


@pytest.mark.asyncio
async def test_what_was_decided_long_ago_is_let_go():
    """Every proposal ever seen was kept to answer a second attempt with the
    first result. A proposal is tried again within moments, not after
    hundreds of others."""
    current = runtime(memory_manager=MemoryManager())
    tasks = MultiTaskRuntime(current)
    coordinator = CognitiveCommitCoordinator(tasks, event_history=4)
    proposals = [
        proposal(
            "memory.append_candidate",
            {"summary": f"The user said thing number {n}", "kind": "fact", "importance": 0.6},
            event_id=f"evt-{n}",
        )
        for n in range(10)
    ]
    async with tasks:
        results = [await coordinator.commit(item) for item in proposals]

    assert all(result.status is CommitStatus.COMMITTED for result in results)
    assert coordinator.result_for(proposals[-1].id) is results[-1]
    assert coordinator.result_for(proposals[0].id) is None
    assert coordinator.remembered_decisions == 4


def test_package_version():
    import ai_character_engine as ace
    assert ace.__version__ == VERSION


@pytest.mark.asyncio
async def test_observed_user_emotion_keeps_valence_and_stance_when_the_worker_supplies_them():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    result = await commits.commit(
        proposal(
            "state.emotion_candidate",
            {"emotion": "grateful", "intensity": 0.8, "valence": 0.9, "stance": 1.0},
        )
    )
    assert result.status is CommitStatus.COMMITTED
    observed = engine.state.custom["observed_user_emotion"]
    assert observed["valence"] == pytest.approx(0.9)
    assert observed["stance"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_observed_user_emotion_has_no_valence_or_stance_when_the_worker_omits_them():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    await commits.commit(
        proposal(
            "state.emotion_candidate",
            {"emotion": "frustrated", "intensity": 0.8},
        )
    )
    observed = engine.state.custom["observed_user_emotion"]
    assert "valence" not in observed and "stance" not in observed


@pytest.mark.asyncio
async def test_several_facts_from_one_message_are_all_committed():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    name = await commits.commit(
        proposal("memory.append_candidate", {"summary": "User is called Dawn", "kind": "name", "importance": 0.9})
    )
    pet = await commits.commit(
        proposal("memory.append_candidate", {"summary": "User has a cat called Bun", "kind": "pet", "importance": 0.8})
    )
    assert (name.status, pet.status) == (CommitStatus.COMMITTED, CommitStatus.COMMITTED)
    stored = [r.summary for r in engine.memory_manager.store.list_for_character(engine.memory_scope_id)]
    assert stored == ["User is called Dawn", "User has a cat called Bun"]


@pytest.mark.asyncio
@pytest.mark.parametrize("garbage", [float("nan"), True, "high", None])
async def test_observed_user_emotion_never_stores_a_score_it_cannot_read(garbage):
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    await commits.commit(
        proposal(
            "state.emotion_candidate",
            {"emotion": "grateful", "intensity": 0.8, "valence": garbage, "stance": garbage},
        )
    )
    observed = engine.state.custom["observed_user_emotion"]
    assert "valence" not in observed and "stance" not in observed


def mood_proposal(mood="sad", intensity=0.7, *, turn_ended_at=100.0, revision=0, confidence=0.9):
    return TaskProposal(
        target="state.mood_candidate",
        payload={"mood": mood, "intensity": intensity},
        base_revision=revision,
        source_task_id="task-mood",
        confidence=confidence,
        provenance={
            "worker_kind": "character_mood",
            "foreground_event_id": "evt-1",
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "evidence_type": "asserted_fact",
            "turn_ended_at": turn_ended_at,
        },
    )


@pytest.mark.asyncio
async def test_her_mood_is_written_with_its_strength_and_the_time():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    commits.clock = lambda: 5000.0
    result = await commits.commit(mood_proposal())
    assert result.status is CommitStatus.COMMITTED
    state = engine.state
    assert (state.emotion, state.mood_intensity, state.mood_updated_at) == ("sad", 0.7, 5000.0)
    assert state.custom[MOOD_TURN_KEY] == 100.0


@pytest.mark.asyncio
async def test_a_mood_outside_her_vocabulary_is_refused():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    result = await commits.commit(mood_proposal(mood="開心"))
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "mood_not_in_vocabulary")
    assert engine.state.emotion == "neutral"


@pytest.mark.asyncio
async def test_an_unsure_reading_of_her_mood_is_refused():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    result = await commits.commit(mood_proposal(confidence=0.5))
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "confidence_below_threshold")


@pytest.mark.asyncio
async def test_a_reading_of_an_earlier_turn_does_not_replace_a_newer_one():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    newer = await commits.commit(mood_proposal("happy", turn_ended_at=200.0))
    older = await commits.commit(mood_proposal("sad", turn_ended_at=100.0))
    assert newer.status is CommitStatus.COMMITTED
    assert (older.status, older.reason) == (CommitStatus.STALE, "newer_mood_already_committed")
    assert engine.state.emotion == "happy"


@pytest.mark.asyncio
async def test_the_users_emotion_keeps_the_time_of_its_turn():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    p = proposal("state.emotion_candidate", {"emotion": "tired", "intensity": 0.6})
    p = replace(p, provenance={**p.provenance, "turn_ended_at": 100.0})
    assert (await commits.commit(p)).status is CommitStatus.COMMITTED
    assert engine.state.custom["observed_user_emotion"]["turn_ended_at"] == 100.0


async def mood_after(*readings, clock=5000.0, start=("sad", 0.8, 5000.0)):
    """Her mood as stored after ``readings`` (mood, intensity, turn) were
    committed one after another, starting from ``start``."""
    engine = runtime()
    engine.state.apply(
        StatePatch(emotion=start[0], mood_intensity=start[1], mood_updated_at=start[2])
    )
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    commits.clock = lambda: clock
    results = [
        await commits.commit(mood_proposal(mood, intensity, turn_ended_at=turn))
        for mood, intensity, turn in readings
    ]
    state = engine.state
    return results, (state.emotion, state.mood_intensity, state.mood_updated_at), state.custom


@pytest.mark.asyncio
async def test_a_neutral_reading_leaves_a_strong_mood_to_fade_by_itself():
    results, mood, custom = await mood_after(("neutral", 0.0, 100.0), start=("sad", 0.8, 4000.0))
    assert results[0].status is CommitStatus.COMMITTED
    assert mood == ("sad", 0.8, 4000.0)
    # Read for this turn all the same: the turn's rules do not override it.
    assert custom[MOOD_TURN_KEY] == 100.0


@pytest.mark.asyncio
async def test_a_weaker_reading_of_another_mood_leaves_her_mood():
    results, mood, custom = await mood_after(("worried", 0.5, 100.0))
    assert results[0].status is CommitStatus.COMMITTED
    assert mood == ("sad", 0.8, 5000.0)
    assert custom[MOOD_TURN_KEY] == 100.0


@pytest.mark.asyncio
async def test_a_stronger_reading_of_another_mood_replaces_hers():
    _, mood, _ = await mood_after(("happy", 0.9, 100.0), clock=6000.0)
    assert mood == ("happy", 0.9, 6000.0)


@pytest.mark.asyncio
async def test_the_same_mood_again_is_the_stronger_of_the_two_as_of_now():
    _, mood, _ = await mood_after(("sad", 0.3, 100.0), clock=5300.0)
    # 0.8 five minutes ago is 0.4 now; the reading said 0.3.
    assert mood[0] == "sad" and mood[2] == 5300.0
    assert mood[1] == pytest.approx(0.4)
    _, mood, _ = await mood_after(("sad", 0.9, 100.0), clock=5300.0)
    assert mood == ("sad", 0.9, 5300.0)


@pytest.mark.asyncio
async def test_a_reading_blends_with_her_mood_as_faded_by_the_coordinators_settings():
    async def mood_set(half_life):
        engine = runtime()
        engine.state.apply(StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0))
        commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
        commits.clock = lambda: 1060.0
        if half_life is not None:
            commits.mood_half_life_seconds = half_life
        await commits.commit(mood_proposal("worried", 0.5))
        return engine.state.emotion

    assert await mood_set(None) == "sad"  # five-minute half-life: still 0.70
    assert await mood_set(60.0) == "worried"  # 0.4 left


@pytest.mark.asyncio
async def test_a_late_reading_of_an_earlier_turn_is_still_stale_when_the_newer_one_kept_her_mood():
    results, mood, _ = await mood_after(("neutral", 0.0, 200.0), ("happy", 1.0, 100.0))
    assert (results[1].status, results[1].reason) == (
        CommitStatus.STALE,
        "newer_mood_already_committed",
    )
    assert mood == ("sad", 0.8, 5000.0)


@pytest.mark.asyncio
async def test_an_observation_moved_onto_a_later_turn_keeps_the_turn_it_observed():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    commits.policies["state.emotion_candidate"] = replace(
        commits.policies["state.emotion_candidate"], stale_policy=StalePolicy.ALLOW_MANUAL_REBASE
    )
    observed = proposal("state.emotion_candidate", {"emotion": "glad", "intensity": 0.6})
    await tasks.run_turn("one more")  # a turn in between
    moved = commits.rebase(observed, reason="still valid")
    assert (await commits.commit(moved)).status is CommitStatus.COMMITTED
    value = engine.state.custom["observed_user_emotion"]
    assert value["base_revision"] == moved.base_revision != observed.base_revision
    assert value["turn_revision"] == observed.base_revision


@pytest.mark.asyncio
async def test_an_observation_on_its_own_turn_is_of_that_turn():
    engine = runtime()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    await commits.commit(proposal("state.emotion_candidate", {"emotion": "glad", "intensity": 0.6}))
    value = engine.state.custom["observed_user_emotion"]
    assert value["turn_revision"] == value["base_revision"] == 0


# --- reply check ----------------------------------------------------------------------


def reply_note(issues=None, *, revision=0):
    return TaskProposal(
        target="context.reply_note_candidate",
        payload={
            "issues": issues
            if issues is not None
            else [{"kind": "repeated", "evidence": "Hello.", "fix": "Open differently."}]
        },
        base_revision=revision,
        source_task_id="task-reply",
        confidence=1.0,
        provenance={"worker_kind": "reply_check", "reply": "Hello. How are you?"},
    )


@pytest.mark.asyncio
async def test_a_note_on_her_reply_is_committed_without_writing_anything():
    engine = runtime()
    before = engine.state.snapshot()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    result = await commits.commit(reply_note())
    assert result.status is CommitStatus.COMMITTED
    assert engine.state.snapshot() == before
    assert engine.memory_manager.store.list_for_character(engine.memory_scope_id) == []


@pytest.mark.asyncio
async def test_a_note_on_a_reply_before_the_last_turn_is_stale_and_not_moved_on():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    async with tasks:
        await tasks.run_foreground_turn(lambda: engine.run_turn("hi"))
    result = await commits.commit(reply_note(revision=0))
    assert (result.status, result.reason) == (CommitStatus.STALE, "foreground_revision_changed")
    with pytest.raises(ValueError):
        commits.rebase(reply_note(revision=0), reason="late")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issues",
    [[], [{"kind": "tone", "evidence": "Hello.", "fix": "Be nicer."}], [{"kind": "repeated", "fix": " "}]],
    ids=["none", "off the list", "no fix"],
)
async def test_a_note_without_a_slip_on_the_list_is_refused(issues):
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime()))
    result = await commits.commit(reply_note(issues))
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "reply_note_without_issue")


# --- user state and diary ------------------------------------------------------------


def user_state(payload=None, *, revision=0):
    return TaskProposal(
        target="state.user_state_candidate",
        payload=payload
        if payload is not None
        else {
            "energy": "low",
            "mood_trend": "down",
            "concerns": ["work never ends"],
            "evidence": ["I am so tired, work never ends"],
        },
        base_revision=revision,
        source_task_id="task-state",
        confidence=1.0,
        provenance={"worker_kind": "user_state"},
    )


@pytest.mark.asyncio
async def test_how_the_user_has_been_is_written_over_what_was_there_dated_by_the_clock():
    engine = runtime()
    engine.state.custom["user_state"] = {"energy": "high", "mood_trend": "up", "concerns": ["x"]}
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    commits.clock = lambda: 5000.0
    result = await commits.commit(user_state())
    assert result.status is CommitStatus.COMMITTED
    stored = engine.state.custom["user_state"]
    assert {key: stored[key] for key in ("energy", "mood_trend", "concerns", "evidence")} == {
        "energy": "low",
        "mood_trend": "down",
        "concerns": ["work never ends"],
        "evidence": ["I am so tired, work never ends"],
    }
    assert stored["updated_at"] == 5000.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"energy": "exhausted"}, {"mood_trend": "sideways"}, {"concerns": ["a", "b", "c", "d"]},
     {"concerns": "work"}, {"evidence": ["1", "2", "3", "4"]}],
    ids=["energy", "trend", "four concerns", "concerns not a list", "four quotes"],
)
async def test_a_user_state_off_its_vocabulary_is_refused(change):
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime()))
    proposal = user_state()
    result = await commits.commit(replace(proposal, payload={**proposal.payload, **change}))
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "user_state_off_vocabulary")


def diary_entry(payload=None, *, revision=0):
    return TaskProposal(
        target="memory.diary_candidate",
        payload=payload
        if payload is not None
        else {
            "date": "2026-10-06",
            "text": "Dawn came by.",
            "evidence": ["The user's name is Dawn."],
            "conversation_ids": ["a"],
            "until": 100.0,
        },
        base_revision=revision,
        source_task_id="task-diary",
        confidence=1.0,
        provenance={"worker_kind": "diary"},
    )


@pytest.mark.asyncio
async def test_her_diary_entry_is_committed_without_writing_anything():
    engine = runtime()
    before = engine.state.snapshot()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    result = await commits.commit(diary_entry())
    assert result.status is CommitStatus.COMMITTED
    assert engine.state.snapshot() == before
    assert engine.memory_manager.store.list_for_character(engine.memory_scope_id) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"text": " "}, {"evidence": []}, {"evidence": "x"}])
async def test_a_diary_entry_without_text_or_evidence_is_refused(change):
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime()))
    proposal = diary_entry()
    result = await commits.commit(replace(proposal, payload={**proposal.payload, **change}))
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "diary_without_evidence")


@pytest.mark.asyncio
async def test_a_diary_entry_written_while_she_talked_on_can_still_be_kept():
    engine = runtime()
    tasks = MultiTaskRuntime(engine)
    commits = CognitiveCommitCoordinator(tasks)
    async with tasks:
        await tasks.run_foreground_turn(lambda: engine.run_turn("hi"))
    stale = await commits.commit(diary_entry(revision=0))
    assert stale.next_action is CommitNextAction.REBASE
    rebased = commits.rebase(diary_entry(revision=0), reason="late")
    assert (await commits.commit(rebased)).status is CommitStatus.COMMITTED
