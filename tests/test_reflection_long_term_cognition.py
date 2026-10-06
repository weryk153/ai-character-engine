from __future__ import annotations

from ai_character_engine._version import VERSION
import json
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
from ai_character_engine.llm import ModelEndpoint
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.long_term_cognition import (
    BeliefClaim,
    BeliefRecord,
    BeliefStatus,
    CognitionEvidenceRef,
    InMemoryLongTermCognitionStore,
    JsonlLongTermCognitionStore,
    LongTermCognitionManager,
    ReflectionRecord,
)
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.tasks import MultiTaskRuntime, TaskProposal
from tests.fakes import system_context


class StaticLLM:
    def __init__(self, text: str = "foreground reply") -> None:
        self.text = text
        self.messages: list[list[Message]] = []

    async def generate(self, messages, *, tools=None):
        self.messages.append(list(messages))
        return LLMResponse(text=self.text, model="foreground")


class JSONClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def generate(self, messages, *, tools=None):
        return LLMResponse(text=json.dumps(self.payload), model="reflection-worker")


class FailOnceBeliefStore(InMemoryLongTermCognitionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_belief_replace = False

    def replace_beliefs(self, character_id, records):
        super().replace_beliefs(character_id, records)
        if self.fail_next_belief_replace:
            self.fail_next_belief_replace = False
            raise OSError("synthetic cognition write failure")


class WordTokenEstimator:
    def estimate_text(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    def estimate_message(self, message: Message) -> int:
        return 1 + self.estimate_text(message.content)

    def estimate_tools(self, tools) -> int:
        return 0


def evidence(
    source_id: str,
    *,
    evidence_type: str = "asserted_fact",
    excerpt: str = "Please keep it concise",
    confidence: float = 0.9,
) -> CognitionEvidenceRef:
    return CognitionEvidenceRef(
        source_type="event",
        source_id=source_id,
        evidence_type=evidence_type,
        excerpt=excerpt,
        confidence=confidence,
    )


def belief(
    object_value: str,
    *,
    status: BeliefStatus = BeliefStatus.ACTIVE,
    source_prefix: str = "evt",
    confidence: float = 0.9,
) -> BeliefRecord:
    refs = (evidence(f"{source_prefix}-1"), evidence(f"{source_prefix}-2"))
    return BeliefRecord(
        character_id="c1",
        claim=BeliefClaim("user", "prefers_response_style", object_value),
        confidence=confidence,
        evidence=refs,
        source_reflection_ids=("r1", "r2"),
        support_count=2,
        status=status,
    )


def engine(*, manager: LongTermCognitionManager | None = None, llm=None) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c1", name="Test", description="Stay in character."),
        llm=llm or StaticLLM(),
        long_term_cognition=manager,
    )


def reflection_proposal(
    *,
    event_id: str,
    object_value: str | None = "concise",
    insight: str = "The user may prefer concise answers.",
    evidence_type: str = "asserted_fact",
    confidence: float = 0.9,
    revision: int = 0,
    proposal_id: str | None = None,
) -> TaskProposal:
    payload: dict = {"insight": insight, "belief_candidate": None}
    if object_value is not None:
        payload["belief_candidate"] = {
            "subject": "user",
            "predicate": "prefers_response_style",
            "object": object_value,
        }
    kwargs = {}
    if proposal_id is not None:
        kwargs["id"] = proposal_id
    return TaskProposal(
        target="cognition.reflection_candidate",
        payload=payload,
        base_revision=revision,
        source_task_id=f"task-{event_id}",
        confidence=confidence,
        provenance={
            "worker_kind": "reflection",
            "foreground_event_id": event_id,
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "evidence_type": evidence_type,
            "evidence": ["Please keep it concise"],
        },
        created_at=datetime.now(UTC),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_one_reflection_is_not_enough_to_promote_belief():
    manager = LongTermCognitionManager()
    runtime = engine(manager=manager)
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime))

    result = await commits.commit(reflection_proposal(event_id="evt-1"))

    assert result.status is CommitStatus.COMMITTED
    assert len(manager.reflections(character_id="c1")) == 1
    assert manager.beliefs(character_id="c1") == ()


@pytest.mark.asyncio
async def test_two_independent_sources_promote_active_belief():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))

    await commits.commit(reflection_proposal(event_id="evt-1"))
    second = await commits.commit(reflection_proposal(event_id="evt-2"))

    assert second.status is CommitStatus.COMMITTED
    beliefs = manager.active_beliefs(character_id="c1")
    assert len(beliefs) == 1
    assert beliefs[0].claim.object == "concise"
    assert beliefs[0].support_count == 2
    assert {item.source_id for item in beliefs[0].evidence} == {"evt-1", "evt-2"}


@pytest.mark.asyncio
async def test_repeated_reflections_from_same_source_do_not_fake_independent_support():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))

    first = reflection_proposal(event_id="evt-1", insight="Maybe concise is preferred.")
    second = reflection_proposal(event_id="evt-1", insight="Concise answers seem preferred.")
    assert (await commits.commit(first)).status is CommitStatus.COMMITTED
    assert (await commits.commit(second)).status is CommitStatus.COMMITTED

    assert len(manager.reflections(character_id="c1")) == 2
    assert manager.beliefs(character_id="c1") == ()


@pytest.mark.asyncio
async def test_unsafe_quoted_evidence_can_be_reflected_but_never_promotes_belief():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))

    for index in (1, 2):
        result = await commits.commit(
            reflection_proposal(
                event_id=f"quote-{index}",
                evidence_type="quoted_reference",
            )
        )
        assert result.status is CommitStatus.COMMITTED

    assert len(manager.reflections(character_id="c1")) == 2
    assert manager.active_beliefs(character_id="c1") == ()


@pytest.mark.asyncio
async def test_conflicting_supported_values_become_contested_and_leave_context_set():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))

    await commits.commit(reflection_proposal(event_id="a1", object_value="concise"))
    await commits.commit(reflection_proposal(event_id="a2", object_value="concise"))
    assert len(manager.active_beliefs(character_id="c1")) == 1

    await commits.commit(
        reflection_proposal(
            event_id="b1",
            object_value="detailed",
            insight="The user may prefer detailed answers.",
        )
    )
    await commits.commit(
        reflection_proposal(
            event_id="b2",
            object_value="detailed",
            insight="Detailed answers may be preferred.",
        )
    )

    beliefs = manager.beliefs(character_id="c1")
    assert {item.claim.object for item in beliefs} == {"concise", "detailed"}
    assert {item.status for item in beliefs} == {BeliefStatus.CONTESTED}
    assert manager.active_beliefs(character_id="c1") == ()


@pytest.mark.asyncio
async def test_reinforcement_preserves_belief_id_and_increases_support():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))
    await commits.commit(reflection_proposal(event_id="evt-1"))
    await commits.commit(reflection_proposal(event_id="evt-2"))
    before = manager.active_beliefs(character_id="c1")[0]

    await commits.commit(reflection_proposal(event_id="evt-3"))
    after = manager.active_beliefs(character_id="c1")[0]

    assert after.id == before.id
    assert after.support_count == 3
    assert {item.source_id for item in after.evidence} == {"evt-1", "evt-2", "evt-3"}


@pytest.mark.asyncio
async def test_reflection_without_structured_claim_is_durable_but_not_forced_into_belief():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))

    result = await commits.commit(
        reflection_proposal(
            event_id="evt-1",
            object_value=None,
            insight="A future turn may need to revisit the unresolved deployment question.",
        )
    )

    assert result.status is CommitStatus.COMMITTED
    record = manager.reflections(character_id="c1")[0]
    assert record.claim is None
    assert manager.beliefs(character_id="c1") == ()


@pytest.mark.asyncio
async def test_without_a_manager_reflection_stays_review_required():
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine()))
    result = await commits.commit(reflection_proposal(event_id="evt-1"))
    assert result.status is CommitStatus.REVIEW_REQUIRED
    assert result.next_action is CommitNextAction.MANUAL_REVIEW
    assert result.reason == "long_term_cognition_not_configured"


@pytest.mark.asyncio
async def test_direct_belief_proposal_is_rejected_as_unsupported_shortcut():
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=LongTermCognitionManager())))
    direct = TaskProposal(
        target="cognition.belief_candidate",
        payload={"subject": "user", "predicate": "prefers", "object": "concise"},
        base_revision=0,
        source_task_id="task-direct",
        confidence=0.99,
        provenance={"worker_kind": "reflection"},
    )
    result = await commits.commit(direct)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "unsupported_target"


@pytest.mark.asyncio
async def test_partial_belief_candidate_is_rejected_at_commit_boundary():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))
    bad = reflection_proposal(event_id="evt-1")
    bad = TaskProposal(
        target=bad.target,
        payload={"insight": "Maybe concise", "belief_candidate": {"subject": "user"}},
        base_revision=bad.base_revision,
        source_task_id=bad.source_task_id,
        confidence=bad.confidence,
        provenance=bad.provenance,
    )
    result = await commits.commit(bad)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "invalid_belief_candidate"
    assert manager.reflections(character_id="c1") == ()


@pytest.mark.asyncio
async def test_low_confidence_reflection_is_rejected_before_cognition_write():
    manager = LongTermCognitionManager()
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))
    result = await commits.commit(reflection_proposal(event_id="evt-1", confidence=0.79))
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "confidence_below_threshold"
    assert manager.reflections(character_id="c1") == ()


@pytest.mark.asyncio
async def test_reflection_commit_rolls_back_both_stores_and_retry_succeeds():
    store = FailOnceBeliefStore()
    manager = LongTermCognitionManager(store=store)
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine(manager=manager)))
    first = reflection_proposal(event_id="evt-1")
    second = reflection_proposal(event_id="evt-2")
    assert (await commits.commit(first)).status is CommitStatus.COMMITTED

    store.fail_next_belief_replace = True
    failed = await commits.commit(second)
    assert failed.status is CommitStatus.RETRYABLE_ERROR
    assert failed.next_action is CommitNextAction.RETRY
    assert len(store.list_reflections("c1")) == 1
    assert store.list_beliefs("c1") == []

    applied = await commits.retry(second)
    assert applied.status is CommitStatus.COMMITTED
    assert len(store.list_reflections("c1")) == 2
    assert len(store.list_beliefs("c1")) == 1


@pytest.mark.asyncio
async def test_new_coordinator_detects_persisted_reflection_duplicate():
    manager = LongTermCognitionManager()
    runtime = engine(manager=manager)
    tasks = MultiTaskRuntime(runtime)
    proposal = reflection_proposal(event_id="evt-1")
    first = CognitiveCommitCoordinator(tasks)
    assert (await first.commit(proposal)).status is CommitStatus.COMMITTED

    second = CognitiveCommitCoordinator(tasks)
    duplicate = await second.commit(proposal)
    assert duplicate.status is CommitStatus.DUPLICATE
    assert len(manager.reflections(character_id="c1")) == 1


def test_jsonl_store_round_trips_reflections_and_beliefs(tmp_path):
    path = tmp_path / "cognition.jsonl"
    store = JsonlLongTermCognitionStore(path)
    manager = LongTermCognitionManager(store=store)
    for index in (1, 2):
        manager.commit_reflection(
            ReflectionRecord(
                character_id="c1",
                insight="The user may prefer concise answers.",
                confidence=0.9,
                evidence=(evidence(f"evt-{index}"),),
                claim=BeliefClaim("user", "prefers_response_style", "concise"),
                source_proposal_id=f"p-{index}",
                source_task_id=f"t-{index}",
            )
        )

    reloaded = LongTermCognitionManager(store=JsonlLongTermCognitionStore(path))
    assert len(reloaded.reflections(character_id="c1")) == 2
    beliefs = reloaded.active_beliefs(character_id="c1")
    assert len(beliefs) == 1
    assert beliefs[0].support_count == 2
    assert beliefs[0].claim.object == "concise"


def test_context_builder_includes_only_active_beliefs_and_tracks_separate_budget():
    active = belief("concise", source_prefix="active")
    contested = belief("detailed", status=BeliefStatus.CONTESTED, source_prefix="contested")
    builder = ContextBuilder(
        budget=ContextBudget(
            context_window_tokens=120,
            reserved_output_tokens=10,
            recent_history_target_tokens=0,
            max_memory_tokens=0,
            max_belief_tokens=18,
        ),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    result = builder.build_for_event_with_trace(
        character=CharacterProfile(id="c1", name="Test", description="Compact character"),
        history=[],
        event=CharacterEvent.user_message("hello"),
        beliefs=(active, contested),
    )
    system = system_context(result.messages)
    assert "Long-term beliefs (revisable hypotheses" in system
    assert "concise" in system
    assert "detailed" not in system
    assert result.trace.included_beliefs == 1
    assert result.trace.selected_belief_ids == (active.id,)
    assert result.trace.belief_tokens > 0
    assert result.trace.memory_tokens == 0


def test_zero_belief_budget_excludes_even_active_beliefs():
    active = belief("concise")
    builder = ContextBuilder(
        budget=ContextBudget(max_belief_tokens=0),
        token_estimator=WordTokenEstimator(),
        # The window is too small for the guide to the notes of the default placement.
        context_placement="turn",
    )
    result = builder.build_for_event_with_trace(
        character=CharacterProfile(id="c1", name="Test", description="Compact character"),
        history=[],
        event=CharacterEvent.user_message("hello"),
        beliefs=(active,),
    )
    assert result.trace.included_beliefs == 0
    assert result.trace.dropped_beliefs == 1
    assert "Long-term beliefs" not in system_context(result.messages)


@pytest.mark.asyncio
async def test_character_runtime_automatically_injects_active_beliefs():
    store = InMemoryLongTermCognitionStore(beliefs=(belief("concise"),))
    manager = LongTermCognitionManager(store=store)
    llm = StaticLLM()
    runtime = engine(manager=manager, llm=llm)

    result = await runtime.process_event(CharacterEvent.user_message("How should you answer?"))

    assert result.context_trace.included_beliefs == 1
    assert "provisional hypotheses" in system_context(llm.messages[0])
    assert "- belief: " in system_context(llm.messages[0])
    assert "prefers_response_style concise" in system_context(llm.messages[0])


@pytest.mark.asyncio
async def test_reflection_worker_emits_structured_belief_candidate_contract():
    runtime = engine()
    tasks = MultiTaskRuntime(runtime)
    client = JSONClient(
        {
            "insight": "The user may prefer concise answers.",
            "belief_candidate": {
                "subject": "user",
                "predicate": "prefers_response_style",
                "object": "concise",
            },
            "confidence": 0.91,
            "evidence": ["Please keep it concise"],
        }
    )
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="reflection", client=client),),
        router=CognitiveModelRouter(
            policies={
                CognitiveRole.REFLECTION: CognitiveRolePolicy(
                    primary_endpoint_ids=("reflection",)
                )
            }
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.REFLECTION,
                    every_n_revisions=1,
                ),
            )
        ),
    )
    async with tasks:
        await bg.run_turn("Please keep it concise")
        results = await bg.collect_all()
    proposal = results[0].output.proposals[0]
    assert proposal.target == "cognition.reflection_candidate"
    assert proposal.payload["belief_candidate"] == {
        "subject": "user",
        "predicate": "prefers_response_style",
        "object": "concise",
    }
    assert proposal.provenance["evidence"] == ["Please keep it concise"]


@pytest.mark.asyncio
async def test_reflection_worker_rejects_half_structured_belief_candidate():
    runtime = engine()
    tasks = MultiTaskRuntime(runtime)
    client = JSONClient(
        {
            "insight": "Maybe concise.",
            "belief_candidate": {"subject": "user", "predicate": "prefers_response_style"},
            "confidence": 0.91,
            "evidence": ["concise"],
        }
    )
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="reflection", client=client),),
        router=CognitiveModelRouter(
            policies={
                CognitiveRole.REFLECTION: CognitiveRolePolicy(
                    primary_endpoint_ids=("reflection",)
                )
            }
        ),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.REFLECTION),)
        ),
    )
    async with tasks:
        await bg.run_turn("concise")
        results = await bg.collect_all()
    assert results[0].status.value == "failed"
    assert "belief_candidate" in (results[0].error or "")


def test_cognition_evidence_independence_is_source_based_not_excerpt_based():
    a = evidence("evt-1", excerpt="first excerpt")
    b = evidence("evt-1", excerpt="second excerpt")
    c = evidence("evt-2", excerpt="third excerpt")
    assert a.independence_key == b.independence_key
    assert a.independence_key != c.independence_key


def test_public_api_and_version():
    import ai_character_engine as ace

    assert ace.__version__ == VERSION
    assert ace.LongTermCognitionManager is LongTermCognitionManager
    assert ace.BeliefClaim is BeliefClaim


def _thought_proposal(**provenance):
    return TaskProposal(
        target="cognition.reflection_candidate",
        payload={"insight": "I think they are worn out", "belief_candidate": None},
        base_revision=0,
        source_task_id="task-1",
        confidence=0.9,
        provenance={
            "worker_kind": "reflection",
            "foreground_event_id": "event-1",
            "evidence_type": "asserted_fact",
            "evidence": ["why do you never listen?", "I am exhausted today"],
            **provenance,
        },
    )


def test_each_quote_of_a_thought_keeps_the_type_of_its_own_line():
    """A question of the user's went in as an asserted fact because the latest
    line was one, and asserted facts can become beliefs."""
    from ai_character_engine.commit.coordinator import _reflection_evidence

    refs = _reflection_evidence(
        _thought_proposal(evidence_types=["user_question", "asserted_fact"])
    )
    assert [(ref.excerpt, ref.evidence_type) for ref in refs] == [
        ("why do you never listen?", "user_question"),
        ("I am exhausted today", "asserted_fact"),
    ]


def test_a_thought_proposed_without_types_per_quote_takes_the_latest_lines_type():
    from ai_character_engine.commit.coordinator import _reflection_evidence

    refs = _reflection_evidence(_thought_proposal())
    assert [ref.evidence_type for ref in refs] == ["asserted_fact", "asserted_fact"]
