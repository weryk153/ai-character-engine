from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from ai_character_engine import (
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionResult,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CharacterEvent,
    CharacterProfile,
    CharacterRuntime,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
    ModelEndpoint,
    MultiTaskRuntime,
    MultiTaskRuntimeConfig,
    TaskPriority,
    TaskStatus,
)
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.tasks.models import TaskProposal


class ForegroundClient:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="foreground reply", model="foreground")


class RoleJSONClient:
    def __init__(self, payload: dict, *, name: str, gate: asyncio.Event | None = None):
        self.payload = payload
        self.name = name
        self.calls = 0
        self.gate = gate
        self.started = asyncio.Event()
        self.max_active = 0
        self.active = 0

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            return LLMResponse(text=json.dumps(self.payload, ensure_ascii=False), model=self.name)
        finally:
            self.active -= 1


class BadJSONClient:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="not json", model="bad")


def character() -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=ForegroundClient(),
    )


def model_runtime(clients: dict[CognitiveRole, object]) -> CognitiveModelRuntime:
    endpoints = []
    policies = {}
    for role, client in clients.items():
        endpoint_id = f"{role.value}-model"
        endpoints.append(ModelEndpoint(endpoint_id=endpoint_id, client=client))
        policies[role] = CognitiveRolePolicy(primary_endpoint_ids=(endpoint_id,))
    return CognitiveModelRuntime(
        endpoints=tuple(endpoints),
        router=CognitiveModelRouter(policies=policies),
    )


def all_clients(*, gate: asyncio.Event | None = None):
    return {
        CognitiveRole.MEMORY: RoleJSONClient(
            {
                "items": [
                    {
                        "summary": "User prefers tea",
                        "kind": "preference",
                        "importance": 0.8,
                        "confidence": 0.9,
                    }
                ],
                "confidence": 0.85,
                "evidence": ["I prefer tea"],
            },
            name="memory",
            gate=gate,
        ),
        CognitiveRole.EMOTION: RoleJSONClient(
            {"emotion": "frustrated", "intensity": 0.7, "confidence": 0.8, "evidence": ["work is annoying"]},
            name="emotion",
            gate=gate,
        ),
        CognitiveRole.SUMMARY: RoleJSONClient(
            {"summary": "The user discussed work and tea.", "confidence": 0.9, "evidence": ["recent turns"]},
            name="summary",
            gate=gate,
        ),
        CognitiveRole.REFLECTION: RoleJSONClient(
            {"insight": "Ask about the unresolved work issue later.", "confidence": 0.75, "evidence": ["unresolved issue"]},
            name="reflection",
            gate=gate,
        ),
        CognitiveRole.VISION: RoleJSONClient(
            {"interpretation": "A red cup is on the desk.", "tags": ["cup", "desk"], "confidence": 0.88, "evidence": ["red cup"]},
            name="vision",
            gate=gate,
        ),
    }


@pytest.mark.asyncio
async def test_memory_worker_emits_typed_non_authoritative_proposal_with_provenance():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),)
        ),
    )
    async with tasks:
        await bg.run_turn("I prefer tea")
        results = await bg.collect_all()
    result = results[0]
    assert result.status is TaskStatus.SUCCEEDED
    assert isinstance(result.output.value, BackgroundCognitionResult)
    proposal = result.output.proposals[0]
    assert proposal.target == "memory.append_candidate"
    assert proposal.base_revision == 1
    assert proposal.source_task_id == result.task_id
    assert proposal.confidence == pytest.approx(0.9)
    assert proposal.provenance["worker_kind"] == "memory_extraction"
    assert proposal.provenance["foreground_event_type"] == "user_message"
    assert proposal.provenance["evidence_type"] == "asserted_fact"
    assert runtime.history[-1].content == "foreground reply"
    assert runtime.state.trust == 50


def test_task_proposal_stale_contract_and_validation():
    proposal = TaskProposal(
        target="memory.append_candidate",
        payload={"x": 1},
        base_revision=2,
        confidence=0.5,
        provenance={"source": "test"},
    )
    assert proposal.is_stale(2) is False
    assert proposal.is_stale(3) is True
    with pytest.raises(ValueError):
        proposal.is_stale(-1)
    with pytest.raises(ValueError):
        TaskProposal(target="x", payload={}, base_revision=0, confidence=1.1)


@pytest.mark.asyncio
async def test_foreground_does_not_wait_for_background_models():
    gate = asyncio.Event()
    clients = all_clients(gate=gate)
    runtime = character()
    tasks = MultiTaskRuntime(runtime, config=MultiTaskRuntimeConfig(worker_count=2))
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({
            CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY],
            CognitiveRole.EMOTION: clients[CognitiveRole.EMOTION],
        }),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),
                BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),
            )
        ),
    )
    async with tasks:
        result = await asyncio.wait_for(bg.run_turn("work is annoying"), timeout=0.5)
        assert result.text == "foreground reply"
        assert tasks.revision == 1
        await asyncio.wait_for(clients[CognitiveRole.MEMORY].started.wait(), timeout=0.5)
        assert tasks.running_background >= 1
        gate.set()
        results = await bg.collect_all()
    assert all(item.status is TaskStatus.SUCCEEDED for item in results)


@pytest.mark.asyncio
async def test_default_cadence_schedules_memory_emotion_every_turn_summary_third_reflection_fifth():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime, config=MultiTaskRuntimeConfig(worker_count=5))
    bg = BackgroundCognitionRuntime(tasks, model_runtime(clients))
    async with tasks:
        for i in range(1, 6):
            await bg.run_turn(f"turn {i}")
        await bg.collect_all()
    assert clients[CognitiveRole.MEMORY].calls == 5
    assert clients[CognitiveRole.EMOTION].calls == 5
    assert clients[CognitiveRole.SUMMARY].calls == 1  # revision 3
    assert clients[CognitiveRole.REFLECTION].calls == 1  # revision 5
    assert clients[CognitiveRole.VISION].calls == 0


@pytest.mark.asyncio
async def test_vision_worker_only_runs_for_vision_events_and_receives_no_raw_payload():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.VISION: clients[CognitiveRole.VISION]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.VISION_INTERPRETATION),)
        ),
    )
    async with tasks:
        await bg.run_turn("normal text")
        event = CharacterEvent(
            type="vision_observation",
            source="vision:camera",
            content="a red cup on the desk",
            payload={"raw": "secret-image-bytes"},
        )
        result = await bg.run_foreground(event)
        handles = await bg.schedule_after_foreground(result)
        # schedule_after_foreground on the same result is deduplicated.
        assert handles == ()
        collected = await bg.collect_all()
    assert clients[CognitiveRole.VISION].calls == 1
    proposal = [
        p for item in collected if item.output for p in item.output.proposals
        if p.target == "context.vision_interpretation_candidate"
    ][0]
    assert "secret-image-bytes" not in repr(dict(proposal.provenance))
    assert proposal.provenance["foreground_event_source"] == "vision:camera"


@pytest.mark.asyncio
async def test_schedule_after_same_foreground_result_is_idempotent():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),)
        ),
    )
    async with tasks:
        result = await tasks.run_turn("I prefer tea")
        first = await bg.schedule_after_foreground(result)
        second = await bg.schedule_after_foreground(result)
        await bg.collect_all()
    assert len(first) == 1
    assert second == ()
    assert clients[CognitiveRole.MEMORY].calls == 1
    assert any(event.action == "deduplicated" for event in bg.events())


@pytest.mark.asyncio
async def test_emotion_summary_reflection_and_vision_use_typed_targets():
    clients = all_clients()
    specs = tuple(BackgroundWorkerSpec(kind) for kind in BackgroundCognitionKind)
    runtime = character()
    tasks = MultiTaskRuntime(runtime, config=MultiTaskRuntimeConfig(worker_count=5))
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime(clients),
        config=BackgroundCognitionConfig(worker_specs=specs),
    )
    event = CharacterEvent(type="vision_observation", source="vision:camera", content="red cup")
    async with tasks:
        await bg.run_foreground(event)
        results = await bg.collect_all()
    targets = {p.target for result in results if result.output for p in result.output.proposals}
    assert targets == {
        "memory.append_candidate",
        "state.emotion_candidate",
        "memory.conversation_summary_candidate",
        "cognition.reflection_candidate",
        "context.vision_interpretation_candidate",
    }


@pytest.mark.asyncio
async def test_worker_timeout_uses_existing_multitask_terminal_status():
    gate = asyncio.Event()
    clients = all_clients(gate=gate)
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.EMOTION: clients[CognitiveRole.EMOTION]}),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.EMOTION_ANALYSIS,
                    timeout_s=0.02,
                ),
            )
        ),
    )
    async with tasks:
        await bg.run_turn("annoyed")
        result = (await bg.collect_all())[0]
    assert result.status is TaskStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_worker_can_be_cancelled_without_touching_authoritative_state():
    gate = asyncio.Event()
    clients = all_clients(gate=gate)
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),)
        ),
    )
    async with tasks:
        await bg.run_turn("I prefer tea")
        handle = bg.handles()[0]
        await clients[CognitiveRole.MEMORY].started.wait()
        assert handle.cancel() is True
        result = await handle.wait()
    assert result.status is TaskStatus.CANCELLED
    assert runtime.state.trust == 50
    assert len(runtime.history) == 2


@pytest.mark.asyncio
async def test_proposal_becomes_stale_after_later_foreground_turn():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),)
        ),
    )
    async with tasks:
        await bg.run_turn("I prefer tea")
        result = (await bg.collect_all())[0]
        proposal = result.output.proposals[0]
        assert bg.proposal_is_stale(proposal) is False
        await bg.run_turn("new foreground turn")
        assert bg.proposal_is_stale(proposal) is True


@pytest.mark.asyncio
async def test_invalid_json_fails_worker_without_breaking_foreground():
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    models = model_runtime({CognitiveRole.SUMMARY: BadJSONClient()})
    bg = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.CONVERSATION_SUMMARY),)
        ),
    )
    async with tasks:
        result = await bg.run_turn("hello")
        background = (await bg.collect_all())[0]
    assert result.text == "foreground reply"
    assert background.status is TaskStatus.FAILED
    assert "JSON object" in background.error


@pytest.mark.asyncio
async def test_per_worker_max_concurrency_is_enforced_even_with_many_runtime_workers():
    gate = asyncio.Event()
    client = RoleJSONClient(
        {"emotion": "neutral", "intensity": 0.1, "confidence": 0.9, "evidence": []},
        name="emotion",
        gate=gate,
    )
    runtime = character()
    tasks = MultiTaskRuntime(runtime, config=MultiTaskRuntimeConfig(worker_count=4))
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.EMOTION: client}),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.EMOTION_ANALYSIS,
                    max_concurrency=1,
                ),
            )
        ),
    )
    async with tasks:
        await bg.run_turn("one")
        await bg.run_turn("two")
        await asyncio.wait_for(client.started.wait(), 0.5)
        await asyncio.sleep(0.02)
        assert client.max_active == 1
        gate.set()
        await bg.collect_all()
    assert client.calls == 2


@pytest.mark.asyncio
async def test_background_lifecycle_observability_records_schedule_and_terminal_state():
    clients = all_clients()
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.EMOTION: clients[CognitiveRole.EMOTION]}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),)
        ),
    )
    async with tasks:
        await bg.run_turn("annoyed")
        await bg.collect_all()
    actions = [event.action for event in bg.events()]
    assert "scheduled" in actions
    assert "succeeded" in actions



@pytest.mark.asyncio
async def test_background_queue_admission_failure_never_fails_committed_foreground_turn():
    gate = asyncio.Event()
    clients = all_clients(gate=gate)
    runtime = character()
    tasks = MultiTaskRuntime(
        runtime,
        config=MultiTaskRuntimeConfig(worker_count=1, max_pending_tasks=1),
    )
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({
            CognitiveRole.MEMORY: clients[CognitiveRole.MEMORY],
            CognitiveRole.EMOTION: clients[CognitiveRole.EMOTION],
        }),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_EXTRACTION),
                BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),
            )
        ),
    )
    async with tasks:
        foreground = await bg.run_turn("foreground must survive")
        assert foreground.text == "foreground reply"
        assert tasks.revision == 1
        assert any(event.action == "schedule_failed" for event in bg.events())
        gate.set()
        await bg.collect_all()
    assert len(runtime.history) == 2

def test_v032_public_api_exports_background_cognition_contracts():
    import ai_character_engine as ace

    assert ace.__version__ == "1.0.0"
    assert ace.BackgroundCognitionKind.CONVERSATION_SUMMARY.value == "conversation_summary"
    assert ace.BackgroundCognitionRuntime is BackgroundCognitionRuntime
    assert ace.BackgroundWorkerSpec is BackgroundWorkerSpec


class CapturingClient(RoleJSONClient):
    def __init__(self, payload: dict, *, name: str):
        super().__init__(payload, name=name)
        self.messages: list[Message] = []

    async def generate(self, messages, *, tools=None):
        self.messages = list(messages)
        return await super().generate(messages, tools=tools)


async def prompts_after_one_turn(role, kind, payload, *, goal_manager=None):
    client = CapturingClient(payload, name=role.value)
    runtime = character()
    runtime.goal_manager = goal_manager
    runtime.history = [Message("user", "earlier question"), Message("assistant", "earlier answer")]
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({role: client}),
        config=BackgroundCognitionConfig(worker_specs=(BackgroundWorkerSpec(kind),)),
    )
    async with tasks:
        await bg.run_turn("I am exhausted today")
        await bg.collect_all()
    system, user = client.messages
    return system.content, user.content


EMOTION_PAYLOAD = {"emotion": "tired", "intensity": 0.6, "confidence": 0.8, "evidence": ["exhausted"]}


@pytest.mark.asyncio
async def test_emotion_worker_never_sees_the_reply_that_followed_the_user():
    system, user = await prompts_after_one_turn(
        CognitiveRole.EMOTION, BackgroundCognitionKind.EMOTION_ANALYSIS, EMOTION_PAYLOAD
    )
    assert "I am exhausted today" in user
    assert "foreground reply" not in user
    assert "Foreground assistant response" not in user


@pytest.mark.asyncio
async def test_emotion_worker_reads_only_what_the_user_said():
    """With a character that insults the user, a local 9B model rated plain
    questions of the user as hostile towards her (stance -1 in two readings
    out of three): it read her tone as his. Without her lines, 0 in all three."""
    _, user = await prompts_after_one_turn(
        CognitiveRole.EMOTION, BackgroundCognitionKind.EMOTION_ANALYSIS, EMOTION_PAYLOAD
    )
    assert "user: earlier question" in user
    assert "earlier answer" not in user
    assert "assistant:" not in user


@pytest.mark.asyncio
async def test_other_workers_still_see_the_foreground_reply():
    _, user = await prompts_after_one_turn(
        CognitiveRole.SUMMARY,
        BackgroundCognitionKind.CONVERSATION_SUMMARY,
        {"summary": "s", "confidence": 0.9, "evidence": []},
    )
    assert "foreground reply" in user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "kind", "payload"),
    [
        (CognitiveRole.MEMORY, BackgroundCognitionKind.MEMORY_EXTRACTION, {"items": [], "confidence": 0.5, "evidence": []}),
        (CognitiveRole.EMOTION, BackgroundCognitionKind.EMOTION_ANALYSIS, EMOTION_PAYLOAD),
        (CognitiveRole.SUMMARY, BackgroundCognitionKind.CONVERSATION_SUMMARY, {"summary": "s", "confidence": 0.9, "evidence": []}),
        (CognitiveRole.REFLECTION, BackgroundCognitionKind.REFLECTION, {"insight": "i", "confidence": 0.9, "evidence": []}),
    ],
)
async def test_workers_are_told_last_to_answer_in_the_users_language(role, kind, payload):
    # Measured on a local 9B model: the rule is ignored in the system prompt and
    # followed when it is the last thing the model reads.
    _, user = await prompts_after_one_turn(role, kind, payload)
    assert user.rstrip().endswith(
        "Text values must be written in the language the user writes in, "
        "not in English unless the user writes English."
    )


@pytest.mark.asyncio
async def test_goal_worker_is_told_which_source_each_motivation_kind_may_cite():
    from ai_character_engine.goals import GoalManager
    from ai_character_engine.goals.models import MOTIVATION_SOURCE_TYPES, MotivationKind

    system, _ = await prompts_after_one_turn(
        CognitiveRole.GOAL,
        BackgroundCognitionKind.GOAL_MOTIVATION,
        {"goals": [], "confidence": 0.5, "evidence": []},
        goal_manager=GoalManager(),
    )
    assert set(MOTIVATION_SOURCE_TYPES) == set(MotivationKind)
    for kind, source_types in MOTIVATION_SOURCE_TYPES.items():
        assert f"{kind.value} -> {'|'.join(sorted(source_types))}" in system
    assert "copied exactly" in system


class GoalClient:
    """Answers with goals built from the id of the event in the prompt."""

    def __init__(self, signals_for):
        self.signals_for = signals_for

    async def generate(self, messages, *, tools=None):
        import re

        event_id = re.search(r'"event": \{.*?"id": "([^"]+)"', messages[1].content).group(1)
        goal = {
            "objective": "Show the drawing when it is done",
            "horizon": "short_term",
            "urgency": 0.7,
            "conflict_key": None,
            "motivation_signals": self.signals_for(event_id),
            "confidence": 0.9,
        }
        payload = {"goals": [goal], "confidence": 0.9, "evidence": []}
        return LLMResponse(text=json.dumps(payload), model="goal")


def signal(kind, source_type, source_id):
    return {
        "kind": kind,
        "strength": 0.8,
        "source_type": source_type,
        "source_id": source_id,
        "rationale": "because",
    }


async def proposed_goals(signals_for):
    from ai_character_engine.goals import GoalManager

    runtime = character()
    runtime.goal_manager = GoalManager()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.GOAL: GoalClient(signals_for)}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.GOAL_MOTIVATION),)
        ),
    )
    async with tasks:
        await bg.run_turn("show me the drawing when it is done")
        result = (await bg.collect_all())[-1]
    assert result.error is None
    return [proposal.payload for proposal in result.output.proposals]


@pytest.mark.asyncio
async def test_a_goal_keeps_the_sources_it_was_given_and_loses_the_invented_ones():
    # A local 9B model cites a belief id that was never supplied in about one
    # goal out of two; the commit check then refuses the whole goal.
    goals = await proposed_goals(
        lambda event_id: [
            signal("explicit_request", "event", event_id),
            signal("belief_alignment", "belief", "belief-the-model-made-up"),
        ]
    )
    assert [[s["source_type"] for s in goal["motivation_signals"]] for goal in goals] == [["event"]]


@pytest.mark.asyncio
async def test_a_goal_resting_only_on_invented_sources_is_not_proposed():
    goals = await proposed_goals(
        lambda event_id: [signal("belief_alignment", "belief", "belief-the-model-made-up")]
    )
    assert goals == []


@pytest.mark.asyncio
async def test_a_source_of_the_wrong_type_for_its_kind_is_not_kept():
    goals = await proposed_goals(
        lambda event_id: [
            signal("belief_alignment", "event", event_id),
            signal("state_pressure", "state", "emotion"),
        ]
    )
    assert [[s["kind"] for s in goal["motivation_signals"]] for goal in goals] == [
        ["state_pressure"]
    ]


def memory_item(summary, evidence):
    return {"summary": summary, "kind": "fact", "importance": 0.5, "confidence": 0.9, "evidence": evidence}


async def proposed_memories(items, *, turns=("I am exhausted today",), every_n_revisions=1):
    client = RoleJSONClient({"items": items, "confidence": 0.9, "evidence": []}, name="memory")
    runtime = character()
    runtime.history = [Message("user", "earlier question"), Message("assistant", "earlier answer")]
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: client}),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.MEMORY_EXTRACTION, every_n_revisions=every_n_revisions
                ),
            )
        ),
    )
    async with tasks:
        for turn in turns:
            await bg.run_turn(turn)
        result = (await bg.collect_all())[-1]
    return result.output.proposals


@pytest.mark.asyncio
async def test_memory_items_are_proposed_only_for_what_the_user_just_said():
    proposals = await proposed_memories(
        [
            memory_item("User is exhausted", "I am exhausted today"),
            memory_item("User asked something earlier", "earlier question"),
            memory_item("User enjoys replies", "foreground reply"),
            memory_item("User said something never said", "I love tea"),
        ]
    )
    assert [p.payload["summary"] for p in proposals] == ["User is exhausted"]
    assert proposals[0].provenance["evidence"] == ["I am exhausted today"]


@pytest.mark.asyncio
async def test_memory_evidence_matches_despite_spacing_the_model_added():
    proposals = await proposed_memories(
        [memory_item("使用者在研究 AI", "我在研究 AI 和做遊戲")], turns=("我在研究AI和做遊戲",)
    )
    assert [p.payload["summary"] for p in proposals] == ["使用者在研究 AI"]


@pytest.mark.asyncio
async def test_a_worker_that_runs_every_other_turn_covers_both_turns():
    proposals = await proposed_memories(
        [
            memory_item("User has a cat", "I have a cat"),
            memory_item("User is exhausted", "I am exhausted today"),
            memory_item("User asked something earlier", "earlier question"),
        ],
        turns=("I have a cat", "I am exhausted today"),
        every_n_revisions=2,
    )
    assert [p.payload["summary"] for p in proposals] == ["User has a cat", "User is exhausted"]


@pytest.mark.asyncio
async def test_memory_items_without_a_quote_field_are_kept_for_older_prompts():
    proposals = await proposed_memories(
        [{"summary": "User prefers tea", "kind": "preference", "importance": 0.8, "confidence": 0.9}]
    )
    assert [p.payload["summary"] for p in proposals] == ["User prefers tea"]


async def emotion_proposal_payload(payload):
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.EMOTION: RoleJSONClient(payload, name="emotion")}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),)
        ),
    )
    async with tasks:
        await bg.run_turn("thank you for staying up with me")
        result = (await bg.collect_all())[0]
    assert result.status is TaskStatus.SUCCEEDED, result.error
    return dict(result.output.proposals[0].payload)


@pytest.mark.asyncio
async def test_emotion_worker_passes_on_valence_and_stance_within_bounds():
    payload = await emotion_proposal_payload(
        {"emotion": "grateful", "intensity": 0.8, "valence": 0.9, "stance": 7, "confidence": 0.9, "evidence": []}
    )
    assert payload == {"emotion": "grateful", "intensity": 0.8, "valence": 0.9, "stance": 1.0}


@pytest.mark.asyncio
async def test_emotion_worker_drops_valence_and_stance_it_cannot_read():
    payload = await emotion_proposal_payload(
        {"emotion": "grateful", "intensity": 0.8, "valence": "very", "stance": None, "confidence": 0.9, "evidence": []}
    )
    assert payload == {"emotion": "grateful", "intensity": 0.8}


@pytest.mark.asyncio
async def test_emotion_worker_is_asked_for_valence_and_stance():
    system, _ = await prompts_after_one_turn(
        CognitiveRole.EMOTION, BackgroundCognitionKind.EMOTION_ANALYSIS, EMOTION_PAYLOAD
    )
    assert '"valence":-1..1' in system and '"stance":-1..1' in system


async def memory_prompt(*, turns, every_n_revisions):
    client = CapturingClient({"items": [], "confidence": 0.5, "evidence": []}, name="memory")
    runtime = character()
    runtime.history = [Message("user", "earlier question"), Message("assistant", "earlier answer")]
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: client}),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.MEMORY_EXTRACTION, every_n_revisions=every_n_revisions
                ),
            )
        ),
    )
    async with tasks:
        for turn in turns:
            await bg.run_turn(turn)
        await bg.collect_all()
    return client.messages[1].content


@pytest.mark.asyncio
async def test_a_memory_worker_that_runs_every_other_turn_is_shown_both_turns_to_extract_from():
    """Measured: told to read only the latest line, a worker on a two-turn cadence
    never stored the name the user gave on the turn in between."""
    prompt = await memory_prompt(turns=("I am Dawn", "thanks for listening"), every_n_revisions=2)
    section = prompt.split("User lines to extract from:\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["- I am Dawn", "- thanks for listening"]


@pytest.mark.asyncio
async def test_a_memory_worker_that_runs_every_turn_extracts_from_the_latest_line_only():
    prompt = await memory_prompt(turns=("I am Dawn",), every_n_revisions=1)
    section = prompt.split("User lines to extract from:\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["- I am Dawn"]


async def committed_memories(items, events, *, every_n_revisions=2):
    """Through the real coordinator: what a host would actually end up storing."""
    from ai_character_engine import CognitiveCommitCoordinator, MemoryManager

    class NoForegroundWrites:
        def importance(self, **_):
            return None

    client = CapturingClient({"items": items, "confidence": 0.9, "evidence": []}, name="memory")
    runtime = CharacterRuntime(
        character=CharacterProfile(id="c", name="C", description="test"),
        llm=ForegroundClient(),
        memory_manager=MemoryManager(write_policy=NoForegroundWrites()),
    )
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.MEMORY: client}),
        config=BackgroundCognitionConfig(
            worker_specs=(
                BackgroundWorkerSpec(
                    BackgroundCognitionKind.MEMORY_EXTRACTION, every_n_revisions=every_n_revisions
                ),
            )
        ),
    )
    commits = CognitiveCommitCoordinator(tasks)
    async with tasks:
        for event in events:
            if isinstance(event, str):
                event = CharacterEvent.user_message(event)
            await bg.run_foreground(event)
        await commits.commit_task_results(await bg.collect_all())
    stored = runtime.memory_manager.store.list_for_character(runtime.memory_scope_id)
    return [record.summary for record in stored], client.messages[1].content


@pytest.mark.asyncio
async def test_a_fact_is_stored_even_when_the_turn_after_it_was_a_question():
    stored, _ = await committed_memories(
        [memory_item("User is called Dawn", "我叫Dawn")], ["我叫Dawn", "你覺得呢？"]
    )
    assert stored == ["User is called Dawn"]


@pytest.mark.asyncio
async def test_something_the_user_only_quoted_is_not_stored_as_their_own_fact():
    stored, _ = await committed_memories(
        [
            memory_item("User hates cats", "我討厭貓"),
            memory_item("User lives in Taipei", "我住台北"),
        ],
        ["他說「我討厭貓」", "我住台北"],
    )
    assert stored == ["User lives in Taipei"]


@pytest.mark.asyncio
async def test_an_event_that_is_not_the_user_speaking_is_neither_a_source_nor_a_veto():
    observation = CharacterEvent(
        type="proactive_observation", source="host", content="The user has been quiet."
    )
    stored, prompt = await committed_memories(
        [
            memory_item("User is called Dawn", "I am Dawn"),
            memory_item("User is quiet", "The user has been quiet."),
        ],
        ["I am Dawn", observation],
    )
    assert stored == ["User is called Dawn"]
    section = prompt.split("User lines to extract from:\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["- I am Dawn"]


@pytest.mark.asyncio
async def test_the_window_counts_turns_not_user_lines():
    observation = CharacterEvent(
        type="proactive_observation", source="host", content="The user has been quiet."
    )
    _, prompt = await committed_memories(
        [], ["covered by the previous run", "also covered", observation, "I am Dawn"]
    )
    section = prompt.split("User lines to extract from:\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["- I am Dawn"]


@pytest.mark.asyncio
async def test_handles_of_finished_work_are_not_kept_for_ever():
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    bg = BackgroundCognitionRuntime(
        tasks,
        model_runtime({CognitiveRole.EMOTION: RoleJSONClient(EMOTION_PAYLOAD, name="emotion")}),
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.EMOTION_ANALYSIS),),
            event_history=4,
        ),
    )
    async with tasks:
        for number in range(10):
            await bg.run_turn(f"line {number}")
            await bg.collect_all()

    assert len(bg.handles()) == 4

