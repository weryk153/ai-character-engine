from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

import ai_character_engine as ace
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.cognition import (
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
)
from ai_character_engine.collaboration import (
    CollaborationResult,
    CollaborationSource,
    CognitiveWorkItem,
    CognitiveWorkPlan,
    SpecialistCollaborationConfig,
    SpecialistCollaborationRuntime,
    SpecialistKind,
    SpecialistRunStatus,
    SpecialistSpec,
    VerificationDecision,
    VerificationReport,
    default_specialists,
)
from ai_character_engine.llm import LLMResponse, ModelEndpoint
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.tasks import MultiTaskRuntime, MultiTaskRuntimeConfig, TaskStatus
from ai_character_engine.tools import ToolDefinition, ToolRegistry
from tests.fakes import FakeLLMClient


class ScriptJSONClient:
    def __init__(self, *payloads, name: str, delay: float = 0.0, gate: asyncio.Event | None = None):
        self.payloads = list(payloads)
        self.name = name
        self.delay = delay
        self.gate = gate
        self.calls = []
        self.started = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.payloads:
            raise AssertionError(f"{self.name} ran out of payloads")
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(text=text, model=self.name)


def planner_payload(*items):
    return {"rationale": "split by evidence type", "work_items": list(items)}


def work(item_id: str, specialist: str, *, source_ids=(), instruction="analyze"):
    return {
        "id": item_id,
        "specialist": specialist,
        "instruction": instruction,
        "source_ids": list(source_ids),
    }


def finding_payload(summary="supported", *, evidence=(), recommendations=(), confidence=0.9):
    return {
        "summary": summary,
        "confidence": confidence,
        "evidence_source_ids": list(evidence),
        "recommendations": list(recommendations),
    }


def verifier_payload(*accepted, decision="accept", issues=(), summary="verified", confidence=0.9):
    return {
        "decision": decision,
        "accepted_work_item_ids": list(accepted),
        "issues": list(issues),
        "summary": summary,
        "confidence": confidence,
    }


def model_runtime(clients: dict[CognitiveRole, ScriptJSONClient]) -> CognitiveModelRuntime:
    endpoints = []
    policies = {}
    for role, client in clients.items():
        endpoint_id = f"ep-{role.value}"
        endpoints.append(ModelEndpoint(endpoint_id, client))
        policies[role] = CognitiveRolePolicy(primary_endpoint_ids=(endpoint_id,))
    return CognitiveModelRuntime(
        endpoints=tuple(endpoints),
        router=CognitiveModelRouter(policies=policies),
    )


def character(*, tool_registry=None) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id="c1", name="Aki", description="test character"),
        llm=FakeLLMClient("foreground"),
        tool_registry=tool_registry,
    )


async def run_collaboration(
    clients,
    *,
    sources=(),
    tool_catalog=(),
    config=None,
    specialists=None,
    runtime=None,
):
    tasks = MultiTaskRuntime(runtime or character(), config=MultiTaskRuntimeConfig(worker_count=1))
    collaboration = SpecialistCollaborationRuntime(
        tasks,
        model_runtime(clients),
        config=config,
        specialists=specialists,
    )
    async with tasks:
        handle = await collaboration.submit(
            "Understand what matters for the next character response",
            sources=sources,
            tool_catalog=tool_catalog,
        )
        result = await handle.wait()
    return result, collaboration, tasks


def test_roles_exist_without_provider_or_model_ids():
    assert CognitiveRole.PLANNER.value == "planner"
    assert CognitiveRole.TOOL.value == "tool"
    assert CognitiveRole.VERIFIER.value == "verifier"


def test_default_specialists_are_bounded_memory_vision_tool_roles():
    specs = default_specialists()
    assert [item.specialist_id for item in specs] == ["memory", "vision", "tool"]
    assert [item.kind for item in specs] == [SpecialistKind.MEMORY, SpecialistKind.VISION, SpecialistKind.TOOL]
    assert [item.role for item in specs] == [CognitiveRole.MEMORY, CognitiveRole.VISION, CognitiveRole.TOOL]


def test_collaboration_config_rejects_unbounded_or_invalid_limits():
    with pytest.raises(ValueError):
        SpecialistCollaborationConfig(max_work_items=0)
    with pytest.raises(ValueError):
        SpecialistCollaborationConfig(max_work_items=13)
    with pytest.raises(ValueError):
        SpecialistCollaborationConfig(max_parallel_specialists=0)
    with pytest.raises(ValueError):
        SpecialistCollaborationConfig(specialist_timeout_s=0)


@pytest.mark.asyncio
async def test_happy_path_plans_parallel_specialists_and_verifies_advisory_findings():
    sources = [
        CollaborationSource("event:1", "event", "I prefer concise answers"),
        CollaborationSource("vision:1", "vision", "A red mug is visible"),
    ]
    clients = {
        CognitiveRole.PLANNER: ScriptJSONClient(
            planner_payload(
                work("m1", "memory", source_ids=("event:1",)),
                work("v1", "vision", source_ids=("vision:1",)),
            ),
            name="planner",
        ),
        CognitiveRole.MEMORY: ScriptJSONClient(
            finding_payload("Preference evidence exists", evidence=("event:1",)), name="memory"
        ),
        CognitiveRole.VISION: ScriptJSONClient(
            finding_payload("A mug observation exists", evidence=("vision:1",)), name="vision"
        ),
        CognitiveRole.VERIFIER: ScriptJSONClient(
            verifier_payload("m1", "v1"), name="verifier"
        ),
    }
    result, _, _ = await run_collaboration(clients, sources=sources)
    assert result.status is TaskStatus.SUCCEEDED
    assert isinstance(result.output.value, CollaborationResult)
    value = result.output.value
    assert value.verification.decision is VerificationDecision.ACCEPT
    assert {item.work_item_id for item in value.accepted_findings} == {"m1", "v1"}
    assert result.output.proposals == ()
    assert result.output.metadata["collaboration"]["authoritative"] is False


@pytest.mark.asyncio
async def test_planner_unknown_specialist_fails_task_before_any_specialist_call():
    planner = ScriptJSONClient(planner_payload(work("x", "invented")), name="planner")
    memory = ScriptJSONClient(finding_payload(), name="memory")
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: planner,
            CognitiveRole.MEMORY: memory,
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        }
    )
    assert result.status is TaskStatus.FAILED
    assert "unknown specialist" in result.error
    assert memory.calls == []


@pytest.mark.asyncio
async def test_planner_cannot_create_dependency_graph_or_spawn_agents():
    raw = work("m1", "memory")
    raw["depends_on"] = ["x"]
    planner = ScriptJSONClient(planner_payload(raw), name="planner")
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: planner,
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        }
    )
    assert result.status is TaskStatus.FAILED
    assert "forbidden fields" in result.error


@pytest.mark.asyncio
async def test_planner_cannot_choose_model_provider_or_tool_call():
    raw = work("m1", "memory")
    raw["model_id"] = "secret-model"
    planner = ScriptJSONClient(planner_payload(raw), name="planner")
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: planner,
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        }
    )
    assert result.status is TaskStatus.FAILED
    assert "model_id" in result.error


@pytest.mark.asyncio
async def test_planner_max_work_items_is_hard_limit_not_soft_prompt_advice():
    items = [work(f"m{i}", "memory") for i in range(5)]
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(*items), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        },
        config=SpecialistCollaborationConfig(max_work_items=4),
    )
    assert result.status is TaskStatus.FAILED
    assert "max_work_items" in result.error


@pytest.mark.asyncio
async def test_duplicate_plan_item_ids_are_rejected():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("same", "memory"), work("same", "vision")), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
            CognitiveRole.VISION: ScriptJSONClient(finding_payload(), name="vision"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        }
    )
    assert result.status is TaskStatus.FAILED
    assert "work item ids must be unique" in result.error


@pytest.mark.asyncio
async def test_planner_may_only_select_known_source_ids():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory", source_ids=("missing",))), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        }
    )
    assert result.status is TaskStatus.FAILED
    assert "unknown source ids" in result.error


@pytest.mark.asyncio
async def test_planner_receives_bounded_source_catalog_for_routing_decisions():
    planner = ScriptJSONClient(
        planner_payload(work("m1", "memory", source_ids=("event:1",))), name="planner"
    )
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: planner,
            CognitiveRole.MEMORY: ScriptJSONClient(
                finding_payload("ok", evidence=("event:1",)), name="memory"
            ),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
        },
        sources=[CollaborationSource("event:1", "event", "SOURCE_CONTENT_FOR_PLANNER")],
    )
    assert result.status is TaskStatus.SUCCEEDED
    payload = json.loads(planner.calls[0][-1].content)
    assert payload["sources"][0]["content"] == "SOURCE_CONTENT_FOR_PLANNER"


@pytest.mark.asyncio
async def test_verifier_receives_only_sources_selected_by_plan_not_unrelated_content():
    verifier = ScriptJSONClient(verifier_payload("m1"), name="verifier")
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory", source_ids=("event:selected",))), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient(
                finding_payload("ok", evidence=("event:selected",)), name="memory"
            ),
            CognitiveRole.VERIFIER: verifier,
        },
        sources=[
            CollaborationSource("event:selected", "event", "VERIFY_THIS_SOURCE"),
            CollaborationSource("event:hidden", "event", "UNRELATED_SECRET_CONTENT"),
        ],
    )
    assert result.status is TaskStatus.SUCCEEDED
    verifier_payload_seen = verifier.calls[0][-1].content
    assert "VERIFY_THIS_SOURCE" in verifier_payload_seen
    assert "UNRELATED_SECRET_CONTENT" not in verifier_payload_seen


@pytest.mark.asyncio
async def test_specialist_receives_only_sources_selected_by_planner():
    memory = ScriptJSONClient(
        finding_payload("selected source only", evidence=("event:selected",)), name="memory"
    )
    sources = [
        CollaborationSource("event:selected", "event", "VISIBLE_TO_SPECIALIST"),
        CollaborationSource("event:hidden", "event", "MUST_NOT_LEAK"),
    ]
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory", source_ids=("event:selected",))), name="planner"
            ),
            CognitiveRole.MEMORY: memory,
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
        },
        sources=sources,
    )
    assert result.status is TaskStatus.SUCCEEDED
    user_payload = memory.calls[0][-1].content
    assert "VISIBLE_TO_SPECIALIST" in user_payload
    assert "MUST_NOT_LEAK" not in user_payload


@pytest.mark.asyncio
async def test_specialist_cannot_cite_source_outside_its_assigned_slice():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory", source_ids=("event:1",))), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient(
                finding_payload(evidence=("event:2",)), name="memory"
            ),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        },
        sources=[
            CollaborationSource("event:1", "event", "one"),
            CollaborationSource("event:2", "event", "two"),
        ],
    )
    assert result.status is TaskStatus.SUCCEEDED
    value = result.output.value
    assert value.findings[0].status is SpecialistRunStatus.FAILED
    assert "outside its work item" in value.findings[0].error
    assert value.verification.decision is VerificationDecision.REJECT


@pytest.mark.asyncio
async def test_specialists_run_in_parallel_under_bounded_fanout():
    gate = asyncio.Event()
    memory = ScriptJSONClient(finding_payload("m"), name="memory", gate=gate)
    vision = ScriptJSONClient(finding_payload("v"), name="vision", gate=gate)
    clients = {
        CognitiveRole.PLANNER: ScriptJSONClient(
            planner_payload(work("m1", "memory"), work("v1", "vision")), name="planner"
        ),
        CognitiveRole.MEMORY: memory,
        CognitiveRole.VISION: vision,
        CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1", "v1"), name="verifier"),
    }
    tasks = MultiTaskRuntime(character(), config=MultiTaskRuntimeConfig(worker_count=1))
    collaboration = SpecialistCollaborationRuntime(tasks, model_runtime(clients))
    async with tasks:
        handle = await collaboration.submit("parallel check")
        await asyncio.wait_for(memory.started.wait(), timeout=0.5)
        await asyncio.wait_for(vision.started.wait(), timeout=0.5)
        assert tasks.running_background == 1  # one collaboration task, internal fan-out is bounded inside it
        gate.set()
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_specialist_failure_isolated_and_verifier_can_accept_remaining_result():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory"), work("v1", "vision")), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient("not json", name="memory"),
            CognitiveRole.VISION: ScriptJSONClient(finding_payload("vision ok"), name="vision"),
            CognitiveRole.VERIFIER: ScriptJSONClient(
                verifier_payload("v1", decision="partial"), name="verifier"
            ),
        }
    )
    assert result.status is TaskStatus.SUCCEEDED
    findings = {item.work_item_id: item for item in result.output.value.findings}
    assert findings["m1"].status is SpecialistRunStatus.FAILED
    assert findings["v1"].status is SpecialistRunStatus.SUCCEEDED
    assert result.output.value.verification.accepted_work_item_ids == ("v1",)


@pytest.mark.asyncio
async def test_specialist_timeout_isolated_from_collaboration_task():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory", delay=0.1),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        },
        config=SpecialistCollaborationConfig(specialist_timeout_s=0.01),
    )
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.value.findings[0].status is SpecialistRunStatus.TIMED_OUT
    assert result.output.value.verification.decision is VerificationDecision.REJECT


@pytest.mark.asyncio
async def test_verifier_partial_acceptance_is_preserved_without_majority_vote():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("m1", "memory"), work("v1", "vision")), name="planner"
            ),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("memory"), name="memory"),
            CognitiveRole.VISION: ScriptJSONClient(finding_payload("vision"), name="vision"),
            CognitiveRole.VERIFIER: ScriptJSONClient(
                verifier_payload(
                    "m1",
                    decision="partial",
                    issues=({"code": "unsupported", "message": "vision is weak", "work_item_id": "v1"},),
                ),
                name="verifier",
            ),
        }
    )
    report = result.output.value.verification
    assert report.decision is VerificationDecision.PARTIAL
    assert report.accepted_work_item_ids == ("m1",)
    assert report.issues[0].work_item_id == "v1"


@pytest.mark.asyncio
async def test_verifier_cannot_accept_failed_or_unknown_work_item():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient("bad-json", name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
        }
    )
    # With no successful specialists, verifier is bypassed and result is deterministically rejected.
    assert result.output.value.verification.decision is VerificationDecision.REJECT
    assert result.output.value.verification.accepted_work_item_ids == ()


@pytest.mark.asyncio
async def test_invalid_verifier_response_becomes_unverified_instead_of_authority():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("ok"), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(
                verifier_payload("not-a-plan-id", decision="partial"), name="verifier"
            ),
        }
    )
    report = result.output.value.verification
    assert report.decision is VerificationDecision.UNVERIFIED
    assert report.accepted_work_item_ids == ()
    assert report.issues[0].code == "verifier_failed"


@pytest.mark.asyncio
async def test_verifier_timeout_becomes_unverified_not_task_failure():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("ok"), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier", delay=0.1),
        },
        config=SpecialistCollaborationConfig(verifier_timeout_s=0.01),
    )
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.value.verification.decision is VerificationDecision.UNVERIFIED


@pytest.mark.asyncio
async def test_verifier_can_be_disabled_but_findings_never_become_implicitly_accepted():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("ok"), name="memory"),
        },
        config=SpecialistCollaborationConfig(require_verifier=False),
    )
    report = result.output.value.verification
    assert report.decision is VerificationDecision.UNVERIFIED
    assert report.accepted_work_item_ids == ()
    assert result.output.value.accepted_findings == ()


@pytest.mark.asyncio
async def test_tool_specialist_sees_catalog_but_cannot_execute_tool_handler():
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return "12:00"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="clock",
            description="read current clock",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        clock,
    )
    tool_client = ScriptJSONClient(
        finding_payload("Use the clock if current time is needed", recommendations=("tool:clock",)),
        name="tool",
    )
    clients = {
        CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("t1", "tool")), name="planner"),
        CognitiveRole.TOOL: tool_client,
        CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("t1"), name="verifier"),
    }
    runtime = character(tool_registry=registry)
    tasks = MultiTaskRuntime(runtime)
    collaboration = SpecialistCollaborationRuntime(tasks, model_runtime(clients))
    async with tasks:
        foreground = await tasks.run_turn("What time-related capabilities do you have?")
        handle = await collaboration.submit_after_foreground(
            foreground,
            objective="Check whether a registered tool could help the next response",
        )
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    assert calls == 0
    tool_prompt = tool_client.calls[0][-1].content
    assert '"name": "clock"' in tool_prompt
    assert "handler" not in tool_prompt


@pytest.mark.asyncio
async def test_tool_specialist_cannot_recommend_unregistered_tool_name():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("t1", "tool")), name="planner"),
            CognitiveRole.TOOL: ScriptJSONClient(
                finding_payload("invent", recommendations=("tool:delete_everything",)), name="tool"
            ),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
        },
        tool_catalog=[
            {
                "name": "clock",
                "description": "read clock",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    finding = result.output.value.findings[0]
    assert finding.status is SpecialistRunStatus.FAILED
    assert "unknown tools" in finding.error


@pytest.mark.asyncio
async def test_submit_after_foreground_copies_event_and_registered_tool_metadata_only():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "lookup",
            "look something up",
            {"type": "object", "properties": {"q": {"type": "string"}}},
        ),
        lambda q: q,
    )
    planner = ScriptJSONClient(
        planner_payload(work("m1", "memory", source_ids=())), name="planner"
    )
    clients = {
        CognitiveRole.PLANNER: planner,
        CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("ok"), name="memory"),
        CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
    }
    runtime = character(tool_registry=registry)
    tasks = MultiTaskRuntime(runtime)
    collaboration = SpecialistCollaborationRuntime(tasks, model_runtime(clients))
    async with tasks:
        foreground = await tasks.run_turn("hello")
        handle = await collaboration.submit_after_foreground(foreground, objective="review context")
        result = await handle.wait()
    assert result.status is TaskStatus.SUCCEEDED
    planner_payload_seen = json.loads(planner.calls[0][-1].content)
    assert any(source.startswith("event:") for source in planner_payload_seen["available_source_ids"])
    assert planner_payload_seen["available_tools"] == ["lookup"]


@pytest.mark.asyncio
async def test_collaboration_result_is_bound_to_task_snapshot_revision_and_can_become_stale():
    clients = {
        CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
        CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("ok"), name="memory"),
        CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
    }
    runtime = character()
    tasks = MultiTaskRuntime(runtime)
    collaboration = SpecialistCollaborationRuntime(tasks, model_runtime(clients))
    async with tasks:
        await tasks.run_turn("first")
        handle = await collaboration.submit("analyze")
        result = await handle.wait()
        value = result.output.value
        assert value.base_revision == 1
        assert value.is_stale(tasks.revision) is False
        await tasks.run_turn("second")
        assert value.is_stale(tasks.revision) is True


def test_collaboration_result_staleness_rejects_negative_revision():
    plan = CognitiveWorkPlan(
        "objective",
        "rationale",
        (CognitiveWorkItem("m1", "memory", "analyze"),),
        0,
    )
    result = CollaborationResult(
        objective="objective",
        plan=plan,
        findings=(),
        verification=VerificationReport(
            decision=VerificationDecision.REJECT,
            accepted_work_item_ids=(),
            issues=(),
            summary="none",
        ),
        base_revision=0,
    )
    with pytest.raises(ValueError):
        result.is_stale(-1)


def test_duplicate_specialist_ids_are_rejected_on_runtime_construction():
    spec = SpecialistSpec(
        specialist_id="same",
        kind=SpecialistKind.MEMORY,
        role=CognitiveRole.MEMORY,
        purpose="one",
        system_prompt="one",
    )
    tasks = MultiTaskRuntime(character())
    models = model_runtime({
        CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("x", "same")), name="planner"),
        CognitiveRole.MEMORY: ScriptJSONClient(finding_payload(), name="memory"),
        CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload(), name="verifier"),
    })
    with pytest.raises(ValueError):
        SpecialistCollaborationRuntime(tasks, models, specialists=(spec, replace(spec, purpose="two")))


@pytest.mark.asyncio
async def test_custom_specialist_still_uses_registered_semantic_role_not_model_id():
    custom = SpecialistSpec(
        specialist_id="relationship",
        kind=SpecialistKind.MEMORY,
        role=CognitiveRole.REFLECTION,
        purpose="inspect relationship evidence",
        system_prompt="Analyze relationship evidence conservatively.",
    )
    reflection = ScriptJSONClient(finding_payload("relationship finding"), name="reflection-model")
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(
                planner_payload(work("r1", "relationship")), name="planner"
            ),
            CognitiveRole.REFLECTION: reflection,
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("r1"), name="verifier"),
        },
        specialists=(custom,),
    )
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.value.findings[0].specialist_id == "relationship"
    assert result.output.value.findings[0].model == "reflection-model"


@pytest.mark.asyncio
async def test_collaboration_task_output_never_contains_authoritative_task_proposals():
    result, _, _ = await run_collaboration(
        {
            CognitiveRole.PLANNER: ScriptJSONClient(planner_payload(work("m1", "memory")), name="planner"),
            CognitiveRole.MEMORY: ScriptJSONClient(finding_payload("remember this"), name="memory"),
            CognitiveRole.VERIFIER: ScriptJSONClient(verifier_payload("m1"), name="verifier"),
        }
    )
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.proposals == ()
    assert result.output.metadata["collaboration"]["authoritative"] is False


def test_core_version():
    assert ace.__version__ == "1.0.0"
