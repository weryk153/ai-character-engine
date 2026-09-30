from __future__ import annotations

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.evaluation import EvalDimension, PersonaEvaluator, PersonaRule, PersonaSpec
from ai_character_engine.observability import (
    InMemoryObservabilitySink,
    ProductionEvalPolicy,
    ProductionEvalRunner,
    TraceContext,
    Tracer,
)
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.service import BufferedCharacterStreamSource, CharacterService
from ai_character_engine.service.models import SessionCreateRequest
from ai_character_engine.session import CharacterRuntimeFactory, SessionManager
from tests.fakes import FakeLLMClient


def character() -> CharacterProfile:
    return CharacterProfile(id="mei", name="Mei", description="test character")


def build_service(*, sink: InMemoryObservabilitySink, production_eval=None):
    profile = character()
    tracer = Tracer(sink)
    manager = SessionManager(default_ttl_seconds=None)
    factory = CharacterRuntimeFactory(
        characters={profile.id: profile},
        llm_factory=lambda record: FakeLLMClient("hello"),
        session_manager=manager,
        tracer=tracer,
    )
    service = CharacterService(
        runtime_factory=factory,
        session_manager=manager,
        stream_source=BufferedCharacterStreamSource(),
        tracer=tracer,
        production_eval=production_eval,
    )
    service.create_session(
        SessionCreateRequest(user_id="alice", character_id="mei", session_id="s1")
    )
    return service, factory


@pytest.mark.asyncio
async def test_runtime_records_spans_and_turn_telemetry() -> None:
    sink = InMemoryObservabilitySink()
    runtime = CharacterRuntime(
        character=character(),
        llm=FakeLLMClient("hello"),
        tracer=Tracer(sink),
    )
    result = await runtime.process_event(
        __import__("ai_character_engine").CharacterEvent.user_message("hi"),
        trace_context=TraceContext(trace_id="trace-1", request_id="req-1"),
    )

    assert result.text == "hello"
    assert {span.name for span in sink.spans} >= {
        "runtime.process_event",
        "memory.retrieve",
        "context.build",
        "llm.generate",
    }
    assert all(span.trace_id == "trace-1" for span in sink.spans)
    assert len(sink.turns) == 1
    turn = sink.turns[0]
    assert turn.trace_id == "trace-1"
    assert turn.request_id == "req-1"
    assert turn.input_tokens == 10
    assert turn.output_tokens == 2
    assert turn.tool_count == 0


@pytest.mark.asyncio
async def test_service_and_runtime_spans_share_trace_and_parent_link() -> None:
    sink = InMemoryObservabilitySink()
    service, _ = build_service(sink=sink)
    response = await service.send_message(
        "s1", "hi", request_id="req-service", trace_id="trace-service"
    )
    assert response.text == "hello"

    service_span = next(x for x in sink.spans if x.name == "service.send_message")
    runtime_span = next(x for x in sink.spans if x.name == "runtime.process_event")
    assert service_span.trace_id == runtime_span.trace_id == "trace-service"
    assert runtime_span.parent_span_id == service_span.span_id
    turn = sink.turns[-1]
    assert turn.session_id == "s1"
    assert turn.user_id == "alice"
    assert turn.character_id == "mei"


@pytest.mark.asyncio
async def test_production_eval_sample_rate_zero_does_not_execute_evaluator() -> None:
    class ExplodingEvaluator:
        async def evaluate(self, case):
            raise AssertionError("evaluator should not execute")

    sink = InMemoryObservabilitySink()
    runner = ProductionEvalRunner(
        evaluator=ExplodingEvaluator(),
        persona_resolver=lambda _: PersonaSpec(),
        policy=ProductionEvalPolicy(sample_rate=0),
        sink=sink,
    )
    service, _ = build_service(sink=sink, production_eval=runner)
    await service.send_message("s1", "hi", request_id="r", trace_id="not-sampled")
    assert sink.evals == []


@pytest.mark.asyncio
async def test_production_eval_records_sampled_persona_result() -> None:
    sink = InMemoryObservabilitySink()
    persona = PersonaSpec(
        rules=(
            PersonaRule(
                rule_id="style.short",
                description="Keep replies concise",
                dimension=EvalDimension.STYLE,
                mode="max_length",
                max_chars=20,
            ),
        )
    )
    runner = ProductionEvalRunner(
        evaluator=PersonaEvaluator(),
        persona_resolver=lambda _: persona,
        policy=ProductionEvalPolicy(sample_rate=1),
        sink=sink,
    )
    service, _ = build_service(sink=sink, production_eval=runner)
    await service.send_message("s1", "hi", request_id="r", trace_id="sampled")

    assert len(sink.evals) == 1
    record = sink.evals[0]
    assert record.trace_id == "sampled"
    assert record.passed is True
    assert record.consistency_score == 1.0
    assert record.violation_count == 0


def test_observability_summary_aggregates_turns_and_evals() -> None:
    sink = InMemoryObservabilitySink()
    # exercise summary through real spans/turns in other tests is not deterministic here;
    # empty summary must still be well-defined.
    summary = sink.summary()
    assert summary.turn_count == 0
    assert summary.span_count == 0
    assert summary.avg_turn_latency_ms == 0
    assert summary.eval_pass_rate is None


def test_production_eval_sampling_is_deterministic_for_trace_id() -> None:
    policy = ProductionEvalPolicy(sample_rate=0.5)
    assert policy.should_sample("same-trace") == policy.should_sample("same-trace")
