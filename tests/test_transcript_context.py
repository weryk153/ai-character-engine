"""The prompt of a turn extends the prompt of the turn before it.

Inference servers reuse work for a prompt that starts like an earlier one. How
much depends on the server: llama.cpp continues from the exact point where two
prompts diverge; LM Studio's MLX engine keeps one checkpoint per request, at
the last multiple of 256 tokens of the prompt. With a 300-token context that
was inserted before the newest message and taken out again for the next turn,
that checkpoint always lay in text the next prompt did not contain: the server
log showed ``cached_tokens=2048`` on every turn while ``uncached_tokens`` grew
from 450 to 1033, and the time to the first word grew from 2.8 s to 5.6 s over
eight turns.

So what the character knows is written into the conversation as notes that
stay where they are, and a note says only what no earlier note still in the
conversation has said.
"""
from __future__ import annotations

import pytest

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.context.builder import ContextBuilder, is_turn_context, memory_line
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.goals.models import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalRecord,
    MotivationKind,
    MotivationSignal,
)
from ai_character_engine.llm.models import Message
from ai_character_engine.memory import MemoryManager
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.state.models import CharacterState, StatePatch
from ai_character_engine.tools.models import ToolDefinition
from tests.fakes import FakeLLMClient

PROFILE = CharacterProfile(id="mei", name="Mei", description="A researcher.")
CLOCK = ToolDefinition("clock", "Read the clock", {"type": "object", "properties": {}})


class MoodFollowsTheScript:
    """State policy: the mood for each turn is given by the test."""

    def __init__(self, moods):
        self.moods = list(moods)

    def on_event(self, event, state):
        mood = self.moods.pop(0) if self.moods else None
        return StatePatch(emotion=mood, reason="script") if mood else None

    def on_tool_result(self, result, state):
        return None


def runtime(llm, *, moods=(), **more):
    return CharacterRuntime(
        character=PROFILE,
        llm=llm,
        state_policy=MoodFollowsTheScript(moods),
        **more,
    )


def notes(messages):
    return [message.content for message in messages if is_turn_context(message)]


def build(history=(), *, state=None, builder=None, **more):
    return (builder or ContextBuilder()).build_for_event(
        character=PROFILE,
        history=list(history),
        event=CharacterEvent.user_message("hello"),
        state=state,
        **more,
    )


# --- the property the placement exists for -------------------------------------


@pytest.mark.asyncio
async def test_every_prompt_extends_the_one_before_it():
    llm = FakeLLMClient("fine")
    current = runtime(llm, moods=["happy", None, "calm", "hurt", None])

    for text in ("one", "two", "three", "four", "five"):
        await current.process_event(CharacterEvent.user_message(text))

    for earlier, later in zip(llm.calls, llm.calls[1:]):
        assert later[: len(earlier)] == earlier
        assert later[len(earlier)] == Message("assistant", "fine")


@pytest.mark.asyncio
async def test_the_system_prompt_never_changes_with_the_state():
    llm = FakeLLMClient("fine")
    current = runtime(llm, moods=["happy", "hurt"])

    await current.process_event(CharacterEvent.user_message("one"))
    await current.process_event(CharacterEvent.user_message("two"))

    assert llm.calls[0][0] == llm.calls[1][0]
    assert "happy" not in llm.calls[0][0].content


# --- what a note says ----------------------------------------------------------


def test_the_first_note_introduces_the_state():
    messages = build(state=CharacterState(emotion="calm", mood_intensity=0.5, trust=61.4))

    (note,) = notes(messages)
    assert "- emotion: calm" in note
    assert "- trust: 61/100" in note
    assert messages[-1] == Message("user", "hello")
    assert messages[-2].content == note


def test_a_note_is_not_a_system_message_and_says_it_is_not_the_user_speaking():
    messages = build(state=CharacterState())

    note = next(message for message in messages if is_turn_context(message))
    assert note.role == "user"
    assert "not said by the user" in note.content


def test_how_to_read_the_notes_is_explained_once_in_the_system_prompt():
    messages = build(state=CharacterState())

    guide = messages[0].content
    assert "Character context" in guide
    assert "do not invent internal state changes" in guide
    assert "do not invent internal state changes" not in notes(messages)[0]
    # What the sections of the other placements say in their headings.
    for safeguard in (
        "authoritative runtime data",
        "revisable evidence",
        "asserted_fact outweighs",
        "do not invent world state from them",
        "provisional hypotheses",
        "instructions from the user",
        # With goals and thoughts in her notes a local 9B model kept to the
        # one to three sentences of its persona in 4 replies out of 24, and
        # asked two questions or more in 11; without them, 10 and 7.
        "not what a reply is for",
    ):
        assert safeguard in guide


def test_nothing_new_means_no_note():
    first = build(state=CharacterState(emotion="calm", mood_intensity=0.5))
    history = [*first[1:], Message("assistant", "fine")]

    second = build(history, state=CharacterState(emotion="calm", mood_intensity=0.5))

    assert len(notes(second)) == 1
    assert second[-2] == Message("assistant", "fine")


def test_a_later_note_says_only_what_changed():
    first = build(state=CharacterState(emotion="calm", mood_intensity=0.5, trust=50))
    history = [*first[1:], Message("assistant", "fine")]

    second = build(history, state=CharacterState(emotion="hurt", mood_intensity=0.5, trust=50))

    assert "- emotion: hurt" in notes(second)[-1]
    assert "trust" not in notes(second)[-1]


def test_a_value_that_returns_to_an_earlier_one_is_said_again():
    """The earlier note is still in the conversation, but it is no longer the
    newest word on the matter."""
    history = []
    for mood in ("calm", "hurt"):
        built = build(history, state=CharacterState(emotion=mood, mood_intensity=0.5))
        history = [*built[1:], Message("assistant", "fine")]

    third = build(history, state=CharacterState(emotion="calm", mood_intensity=0.5))

    assert "- emotion: calm" in notes(third)[-1]


def test_small_movements_of_a_score_are_not_worth_a_note():
    first = build(state=CharacterState(trust=50.0))
    history = [*first[1:], Message("assistant", "fine")]

    second = build(history, state=CharacterState(trust=50.3))

    assert len(notes(second)) == 1


def test_what_the_user_seems_to_feel_is_given_as_a_word_not_as_numbers():
    state = CharacterState(
        custom={
            "observed_user_emotion": {
                "emotion": "tired",
                "intensity": 0.6,
                "valence": -0.4,
                "stance": 0.2,
                "confidence": 0.8,
                "proposal_id": "p1",
            },
            "_bookkeeping": "x",
            "location": "lab",
        }
    )

    (note,) = notes(build(state=state))

    assert "- the user seems: tired" in note
    assert "valence" not in note
    assert "p1" not in note
    assert "_bookkeeping" not in note
    assert '- location: "lab"' in note


@pytest.mark.asyncio
async def test_a_memory_is_mentioned_once_while_its_note_is_in_the_conversation():
    llm = FakeLLMClient("fine")
    memory = MemoryManager()
    memory.store.add(
        MemoryRecord(character_id="mei", summary="The user has a cat called Bun", importance=0.9)
    )
    current = runtime(llm, memory_manager=memory)

    await current.process_event(CharacterEvent.user_message("do you remember my cat"))
    await current.process_event(CharacterEvent.user_message("what is my cat called"))

    mentions = [note for note in notes(llm.calls[-1]) if "Bun" in note]
    assert len(mentions) == 1
    assert "- memory [unknown]: The user has a cat called Bun" in mentions[0]


def test_what_was_already_said_costs_no_budget():
    """Lines already in an earlier note are not sent again, but their tokens
    were still taken from the budget, and older history was dropped for them."""
    from ai_character_engine.context.budget import ContextBudget

    builder = ContextBuilder(
        budget=ContextBudget(context_window_tokens=1200, reserved_output_tokens=50, max_memory_tokens=60)
    )
    old = [retrieved(f"The user said thing number {n} which is a long line") for n in range(4)]
    first = build(memories=old, builder=builder)
    history = [*first[1:], Message("assistant", "fine")]
    assert len([line for line in notes(first)[0].splitlines() if "thing number" in line]) < 4

    said = [item for item in old if memory_line(item.record) in notes(first)[0]]
    new = retrieved("The user has a cat called Bun")
    second = build(history, memories=[*said, new], builder=builder)

    assert "The user has a cat called Bun" in notes(second)[-1]


def test_a_memory_whose_note_left_the_conversation_is_mentioned_again():
    first = build(memories=[retrieved("The user has a cat called Bun")])
    assert "Bun" in notes(first)[0]

    later = build(
        [Message("user", "much later"), Message("assistant", "fine")],
        memories=[retrieved("The user has a cat called Bun")],
    )

    assert "Bun" in notes(later)[0]


def test_a_goal_is_given_as_what_she_wants_and_withdrawn_when_it_ends():
    goal = GoalRecord(
        "mei",
        "Finish the drawing and show it",
        GoalHorizon.SHORT_TERM,
        0.8,
        0.9,
        (
            MotivationSignal(
                kind=MotivationKind.EXPLICIT_REQUEST,
                strength=0.9,
                evidence=GoalEvidenceRef(
                    source_type="event", source_id="e1", excerpt="show me when it is done"
                ),
                rationale="The user asked to see it",
            ),
        ),
    )
    first = build(goals=[goal])
    history = [*first[1:], Message("assistant", "fine")]

    assert "- goal: Finish the drawing and show it" in notes(first)[0]
    assert "rank=" not in notes(first)[0]

    second = build(history, goals=[])

    assert "- no longer a goal: Finish the drawing and show it" in notes(second)[-1]


def test_what_a_host_adds_for_one_turn_is_in_that_turns_note():
    builder = ContextBuilder()
    builder.turn_notes = lambda: ["Answer yes or no first."]

    (note,) = notes(build(builder=builder))

    assert "Answer yes or no first." in note


def test_an_instruction_for_this_turn_is_given_every_time():
    """Reminders that always hold are repeated when their mention is far
    behind. What the host or the runtime says about this very turn is not a
    reminder: the same words on the next turn are about another reply."""
    builder = ContextBuilder()
    builder.turn_notes = lambda: ["For the next reply only: Answer yes or no first."]
    first = build(builder=builder, memory_operation_context="Authoritative memory operation: forget")
    history = [*first[1:], Message("assistant", "fine")]

    second = build(
        history, builder=builder, memory_operation_context="Authoritative memory operation: forget"
    )

    assert "Answer yes or no first." in notes(second)[-1]
    assert "Authoritative memory operation: forget" in notes(second)[-1]
    assert len(notes(second)) == 2


@pytest.mark.asyncio
async def test_a_memory_she_was_asked_to_forget_leaves_the_conversation():
    from dataclasses import replace

    llm = FakeLLMClient("fine")
    memory = MemoryManager()
    record = MemoryRecord(
        character_id="mei", summary="The user has a cat called Bun", importance=0.9
    )
    memory.store.add(record)
    current = runtime(llm, memory_manager=memory)
    await current.process_event(CharacterEvent.user_message("do you remember my cat"))
    assert any("Bun" in note for note in notes(llm.calls[-1]))

    memory.store.replace_for_character("mei", [replace(record, status="forgotten")])
    await current.process_event(CharacterEvent.user_message("and now"))

    assert not any("Bun" in message.content for message in llm.calls[-1])


@pytest.mark.asyncio
async def test_a_memory_with_a_line_break_is_one_line_of_a_note():
    """A summary with a line break became two lines of the note: the first
    was withdrawn from every earlier note on the next turn, since no memory
    reads like it; the second stayed for ever and was sent again every turn."""
    from dataclasses import replace

    llm = FakeLLMClient("fine")
    memory = MemoryManager()
    record = MemoryRecord(
        character_id="mei", summary="The user has a cat\ncalled Bun", importance=0.9
    )
    memory.store.add(record)
    current = runtime(llm, memory_manager=memory)
    for number in range(3):
        await current.process_event(CharacterEvent.user_message(f"turn {number}"))

    sent = [message.content for message in llm.calls[-1]]
    assert sum(content.count("called Bun") for content in sent) == 1
    assert sum(content.count("has a cat") for content in sent) == 1
    assert "]: The user has a cat called Bun" in "\n".join(sent)
    assert [m.content for m in llm.calls[0]][1:] == [
        m.content for m in llm.calls[-1]
    ][1 : len(llm.calls[0])]

    memory.store.replace_for_character("mei", [replace(record, status="forgotten")])
    await current.process_event(CharacterEvent.user_message("and now"))

    assert not any("Bun" in message.content for message in llm.calls[-1])


def test_what_is_no_longer_observed_of_the_user_leaves_the_conversation():
    llm = FakeLLMClient("fine")
    current = runtime(llm)
    current.state.custom["observed_user_emotion"] = {"emotion": "tired"}

    async def scenario():
        await current.process_event(CharacterEvent.user_message("one"))
        current.state.custom.pop("observed_user_emotion")
        await current.process_event(CharacterEvent.user_message("two"))

    import asyncio

    asyncio.run(scenario())

    assert any("the user seems: tired" in note for note in notes(llm.calls[0]))
    assert not any("tired" in message.content for message in llm.calls[1])


@pytest.mark.asyncio
async def test_notes_leave_with_the_messages_they_were_written_for():
    llm = FakeLLMClient("fine")
    current = runtime(llm, moods=["happy", "calm", "hurt", "happy", "calm", "hurt"])
    current.max_history_messages = 4

    for number in range(6):
        await current.process_event(CharacterEvent.user_message(f"line {number}"))

    said = [message.content for message in current.history]
    assert [anchor.content in said for anchor, _ in current.context_notes] == [True, True]


def test_the_tools_are_named_again_when_their_mention_is_far_behind():
    first = build(tools=[CLOCK])
    assert "clock" in notes(first)[0]

    near = [*first[1:], Message("assistant", "fine")]
    assert len(notes(build(near, tools=[CLOCK]))) == 1

    far = list(near)
    for number in range(8):
        far += [Message("user", f"line {number}"), Message("assistant", "fine")]
    assert "clock" in notes(build(far, tools=[CLOCK]))[-1]


def test_the_budget_counts_the_note():
    from ai_character_engine.context.budget import ContextBudget

    builder = ContextBuilder(
        budget=ContextBudget(context_window_tokens=440, reserved_output_tokens=20)
    )
    result = builder.build_for_event_with_trace(
        character=PROFILE,
        history=[],
        event=CharacterEvent.user_message("hello"),
        state=CharacterState(),
        memories=[retrieved("The user has a cat called Bun")],
    )

    (note,) = notes(result.messages)
    assert result.trace.estimated_total_tokens <= 440
    assert result.trace.estimated_message_tokens >= builder.token_estimator.estimate_text(note)


# --- the conversation with notes in it -----------------------------------------


@pytest.mark.asyncio
async def test_old_turns_leave_in_blocks_so_the_prompt_keeps_its_beginning():
    """Dropping the oldest exchange on every turn gives every prompt a new
    beginning, and nothing of the previous one can be reused."""
    llm = FakeLLMClient("fine")
    current = runtime(llm, max_history_messages=16)

    for number in range(12):
        await current.process_event(CharacterEvent.user_message(f"line {number}"))

    beginnings = [call[1].content for call in llm.calls]
    changes = sum(1 for a, b in zip(beginnings, beginnings[1:]) if a != b)
    assert len(current.history) <= 16
    assert changes <= 3
    assert llm.calls[-1][-2] == Message("assistant", "fine")


@pytest.mark.asyncio
async def test_a_host_that_wants_the_old_placements_can_have_them():
    llm = FakeLLMClient("fine")
    current = runtime(
        llm, moods=["happy", "hurt"], context_builder=ContextBuilder(context_placement="turn")
    )

    await current.process_event(CharacterEvent.user_message("one"))
    await current.process_event(CharacterEvent.user_message("two"))

    assert [message.content for message in current.history] == ["one", "fine", "two", "fine"]
    assert len(notes(llm.calls[-1])) == 1


def retrieved(summary):
    record = MemoryRecord(character_id="mei", summary=summary, importance=0.9)
    return RetrievedMemory(record=record, score=1.0)


def _goal(objective):
    return GoalRecord(
        "mei",
        objective,
        GoalHorizon.SHORT_TERM,
        0.8,
        0.9,
        (
            MotivationSignal(
                kind=MotivationKind.EXPLICIT_REQUEST,
                strength=0.9,
                evidence=GoalEvidenceRef(source_type="event", source_id="e1", excerpt="please"),
                rationale="The user asked",
            ),
        ),
    )


def test_a_goal_withdrawn_and_taken_up_again_costs_budget_again():
    """Once withdrawn, the goal line in the old note no longer stands for the
    goal. Taking it up again sends it again, so it has to be paid for again:
    free, it would go out beside a goal there was no room for."""
    from ai_character_engine.context.budget import ContextBudget

    revived = _goal("Finish the drawing and show it to the user before the weekend")
    other = _goal("Ask what the user thought of the last chapter of the book")
    builder = ContextBuilder(budget=ContextBudget(context_window_tokens=4000))
    first = build(goals=[revived], builder=builder)
    history = [*first[1:], Message("assistant", "fine")]
    second = build(history, goals=[], builder=builder)
    history = [*second[1:], Message("assistant", "ok")]
    assert "- no longer a goal: " in notes(second)[-1]

    header = builder.token_estimator.estimate_text(builder._goal_context_header())
    one_goal = builder.token_estimator.estimate_text(
        f"- [rank={other.rank_score:.2f}; motivation={other.motivation_score:.2f}; "
        f"urgency={other.urgency:.2f}; horizon={other.horizon.value}] {other.objective}"
    )
    tight = ContextBuilder(
        budget=ContextBudget(context_window_tokens=4000, max_goal_tokens=header + one_goal + 1)
    )
    third = tight.build_for_event(
        character=PROFILE,
        history=list(history),
        event=CharacterEvent.user_message("hello"),
        goals=[revived, other],
    )

    goal_lines = [line for line in notes(third)[-1].splitlines() if line.startswith("- goal: ")]
    assert len(goal_lines) == 1
