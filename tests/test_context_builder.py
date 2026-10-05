from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.context.builder import ContextBuilder


def test_system_prompt_contains_character_data() -> None:
    character = CharacterProfile(
        id="alice",
        name="Alice",
        description="A test character.",
        personality=["calm"],
        speaking_style=["short sentences"],
        rules=["stay in character"],
    )
    messages = ContextBuilder().build(
        character=character,
        history=[],
        user_message="Hello",
    )

    assert messages[0].role == "system"
    assert "Alice" in messages[0].content
    assert "calm" in messages[0].content
    assert messages[-1].role == "user"
    assert messages[-1].content == "Hello"


from ai_character_engine import CharacterRuntime, StatePatch
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import Message
from ai_character_engine.memory.models import MemoryRecord, RetrievedMemory
from ai_character_engine.state.models import CharacterState
from tests.fakes import FakeLLMClient

CHARACTER = CharacterProfile(id="alice", name="Alice", description="A test character.")
HISTORY = [Message("user", "earlier"), Message("assistant", "earlier reply")]


def remembered(summary):
    return RetrievedMemory(
        record=MemoryRecord(character_id="alice", kind="fact", summary=summary, importance=0.8),
        score=0.9,
    )


# The tests below describe the "turn" placement: the whole context in one
# message that is never kept. tests/test_transcript_context.py describes the
# default.
def built(builder=None, *, state=None, memories=(), text="Hello"):
    return (builder or ContextBuilder(context_placement="turn")).build_for_event(
        character=CHARACTER,
        history=HISTORY,
        event=CharacterEvent.user_message(text),
        state=state,
        memories=memories,
    )


def test_what_changes_between_turns_never_touches_the_prompt_before_the_history():
    """Measured on a local 9B model through the host bridge: with state or
    retrieved memories in the system prompt, time to the first token grew from
    2 s to 5 s over eight turns, because every change makes the server re-read
    the whole conversation. With a system prompt that never changes it stayed
    at 2 s."""
    calm = built(state=CharacterState(emotion="calm", trust=50))
    hurt = built(
        state=CharacterState(emotion="hurt", trust=20),
        memories=(remembered("User is called Dawn"),),
        text="Another line",
    )

    assert calm[:-2] == hurt[:-2]
    assert len(calm) == len(HISTORY) + 3


def test_state_and_memories_sit_between_the_history_and_the_newest_message():
    messages = built(
        state=CharacterState(emotion="hurt", trust=20),
        memories=(remembered("User is called Dawn"),),
    )

    context = messages[-2]
    assert context.role == "user"
    assert "emotion: hurt" in context.content
    assert "User is called Dawn" in context.content
    assert "emotion: hurt" not in messages[0].content
    assert messages[1:3] == HISTORY


def test_the_context_is_not_a_system_message():
    """Measured by replaying whole conversations on a local 9B model (mean time
    to first token over turns 5-8): context in a system message placed before
    the newest message 4.6 s, the same context in a user-role message 1.9 s,
    no context 1.8 s. Chat templates may move every system message to the
    top, which changes the start of the prompt each turn."""
    roles = [message.role for message in built(state=CharacterState(emotion="hurt"))]

    assert roles == ["system", "user", "assistant", "user", "user"]


def test_the_newest_message_is_exactly_what_was_said():
    """Hosts and custom model clients read the last message as the user's
    words: the README example and the voice pipeline both do."""
    messages = built(state=CharacterState(emotion="hurt"))

    assert messages[-1] == Message("user", "Hello")


def test_the_model_is_told_the_context_was_not_said_by_the_user():
    context = built(state=CharacterState(emotion="hurt"))[-2]

    assert context.content.index("not said by the user") < context.content.index("emotion: hurt")


def test_without_anything_to_say_there_is_no_context_message():
    messages = built()

    assert [message.role for message in messages] == ["system", "user", "assistant", "user"]


def test_a_host_can_keep_everything_in_the_system_prompt():
    messages = built(
        ContextBuilder(context_placement="system"),
        state=CharacterState(emotion="hurt"),
        memories=(remembered("User is called Dawn"),),
    )

    assert "emotion: hurt" in messages[0].content
    assert "User is called Dawn" in messages[0].content
    assert messages[-1].content == "Hello"


def test_unknown_placement_is_refused():
    import pytest

    with pytest.raises(ValueError):
        ContextBuilder(context_placement="somewhere")


async def test_history_keeps_what_the_user_said_not_the_context_sent_with_it():
    class Moody:
        def on_event(self, event, state):
            return StatePatch(emotion="hurt")

        def on_tool_result(self, result, state):
            return None

    llm = FakeLLMClient("reply")
    runtime = CharacterRuntime(
        character=CHARACTER,
        llm=llm,
        state_policy=Moody(),
        context_builder=ContextBuilder(context_placement="turn"),
    )

    await runtime.run_turn("first")
    await runtime.run_turn("second")

    assert [message.content for message in runtime.history] == ["first", "reply", "second", "reply"]
    sent = llm.calls[1]
    assert [message.content for message in sent[1:3]] == ["first", "reply"]
    assert "emotion: hurt" in sent[-2].content
    assert sent[-1].content == "second"


from ai_character_engine.tools.models import ToolDefinition

CLOCK = ToolDefinition("get_current_time", "Get the current local time.", {"type": "object", "properties": {}})
SEARCH = ToolDefinition("web_search", "Search the web.", {"type": "object", "properties": {}})


def built_with_tools(tools, builder=None):
    return (builder or ContextBuilder(context_placement="turn")).build_for_event(
        character=CHARACTER,
        history=HISTORY,
        event=CharacterEvent.user_message("What time is it?"),
        tools=tools,
    )


def test_the_turn_context_reminds_the_model_which_tools_it_has():
    """Measured on a local 9B model with a 3000-character persona and one clock
    tool: asked the time six times it never called the tool and made a time up.
    With this note in the turn context it called the tool four times out of
    six, and never for small talk. The same note at the end of the system
    prompt changed nothing."""
    context = built_with_tools([CLOCK, SEARCH])[-2].content

    assert "get_current_time, web_search" in context
    assert "Never guess" in context


def test_without_tools_there_is_no_tool_note():
    messages = built_with_tools(None)

    assert all("Never guess" not in message.content for message in messages)


def test_the_tool_note_follows_the_context_wherever_the_host_put_it():
    messages = built_with_tools([CLOCK], ContextBuilder(context_placement="system"))

    assert "get_current_time" in messages[0].content
    assert messages[-1].content == "What time is it?"


def test_a_tight_budget_drops_the_tool_note_instead_of_failing_the_turn():
    from ai_character_engine.context.budget import ContextBudget

    roomy = ContextBuilder(
        budget=ContextBudget(context_window_tokens=400, reserved_output_tokens=20),
        context_placement="turn",
    )
    tight = ContextBuilder(
        budget=ContextBudget(context_window_tokens=150, reserved_output_tokens=20),
        context_placement="turn",
    )

    assert any("Never guess" in m.content for m in built_with_tools([CLOCK], roomy))
    assert all("Never guess" not in m.content for m in built_with_tools([CLOCK], tight))


def test_bookkeeping_in_custom_state_never_reaches_the_model():
    """The commit coordinator stores its own ids next to what it observed, and
    the whole custom state used to be printed into the prompt."""
    state = CharacterState(
        custom={
            "observed_user_emotion": {
                "emotion": "tired",
                "intensity": 0.8,
                "valence": -0.4,
                "stance": 1.0,
                "confidence": 0.9,
                "base_revision": 3,
                "source_task_id": "470b50f12f1e4a8096268286bf4ff52d",
                "proposal_id": "d4e869422238499ea1e82ad78525d734",
            },
            "_relationship_applied_observation": "d4e869422238499ea1e82ad78525d734",
            "location": "lab",
        }
    )

    context = built(state=state)[-2].content

    assert '"emotion": "tired"' in context
    assert '"location": "lab"' in context
    for leaked in ("d4e869422238499ea1e82ad78525d734", "470b50f12f1e4a8096268286bf4ff52d",
                   "base_revision", "_relationship_applied_observation"):
        assert leaked not in context


def test_the_time_of_an_observation_never_reaches_the_model():
    """The commit coordinator dates observed_user_emotion with the turn it
    came from (turn_ended_at); that timing is for staleness checks, not for
    the model to read aloud as a stray epoch float."""
    state = CharacterState(
        custom={
            "observed_user_emotion": {
                "emotion": "tired",
                "intensity": 0.8,
                "turn_ended_at": 1234567890.5,
            },
        }
    )

    context = ContextBuilder().build_state_context(state)

    assert '"emotion": "tired"' in context
    assert "turn_ended_at" not in context
    assert "1234567890" not in context


def test_the_turn_revision_of_an_observation_never_reaches_the_model():
    """turn_revision is the commit coordinator's own bookkeeping, like
    turn_ended_at: it must not reach the model as a stray revision number."""
    state = CharacterState(
        custom={
            "observed_user_emotion": {
                "emotion": "tired",
                "intensity": 0.8,
                "turn_revision": 42,
            },
        }
    )

    context = ContextBuilder().build_state_context(state)

    assert '"emotion": "tired"' in context
    assert "turn_revision" not in context
