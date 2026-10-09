"""CharacterCompanion: one object a host talks to.

It wires the pieces a host otherwise has to assemble and get right itself: the
streamed foreground turn, background cognition, commits that survive a slow
local model, a character whose state moves, per-conversation memory, and
persistence. Every scenario below was first met in a real host.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import (
    CHARACTER_MOODS,
    CharacterCompanion,
    CompanionClosed,
    CompanionSettings,
    CompanionSnapshot,
    TurnInterrupted,
)
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk, Message
from ai_character_engine.state.models import StatePatch
from ai_character_engine.tools.models import ToolCall, ToolDefinition
from ai_character_engine.companion.companion import ALREADY_SAID, NOT_AN_ASSISTANT
from tests.fakes import system_context

WARM = {
    "emotion": "glad",
    "intensity": 1.0,
    "valence": 0.9,
    "stance": 1.0,
    "confidence": 1.0,
    "evidence": ["thank you"],
}
NEUTRAL = {**WARM, "emotion": "calm", "valence": 0.0, "stance": 0.0, "evidence": []}


class Foreground:
    """The character's model. Streams a reply in two parts."""

    def __init__(self, parts=("Hello. ", "How are you?"), *, gate=None):
        self.parts = parts
        self.gate = gate
        self.calls: list[list[Message]] = []
        self.started = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return LLMResponse(text="".join(self.parts), model="foreground")

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        yield LLMStreamChunk(text=self.parts[0])
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        for part in self.parts[1:]:
            yield LLMStreamChunk(text=part)
        yield LLMStreamChunk(
            final=True, response=LLMResponse(text="".join(self.parts), model="foreground")
        )


class Worker:
    """A background model that answers with fixed JSON, or a function of the prompt."""

    running = 0
    most_at_once = 0
    order: list[str] = []

    def __init__(self, payload, *, gate=None, fail=False, name=""):
        self.payload = payload
        self.gate = gate
        self.fail = fail
        self.name = name
        self.calls = 0
        self.abandoned = 0

    @classmethod
    def reset(cls):
        cls.running, cls.most_at_once, cls.order = 0, 0, []

    async def generate(self, messages, *, tools=None):
        self.calls += 1
        Worker.order.append(self.name)
        Worker.running += 1
        Worker.most_at_once = max(Worker.most_at_once, Worker.running)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            if self.fail:
                raise RuntimeError("model is down")
            payload = self.payload(messages) if callable(self.payload) else self.payload
            return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="worker")
        except asyncio.CancelledError:
            self.abandoned += 1
            raise
        finally:
            Worker.running -= 1


def warm_only_when_thanked(messages):
    thanked = "Latest event/user content:\nthank you for last night" in messages[1].content
    return WARM if thanked else NEUTRAL


def companion(tmp_path, workers=None, *, llm=None, clock=None, **settings):
    defaults = {
        "emotion_every": 0,
        "memory_every": 0,
        "goal_every": 0,
        "reflection_every": 0,
        "summary_every": 0,
        "self_memory_every": 0,
        "mood_every": 0,
    }
    defaults.update(settings)
    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=llm or Foreground(),
        background_llm=workers or {},
        storage_dir=tmp_path / "engine",
        settings=CompanionSettings(**defaults),
        clock=clock,
    )


def run(coroutine):
    Worker.reset()
    return asyncio.run(coroutine)


async def until(condition, *, steps=500):
    for _ in range(steps):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


# --- the foreground turn -------------------------------------------------------


def test_a_reply_is_streamed_and_returned(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        heard = []
        result = await current.reply("hi", conversation_id="a", on_text_delta=heard.append)
        await current.close()
        return heard, result.text

    assert run(scenario()) == (["Hello. ", "How are you?"], "Hello. How are you?")


def test_a_host_with_one_conversation_needs_no_conversation_id(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("call me Alex")
        await current.reply("what is my name")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    assert "call me Alex" in run(scenario())


def test_each_conversation_keeps_its_own_history(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first in a", conversation_id="a")
        await current.reply("first in b", conversation_id="b")
        await current.reply("second in a", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "first in a" in sent
    assert "first in b" not in sent
    assert sent[-1] == "second in a"


def test_coming_back_to_a_conversation_continues_its_prompt(tmp_path):
    """What the model was sent is the conversation plus the notes written into
    it. Both have to come back, or the server reads everything again."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first in a", conversation_id="a")
        before = llm.calls[-1]
        await current.reply("first in b", conversation_id="b")
        await current.reply("second in a", conversation_id="a")
        await current.close()
        return before, llm.calls[-1]

    before, after = run(scenario())
    assert after[: len(before)] == before


def test_coming_back_to_the_oldest_conversation_kept_finds_it(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, conversations_kept=2)
        for conversation in ("b", "c", "a"):
            await current.reply(f"first in {conversation}", conversation_id=conversation)
        await current.reply("second in b", conversation_id="b")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    assert "first in b" in run(scenario())


def test_a_loaded_conversation_is_continued_from_what_the_host_had(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        current.load_conversation(
            "a",
            [
                Message("assistant", "an orphan reply"),
                Message("user", "where were we"),
                Message("assistant", "at the time machine"),
            ],
        )
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "at the time machine" in sent
    assert "an orphan reply" not in sent


def test_a_host_can_ask_whether_a_conversation_still_has_to_be_loaded(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        before = current.has_conversation("a")
        current.load_conversation("a", [Message("user", "old"), Message("assistant", "older")])
        loaded = current.has_conversation("a")
        await current.reply("hello", conversation_id="b")
        await current.reply("hello", conversation_id="c")
        await current.close()
        return before, loaded, current.has_conversation("b"), current.has_conversation("c")

    assert run(scenario()) == (False, True, True, True)


def test_loading_again_does_not_erase_what_was_said_since(tmp_path):
    """A page reload makes the host load the conversation again."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        current.load_conversation("a", [Message("user", "old"), Message("assistant", "older")])
        await current.reply("new line", conversation_id="a")
        current.load_conversation("a", [])
        await current.reply("another", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    assert "new line" in run(scenario())


def test_tools_run_and_text_still_arrives_as_it_is_generated(tmp_path):
    class ToolUsing(Foreground):
        async def stream_generate(self, messages, *, tools=None):
            self.calls.append(list(messages))
            if messages[-1].role == "tool":
                yield LLMStreamChunk(text="It is 12:34.")
                yield LLMStreamChunk(
                    final=True, response=LLMResponse(text="It is 12:34.", model="foreground")
                )
                return
            yield LLMStreamChunk(text="Let me check. ")
            yield LLMStreamChunk(
                final=True,
                response=LLMResponse(
                    text="Let me check. ",
                    tool_calls=(ToolCall("one", "clock", {}),),
                    model="foreground",
                ),
            )

    async def scenario():
        current = companion(tmp_path, llm=ToolUsing())
        current.tools.register(
            ToolDefinition("clock", "Read the clock", {"type": "object", "properties": {}}),
            lambda: "12:34",
        )
        heard = []
        result = await current.reply("what time is it", conversation_id="a", on_text_delta=heard.append)
        await current.close()
        return heard, result.text, [r.output for r in result.tool_results]

    assert run(scenario()) == (["Let me check. ", "It is 12:34."], "It is 12:34.", ["12:34"])


def test_a_note_from_the_host_reaches_this_reply_only(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply(
            "is it raining", conversation_id="a", notes=("Answer yes or no first.",)
        )
        with_note = [message.content for message in llm.calls[-1]]
        await current.reply("and tomorrow", conversation_id="a")
        await current.close()
        return with_note, [message.content for message in llm.calls[-1]]

    with_note, afterwards = run(scenario())
    assert "For the next reply only: Answer yes or no first." in with_note[-2]
    assert with_note[-1] == "is it raining"
    # It stays where it was said, so that the next prompt extends this one,
    # and is not said again.
    assert afterwards[: len(with_note)] == with_note
    assert "Answer yes or no first." not in "".join(afterwards[len(with_note) :])


def test_an_instruction_passed_on_every_turn_reaches_every_turn(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        newest = []
        for text in ("is it raining", "is it cold", "is it late"):
            await current.reply(text, conversation_id="a", notes=("Answer yes or no first.",))
            newest.append(llm.calls[-1][-2].content)
        await current.close()
        return newest

    for note in run(scenario()):
        assert "For the next reply only: Answer yes or no first." in note


def test_what_the_host_no_longer_knows_is_taken_back(tmp_path):
    """A host whose user corrected or deleted a memory. Notes stay in the
    conversation, so the old line has to be taken out of the note it is in."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply(
            "hello", conversation_id="a", notes=["- about the user: lives in Taipei"]
        )
        await current.reply(
            "and now", conversation_id="a", notes=["- about the user: moved to Taichung"]
        )
        corrected = "".join(message.content for message in llm.calls[-1])
        await current.reply("and then", conversation_id="a")
        await current.close()
        return corrected, "".join(message.content for message in llm.calls[-1])

    corrected, cleared = run(scenario())
    assert "Taipei" not in corrected
    assert "moved to Taichung" in corrected
    assert "Taichung" not in cleared


def test_what_the_host_knows_about_her_is_said_once_not_every_turn(tmp_path):
    """A host with a memory of its own hands it over with every turn. A note
    that starts with "- " is knowledge, not an instruction for one reply."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        known = ["- about the user: has a cat called Bun"]
        await current.reply("hello", conversation_id="a", notes=known)
        first = system_context(llm.calls[-1])
        known.append("- about the user: works at night")
        await current.reply("and now", conversation_id="a", notes=known)
        await current.close()
        return first, [m.content for m in llm.calls[-1]]

    first, second = run(scenario())
    assert "- about the user: has a cat called Bun" in first
    assert "For the next reply only" not in first
    assert "".join(second).count("has a cat called Bun") == 1
    assert "- about the user: works at night" in second[-2]


def test_the_host_can_keep_the_reply_as_it_displayed_it(tmp_path):
    """Hosts normalize what they show: script variant, stage directions."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hi", conversation_id="a")
        current.replace_reply("Hello.")
        await current.reply("again", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "Hello." in sent
    assert "Hello. How are you?" not in sent


def test_a_host_shared_by_several_callers_prepares_and_tidies_inside_the_turn(tmp_path):
    """Two callers at once: what one of them does between its turns happens
    during the other one's turn. Rewriting the character or registering tools
    there is refused or lands in the wrong turn, and replace_reply() afterwards
    may already meet the next reply."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        pianist = CharacterProfile(id="mei", name="Mei", description="A pianist.")

        def prepare():
            current.character = pianist

        first = asyncio.ensure_future(current.reply("first", conversation_id="a"))
        await llm.started.wait()
        second = asyncio.ensure_future(
            current.reply(
                "second",
                conversation_id="b",
                before_turn=prepare,
                remember_as=lambda text: text.upper(),
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
        llm.gate.set()
        await first
        result = await second
        await current.reply("third", conversation_id="b")
        await current.close()
        return llm.calls[0][0].content, llm.calls[1][0].content, result.text, [
            message.content for message in llm.calls[-1]
        ]

    first_system, second_system, returned, sent = run(scenario())
    assert "A researcher." in first_system
    assert "A pianist." in second_system
    assert returned == "Hello. How are you?"
    assert "HELLO. HOW ARE YOU?" in sent


def test_the_character_can_be_rewritten_between_turns(tmp_path):
    """A host that edits the persona while running must not need a restart."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hi", conversation_id="a")
        current.character = CharacterProfile(id="mei", name="Mei", description="A pianist.")
        await current.reply("what do you do", conversation_id="a")
        with pytest.raises(ValueError):
            current.character = CharacterProfile(id="someone-else", name="X", description="Y")
        await current.close()
        return llm.calls[-1][0].content, current.snapshot().trust

    system, trust = run(scenario())
    assert "A pianist." in system
    assert "A researcher." not in system
    assert trust == pytest.approx(50.6)


def test_a_picture_is_described_to_her_and_not_kept(tmp_path):
    from ai_character_engine.vision import VisionPipeline
    from ai_character_engine.vision.models import ImageInput, VisionAnalysis, VisionFrame
    from ai_character_engine.vision.providers import CallableVisionProvider
    from tests.test_vision import PNG

    async def look(image, prompt):
        return VisionAnalysis(text="A red square next to a blue circle.", provider="fake")

    async def scenario():
        llm = Foreground()
        current = CharacterCompanion(
            character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
            llm=llm,
            background_llm={},
            vision=VisionPipeline(provider=CallableVisionProvider(look)),
        )
        blind = companion(tmp_path)
        frame = VisionFrame(image=ImageInput.from_bytes(PNG, mime_type="image/png"))
        await current.reply("what is this", conversation_id="a", frames=(frame,))
        # Her words, then what the picture shows, as the last two messages.
        seen = "\n".join(message.content for message in llm.calls[-1][-2:])
        await current.reply("and now", conversation_id="a")
        await current.close()
        await blind.close()
        return seen, [message.content for message in llm.calls[-1]], current.sees, blind.sees

    seen, afterwards, sees, blind = run(scenario())
    assert "what is this" in seen
    assert "A red square next to a blue circle." in seen
    assert "what is this" in afterwards
    assert not any("red square" in content for content in afterwards)
    assert (sees, blind) == (True, False)


# --- a character whose state moves ---------------------------------------------


def test_every_turn_builds_trust_and_survives_a_restart(tmp_path):
    async def scenario():
        first = companion(tmp_path)
        await first.reply("hello", conversation_id="a")
        await first.reply("nice weather", conversation_id="a")
        await first.close()
        return companion(tmp_path).snapshot()

    assert run(scenario()).trust == pytest.approx(50.6)


def test_a_warm_turn_lifts_the_mood_before_the_next_turn(tmp_path):
    async def scenario():
        current = companion(tmp_path, {"emotion": Worker(WARM)}, emotion_every=1)
        await current.reply("thank you for last night", conversation_id="a")
        await current.settle()
        snapshot = current.snapshot()
        llm = current.runtime.llm
        await current.reply("anyway", conversation_id="a")
        await current.close()
        return snapshot, system_context(llm.calls[-1])

    snapshot, context = run(scenario())
    assert snapshot.emotion == "happy"
    assert snapshot.favorability == pytest.approx(54.0)
    assert "emotion: happy" in context


def test_a_late_result_is_used_when_nothing_newer_of_its_kind_is_coming(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        worker = Worker(warm_only_when_thanked, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, emotion_every=2)
        await current.reply("hi", conversation_id="a")
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("by the way", conversation_id="a")
        gate.set()
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    assert run(scenario()).favorability == pytest.approx(54.0)


def test_a_late_result_gives_way_to_a_newer_one_of_its_kind(tmp_path):
    """The coordinator accepts one emotion observation per revision. An old one
    moved onto the current revision takes that place, and the observation that
    belongs there is then refused as a conflict."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(warm_only_when_thanked, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, emotion_every=1)
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("by the way", conversation_id="a")
        gate.set()
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return worker.abandoned, snapshot

    abandoned, snapshot = run(scenario())
    assert abandoned >= 1
    assert snapshot.favorability == pytest.approx(50.0)
    # The warm observation would have made her happy; the unremarkable one
    # that took its place leaves her mood alone.
    assert snapshot.emotion == "neutral"


def test_a_turn_that_asked_for_no_background_work_replaces_nothing(tmp_path):
    """Proactive remarks are turns too, but nothing is scheduled after them."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(warm_only_when_thanked, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, emotion_every=1)
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("(a remark of her own)", conversation_id="a", skip_memory=True)
        gate.set()
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    assert run(scenario()).favorability == pytest.approx(54.0)


def test_a_finished_result_still_gives_way_to_the_newer_job(tmp_path):
    """A job that finished while she was replying cannot be cancelled any more
    when the job of that reply is scheduled."""

    async def scenario():
        gate = asyncio.Event()
        talking = asyncio.Event()
        talking.set()
        llm = Foreground(gate=talking)
        worker = Worker(warm_only_when_thanked, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, llm=llm, emotion_every=1)
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls == 1)
        # It gives way to this turn, which schedules nothing, and starts again.
        await current.reply("(a remark of her own)", conversation_id="a", skip_memory=True)
        await until(lambda: worker.calls == 2)
        talking.clear()
        llm.started.clear()
        turn = asyncio.ensure_future(current.reply("by the way", conversation_id="a"))
        await llm.started.wait()
        gate.set()
        await until(lambda: Worker.running == 0)
        talking.set()
        await turn
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    snapshot = run(scenario())
    assert snapshot.favorability == pytest.approx(50.0)
    # The warm observation would have made her happy; the unremarkable one
    # that took its place leaves her mood alone.
    assert snapshot.emotion == "neutral"


def test_a_result_that_arrives_too_late_is_dropped(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        worker = Worker(warm_only_when_thanked, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, emotion_every=4, max_turns_late=1)
        for text in ("one", "two", "three"):
            await current.reply(text, conversation_id="a")
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("five", conversation_id="a")
        await current.reply("six", conversation_id="a")
        gate.set()
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    snapshot = run(scenario())
    assert snapshot.favorability == pytest.approx(50.0)
    assert snapshot.emotion == "neutral"


# --- sharing one model ---------------------------------------------------------


def test_background_work_waits_while_the_character_is_replying(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        gate.set()
        llm = Foreground(gate=gate)
        worker = Worker(NEUTRAL)
        current = companion(tmp_path, {"emotion": worker}, llm=llm, emotion_every=1)
        await current.reply("earlier", conversation_id="a")
        gate.clear()
        llm.started.clear()
        turn = asyncio.ensure_future(current.reply("now", conversation_id="a"))
        await llm.started.wait()
        for _ in range(50):
            await asyncio.sleep(0)
        calling_the_model_meanwhile = Worker.running
        gate.set()
        await turn
        await current.settle()
        await current.close()
        return calling_the_model_meanwhile, worker.calls >= 1

    assert run(scenario()) == (0, True)


def test_the_mood_never_waits_for_other_background_work(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        workers = {
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}, gate=gate),
            "emotion": Worker(WARM),
        }
        current = companion(tmp_path, workers, emotion_every=2, memory_every=1)
        await current.reply("first", conversation_id="a")
        await until(lambda: workers["memory"].calls)
        await current.reply("thank you for last night", conversation_id="a")
        await until(lambda: current.snapshot().emotion == "happy")
        mood_while_memory_still_runs = current.snapshot().emotion
        gate.set()
        await current.settle()
        await current.close()
        return mood_while_memory_still_runs

    assert run(scenario()) == "happy"


def test_after_a_reply_the_mood_is_read_before_other_work_starts(tmp_path):
    """On one local model two calls at once both take longer. The mood decides
    the next reply and has the least time, so it goes first."""

    async def scenario():
        gate = asyncio.Event()
        workers = {
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}),
            "emotion": Worker(NEUTRAL, gate=gate),
        }
        current = companion(tmp_path, workers, emotion_every=1, memory_every=1)
        await current.reply("hello", conversation_id="a")
        await until(lambda: workers["emotion"].calls)
        for _ in range(50):
            await asyncio.sleep(0)
        started_meanwhile = workers["memory"].calls
        gate.set()
        await current.settle()
        await current.close()
        return started_meanwhile, workers["memory"].calls

    assert run(scenario()) == (0, 1)


def test_work_under_way_is_abandoned_when_she_starts_to_reply_but_only_once(tmp_path):
    """Goal and reflection calls take longer than the pause between two turns;
    abandoned every time, they would never finish."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(cat_fact_when_quoted, gate=gate)
        current = companion(tmp_path, {"memory": worker}, memory_every=4)
        for text in ("one", "two", "three"):
            await current.reply(text, conversation_id="a")
        await current.reply("I have a cat called Bun", conversation_id="a")
        await until(lambda: Worker.running == 1)
        await current.reply("five", conversation_id="a")
        await until(lambda: worker.calls == 2)
        after_one_reply = worker.abandoned
        await current.reply("six", conversation_id="a")
        for _ in range(50):
            await asyncio.sleep(0)
        after_two_replies = worker.abandoned
        gate.set()
        await current.settle()
        found = current.memories("a")
        await current.close()
        return after_one_reply, after_two_replies, worker.calls, found

    assert run(scenario()) == (1, 1, 2, ["The user has a cat called Bun"])


def test_while_one_job_has_the_model_the_next_one_waits(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        workers = {
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}, gate=gate),
            "goal": Worker({"goals": [], "confidence": 0, "evidence": []}),
        }
        current = companion(tmp_path, workers, memory_every=1, goal_every=1)
        await current.reply("hello", conversation_id="a")
        await until(lambda: workers["memory"].calls)
        for _ in range(50):
            await asyncio.sleep(0)
        meanwhile = workers["goal"].calls
        gate.set()
        await current.settle()
        await current.close()
        return meanwhile, workers["goal"].calls

    assert run(scenario()) == (0, 1)


def test_the_other_background_jobs_use_the_model_one_at_a_time(tmp_path):
    async def scenario():
        workers = {
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}, name="memory"),
            "goal": Worker({"goals": [], "confidence": 0, "evidence": []}, name="goal"),
            "reflection": Worker(
                {"insight": "i", "belief_candidate": None, "confidence": 0.9, "evidence": []},
                name="reflection",
            ),
        }
        current = companion(tmp_path, workers, memory_every=1, goal_every=1, reflection_every=1)
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await current.close()
        return list(Worker.order), Worker.most_at_once

    order, most_at_once = run(scenario())
    assert order == ["memory", "goal", "reflection"]
    assert most_at_once == 1


def test_a_worker_that_is_down_never_reaches_the_conversation(tmp_path):
    async def scenario():
        current = companion(tmp_path, {"emotion": Worker(WARM, fail=True)}, emotion_every=1)
        result = await current.reply("thank you", conversation_id="a")
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return result.text, snapshot

    text, snapshot = run(scenario())
    assert text == "Hello. How are you?"
    assert snapshot.emotion == "neutral"


# --- memory, goals and thoughts ------------------------------------------------


def cat_fact_when_quoted(messages):
    if "I have a cat called Bun" not in messages[1].content.split("User lines to extract from:")[1]:
        return {"items": [], "confidence": 0.5, "evidence": []}
    return {
        "items": [
            {
                "summary": "The user has a cat called Bun",
                "kind": "fact",
                "importance": 0.8,
                "confidence": 0.9,
                "evidence": "I have a cat called Bun",
            }
        ],
        "confidence": 0.9,
        "evidence": [],
    }


def test_what_was_learned_is_in_the_context_of_that_conversation_only(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(
            tmp_path, {"memory": Worker(cat_fact_when_quoted)}, llm=llm, memory_every=1
        )
        await current.reply("I have a cat called Bun", conversation_id="a")
        await current.settle()
        await current.reply("what is my cat called", conversation_id="a")
        in_a = system_context(llm.calls[-1])
        await current.reply("what is my cat called", conversation_id="b")
        in_b = system_context(llm.calls[-1])
        await current.close()
        return in_a, in_b, current.memories("a"), current.memories("b")

    in_a, in_b, stored_a, stored_b = run(scenario())
    assert "The user has a cat called Bun" in in_a
    assert "Bun" not in in_b
    assert stored_a == ["The user has a cat called Bun"]
    assert stored_b == []


def test_an_interrupted_turn_does_not_hide_what_was_said_before_it(tmp_path):
    """Memory runs every second turn and reads what was said since its last
    run. An interrupted turn leaves a line in the conversation without being a
    turn; counting lines instead of looking for the last one read skipped the
    line before it."""

    async def scenario():
        talking = asyncio.Event()
        talking.set()
        llm = Foreground(gate=talking)
        current = companion(
            tmp_path, {"memory": Worker(cat_fact_when_quoted)}, llm=llm, memory_every=2
        )
        await current.reply("I have a cat called Bun", conversation_id="a")
        talking.clear()
        llm.started.clear()
        turn = asyncio.ensure_future(current.reply("tell me a story", conversation_id="a"))
        await llm.started.wait()
        current.interrupt("Hello.")
        with pytest.raises(TurnInterrupted):
            await turn
        talking.set()
        await current.reply("ok", conversation_id="a")
        await current.settle()
        found = current.memories("a")
        await current.close()
        return found

    assert run(scenario()) == ["The user has a cat called Bun"]


def test_opening_an_old_conversation_does_not_send_all_of_it_to_be_remembered_again(tmp_path):
    seen = []

    def nothing_new(messages):
        seen.append(messages[1].content.split("User lines to extract from:")[1])
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        current = companion(tmp_path, {"memory": Worker(nothing_new)}, memory_every=2)
        await current.reply("a new conversation", conversation_id="new")
        old = []
        for number in range(10):
            old += [Message("user", f"old line {number}"), Message("assistant", "fine")]
        current.load_conversation("old", old)
        await current.reply("back again", conversation_id="old")
        # Every second turn of each conversation: this one is due.
        await current.reply("and again", conversation_id="old")
        await current.settle()
        await current.close()

    run(scenario())
    (lines,) = seen
    assert "back again" in lines and "and again" in lines
    assert "old line 9" not in lines


def test_what_she_thinks_is_written_in_the_language_the_host_names(tmp_path):
    """The workers' instructions are in English and ask for the language the
    user writes in. A small model has to work that out from the transcript and
    does not always: in one conversation in Chinese, four memories out of five
    came back in English. A host that knows the language says so."""
    asked = []

    def nothing(messages):
        asked.append(messages[1].content)
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        named = companion(
            tmp_path / "named", {"memory": Worker(nothing)}, memory_every=1, language="繁體中文"
        )
        unnamed = companion(tmp_path / "unnamed", {"memory": Worker(nothing)}, memory_every=1)
        for current in (named, unnamed):
            await current.reply("hello", conversation_id="a")
            await current.settle()
            await current.close()

    run(scenario())
    named, unnamed = asked
    assert named.rstrip().endswith("Text values must be written in 繁體中文.")
    assert "the language the user writes in" in unnamed


def test_the_host_can_show_her_memory_to_its_user_and_take_the_edit_back(tmp_path):
    """A memory page: the user reads what she remembers of a conversation,
    corrects one line, deletes one and adds one."""

    async def scenario():
        llm = Foreground()
        current = companion(
            tmp_path, {"memory": Worker(cat_fact_when_quoted)}, llm=llm, memory_every=1
        )
        await current.reply("I have a cat called Bun", conversation_id="a")
        await current.settle()
        await current.reply("what is my cat called", conversation_id="a")
        before = current.memories("a")

        current.rewrite_memories("a", ["The user has a dog called Rex", "", "The user lives in Taipei"])
        await current.reply("and now", conversation_id="a")
        sent = "".join(message.content for message in llm.calls[-1])
        await current.close()
        return before, current.memories("a"), current.memories("b"), sent

    before, after, elsewhere, sent = run(scenario())
    assert before == ["The user has a cat called Bun"]
    assert after == ["The user has a dog called Rex", "The user lives in Taipei"]
    assert elsewhere == []
    # What the user said stays said; what she noted of it is taken back.
    assert "The user has a cat called Bun" not in sent


def test_she_has_all_she_remembers_of_a_conversation_not_only_what_the_words_match(tmp_path):
    """The default retriever takes memories that share words with the newest
    message. "hello" shares none with where the user lives, and a user who
    then asks "guess where I am" is met by someone who has forgotten."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, memories_recalled=3)
        current.rewrite_memories(
            "a", [f"The user said thing number {number}" for number in range(5)]
        )
        current.rewrite_memories("b", ["The user lives in Taipei"])
        await current.reply("hello", conversation_id="b")
        in_b = system_context(llm.calls[-1])
        await current.reply("number 4 please", conversation_id="a")
        in_a = system_context(llm.calls[-1])
        await current.close()
        return in_b, in_a

    in_b, in_a = run(scenario())
    assert "The user lives in Taipei" in in_b
    assert "thing number" not in in_b
    # No more than the host allows, and what the message is about comes first.
    assert in_a.count("The user said thing number") == 3
    assert "The user said thing number 4" in in_a


def test_an_edit_forgets_only_what_the_user_saw_and_removed(tmp_path):
    """A memory page shows A; while it is open a background job adds B; the
    user saves "A and C". B was never shown, so it is not what the user
    removed."""

    async def scenario():
        current = companion(tmp_path)
        current.rewrite_memories("a", ["A"])
        shown = current.memories("a")
        current.rewrite_memories("a", ["A", "B"])  # arrived after the page was shown
        current.rewrite_memories("a", ["A", "C"], edited_from=shown)
        await current.close()
        return current.memories("a")

    assert run(scenario()) == ["A", "B", "C"]


def test_her_memory_can_be_read_and_edited_while_she_makes_a_remark_of_her_own(tmp_path):
    """A turn that is kept out of memory takes the memory manager away from the
    runtime for its duration. The memory page of a host is open meanwhile."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        current.rewrite_memories("a", ["The user lives in Taipei"])
        remark = asyncio.ensure_future(
            current.reply("(a remark of her own)", conversation_id="a", skip_memory=True)
        )
        await llm.started.wait()
        shown = current.memories("a")
        current.rewrite_memories("a", ["The user lives in Taichung"])
        llm.gate.set()
        await remark
        await current.close()
        return shown, current.memories("a")

    assert run(scenario()) == (["The user lives in Taipei"], ["The user lives in Taichung"])


def test_an_edited_memory_survives_a_restart(tmp_path):
    async def scenario():
        first = companion(tmp_path)
        first.rewrite_memories("a", ["The user lives in Taipei"])
        await first.close()
        return companion(tmp_path).memories("a")

    assert run(scenario()) == ["The user lives in Taipei"]


def test_a_late_memory_lands_in_the_conversation_it_came_from(tmp_path):
    """The coordinator writes memory into whatever scope is current when it
    commits. If the user has moved to another conversation by then, something
    private crosses into it."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(cat_fact_when_quoted, gate=gate)
        current = companion(tmp_path, {"memory": worker}, memory_every=1)
        await current.reply("I have a cat called Bun", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("hello", conversation_id="b")
        gate.set()
        await current.settle()
        found = current.memories("a"), current.memories("b")
        await current.close()
        return found

    assert run(scenario()) == (["The user has a cat called Bun"], [])


def test_a_fact_the_user_stated_is_kept_however_late_it_is_extracted(tmp_path):
    """Memory jobs queue behind everything else. Dropping a late one loses the
    facts of those turns for good: no later job reads the same lines."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(cat_fact_when_quoted, gate=gate)
        current = companion(tmp_path, {"memory": worker}, memory_every=4, max_turns_late=1)
        for text in ("one", "two", "three"):
            await current.reply(text, conversation_id="a")
        await current.reply("I have a cat called Bun", conversation_id="a")
        await until(lambda: worker.calls)
        for text in ("five", "six", "seven"):
            await current.reply(text, conversation_id="a")
        gate.set()
        await current.settle()
        found = current.memories("a")
        await current.close()
        return found

    assert run(scenario()) == ["The user has a cat called Bun"]


def goal_citing_the_event(messages):
    event_id = re.search(r'"event": \{.*?"id": "([^"]+)"', messages[1].content).group(1)
    return {
        "goals": [
            {
                "objective": "Finish the drawing and show it",
                "horizon": "short_term",
                "urgency": 0.8,
                "conflict_key": None,
                "motivation_signals": [
                    {
                        "kind": "explicit_request",
                        "strength": 0.9,
                        "source_type": "event",
                        "source_id": event_id,
                        "rationale": "The user asked to see it",
                    }
                ],
                "confidence": 0.9,
            }
        ],
        "confidence": 0.9,
        "evidence": [],
    }


def test_goals_and_thoughts_are_kept_and_reach_the_next_reply(tmp_path):
    async def scenario():
        workers = {
            "goal": Worker(goal_citing_the_event),
            "reflection": Worker(
                {
                    "insight": "They are looking forward to the drawing",
                    "belief_candidate": None,
                    "confidence": 0.9,
                    "evidence": ["show me when it is done"],
                }
            ),
        }
        current = companion(tmp_path, workers, goal_every=1, reflection_every=1)
        await current.reply("show me when it is done", conversation_id="a")
        await current.settle()
        await current.close()

        llm = Foreground()
        reopened = companion(tmp_path, llm=llm)
        await reopened.reply("hello again", conversation_id="a")
        await reopened.close()
        return reopened.snapshot(), system_context(llm.calls[-1])

    snapshot, context = run(scenario())
    assert list(snapshot.goals) == ["Finish the drawing and show it"]
    assert list(snapshot.thoughts) == ["They are looking forward to the drawing"]
    assert "Finish the drawing and show it" in context
    assert "They are looking forward to the drawing" in context


# --- interruption --------------------------------------------------------------


def test_interrupted_during_playback_she_remembers_only_what_was_heard(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        await current.reply("tell me", conversation_id="a")
        current.interrupt("Hello.")
        history = [message.content for message in current.runtime.history]
        await current.close()
        return history

    assert run(scenario()) == ["tell me", "Hello. [Interrupted by user]"]


def test_interrupted_while_generating_the_turn_ends_and_both_sides_are_kept(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        turn = asyncio.ensure_future(current.reply("tell me", conversation_id="a"))
        await llm.started.wait()
        current.interrupt("Hello.")
        with pytest.raises(TurnInterrupted):
            await turn
        history = [message.content for message in current.runtime.history]
        llm.gate.set()
        await current.reply("sorry, go on", conversation_id="a")
        await current.close()
        return history, current.busy

    history, busy = run(scenario())
    assert history == ["tell me", "Hello. [Interrupted by user]"]
    assert busy is False


def test_background_work_goes_on_after_an_interrupted_turn(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        worker = Worker(WARM)
        current = companion(tmp_path, {"emotion": worker}, llm=llm, emotion_every=1)
        turn = asyncio.ensure_future(current.reply("tell me", conversation_id="a"))
        await llm.started.wait()
        current.interrupt("Hello.")
        with pytest.raises(TurnInterrupted):
            await turn
        llm.gate.set()
        await current.reply("thank you for last night", conversation_id="a")
        await asyncio.wait_for(current.settle(), 2)
        snapshot = current.snapshot()
        await current.close()
        return snapshot.emotion

    assert run(scenario()) == "happy"


def test_an_interruption_reaches_the_turn_it_was_meant_for(tmp_path):
    """Two conversations at once: one reply is being generated, the other one
    waits for it. Interrupting the one that waits must not cut the other."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        first = asyncio.ensure_future(current.reply("first", conversation_id="a"))
        await llm.started.wait()
        second = asyncio.ensure_future(current.reply("second", conversation_id="b"))
        for _ in range(20):
            await asyncio.sleep(0)
        current.interrupt("", conversation_id="b")
        llm.gate.set()
        finished = await first
        with pytest.raises(TurnInterrupted):
            await second
        await current.reply("again", conversation_id="a")
        await current.close()
        return finished.text, [message.content for message in llm.calls[-1]]

    text, sent = run(scenario())
    assert text == "Hello. How are you?"
    assert "Hello. How are you?" in sent
    assert "second" not in sent


def test_an_interruption_without_a_name_is_for_the_reply_that_can_be_heard(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        talking = asyncio.ensure_future(current.reply("first", conversation_id="a"))
        await llm.started.wait()
        waiting = asyncio.ensure_future(current.reply("second", conversation_id="b"))
        for _ in range(20):
            await asyncio.sleep(0)
        current.interrupt("Hello.")
        with pytest.raises(TurnInterrupted):
            await talking
        llm.gate.set()
        answered = await waiting
        await current.close()
        return answered.text

    assert run(scenario()) == "Hello. How are you?"


def test_two_turns_of_one_conversation_are_told_apart_by_the_name_the_host_gave_them(tmp_path):
    """The same conversation open in two windows: one reply is being
    generated, the next one waits. The host interrupts the first."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        talking = asyncio.ensure_future(
            current.reply("first", conversation_id="a", turn_id="window 1")
        )
        await llm.started.wait()
        waiting = asyncio.ensure_future(
            current.reply("second", conversation_id="a", turn_id="window 2")
        )
        for _ in range(20):
            await asyncio.sleep(0)
        current.interrupt("Hello.", turn_id="window 1")
        with pytest.raises(TurnInterrupted):
            await talking
        llm.gate.set()
        await waiting
        history = [message.content for message in current.runtime.history]
        await current.close()
        return history

    assert run(scenario()) == [
        "first",
        "Hello. [Interrupted by user]",
        "second",
        "Hello. How are you?",
    ]


def test_interrupting_a_reply_that_was_played_does_not_reach_the_one_being_generated(tmp_path):
    """Two conversations: the reply of one is being played, the reply of the
    other is being generated. The listener of the first interrupts."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="window 1")
        llm.gate.clear()
        llm.started.clear()
        talking = asyncio.ensure_future(
            current.reply("second", conversation_id="b", turn_id="window 2")
        )
        await llm.started.wait()
        current.interrupt("Hello.", conversation_id="a", turn_id="window 1")
        llm.gate.set()
        answered = await talking
        history = [message.content for message in current.runtime.history]
        await current.close()
        return answered.text, history

    assert run(scenario()) == ("Hello. How are you?", ["second", "Hello. How are you?"])


def test_interrupting_a_turn_that_has_not_begun_leaves_the_last_reply_alone(tmp_path):
    """Without a conversation named, the interruption is for the turn asked
    for last, not for a reply that was played to the end."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("tell me", conversation_id="a")
        async with current._turn_lock:  # a commit of background work holds it
            turn = asyncio.ensure_future(current.reply("and then", conversation_id="a"))
            for _ in range(20):
                await asyncio.sleep(0)
            current.interrupt("")
        with pytest.raises(TurnInterrupted):
            await turn
        history = [message.content for message in current.runtime.history]
        await current.close()
        return history, len(llm.calls)

    assert run(scenario()) == (["tell me", "Hello. How are you?"], 1)


def test_a_reply_being_played_in_another_conversation_can_still_be_interrupted(tmp_path):
    """A's reply is being played while B is given a short reply. A's listener
    interrupts: the conversation at hand is B's by then, but A's reply is the
    one that was cut short."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="window 1")
        await current.reply("second", conversation_id="b", turn_id="window 2")
        current.interrupt("Hello.", conversation_id="a", turn_id="window 1")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "Hello. [Interrupted by user]" in sent
    assert "Hello. How are you?" not in sent


def test_interrupting_a_remark_of_her_own_does_not_touch_the_reply_before_it(tmp_path):
    """A remark kept out of memory is not in the conversation. An interruption
    of it must not fall on the reply that was played to the end before."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("(remark)", conversation_id="a", turn_id="w2", skip_memory=True)
        await current.reply("second", conversation_id="b", turn_id="w3")
        current.interrupt("Hel", conversation_id="a", turn_id="w2")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "Hello. How are you?" in sent
    assert "[Interrupted by user]" not in "".join(sent)


def test_a_reply_being_played_elsewhere_is_cut_while_another_is_being_generated(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        llm.gate.clear()
        llm.started.clear()
        talking = asyncio.ensure_future(current.reply("second", conversation_id="b", turn_id="w2"))
        await llm.started.wait()
        current.interrupt("Hello.", conversation_id="a", turn_id="w1")
        llm.gate.set()
        answered = await talking
        await current.reply("go on", conversation_id="a")
        await current.close()
        return answered.text, [message.content for message in llm.calls[-1]]

    text, sent = run(scenario())
    assert text == "Hello. How are you?"
    assert "Hello. [Interrupted by user]" in sent


def test_a_reply_played_in_the_same_conversation_is_cut_while_the_next_is_generated(tmp_path):
    """Two windows on one conversation: the first reply is still being played
    when the second window's reply begins. Interrupting the first must cut
    that reply, not be dropped because the model is busy."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        llm.gate.clear()
        llm.started.clear()
        llm.parts = ("Sure. ", "Let us begin.")
        talking = asyncio.ensure_future(current.reply("second", conversation_id="a", turn_id="w2"))
        await llm.started.wait()
        current.interrupt("Hel", conversation_id="a", turn_id="w1")
        llm.gate.set()
        answered = await talking
        llm.parts = ("Right. ", "Go on then.")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return answered.text, [message.content for message in llm.calls[-1]]

    text, sent = run(scenario())
    assert text == "Sure. Let us begin."
    assert "Hello. How are you?" not in sent
    assert sent.count("Sure. Let us begin.") == 1
    assert "Hel [Interrupted by user]" in sent
    assert sent.index("Hel [Interrupted by user]") < sent.index("second")


def test_a_memory_with_a_line_break_can_still_be_taken_back(tmp_path):
    """A host's memory page is one line per memory; a summary with a line
    break came back as two lines and could never be removed."""

    async def scenario():
        current = companion(tmp_path)
        current.rewrite_memories("a", ["The user has a cat\ncalled Bun"])
        shown = current.memories("a")
        current.rewrite_memories("a", [], edited_from=shown)
        await current.close()
        return shown, current.memories("a")

    assert run(scenario()) == (["The user has a cat called Bun"], [])


def test_a_host_that_cancels_its_own_task_can_report_what_was_heard_afterwards(tmp_path):
    """Hosts commonly interrupt by cancelling the task that awaits the reply
    and only then learn from the frontend how much was heard."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        turn = asyncio.ensure_future(current.reply("tell me", conversation_id="a"))
        await llm.started.wait()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        current.interrupt("Hello.")
        history = [message.content for message in current.runtime.history]
        await current.close()
        return history

    assert run(scenario()) == ["tell me", "Hello. [Interrupted by user]"]


# --- lifecycle -----------------------------------------------------------------


def test_without_a_storage_directory_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        current = CharacterCompanion(
            character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
            llm=Foreground(),
        )
        await current.reply("hello", conversation_id="a")
        await current.close()

    run(scenario())
    assert list(tmp_path.iterdir()) == []


def test_flushing_saves_state_without_stopping_the_companion(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        await current.reply("hello", conversation_id="a")
        (tmp_path / "engine" / "state.json").unlink()
        current.flush()
        saved = (tmp_path / "engine" / "state.json").is_file()
        await current.reply("still there", conversation_id="a")
        await current.close()
        return saved, current.snapshot().trust

    saved, trust = run(scenario())
    assert saved is True
    assert trust == pytest.approx(50.6)


def test_a_closed_companion_refuses_further_turns(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        await current.close()
        with pytest.raises(RuntimeError):
            await current.reply("hello", conversation_id="a")

    run(scenario())


def test_retiring_lets_her_finish_her_sentence_and_nothing_more(tmp_path):
    """A host replaces the companion when its settings are saved, which can be
    while she is talking. Cutting her off there would end the host's turn with
    a cancellation nobody asked for."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        worker = Worker(WARM)
        current = companion(tmp_path, {"emotion": worker}, llm=llm, emotion_every=1)
        talking = asyncio.ensure_future(current.reply("hello", conversation_id="a"))
        await llm.started.wait()
        waiting = asyncio.ensure_future(current.reply("and me", conversation_id="b"))
        for _ in range(20):
            await asyncio.sleep(0)
        current.retire()
        llm.gate.set()
        finished = await talking
        # Its own error, so that a host can ask the successor instead.
        with pytest.raises(CompanionClosed):
            await waiting
        for _ in range(50):
            await asyncio.sleep(0)
        await current.close()
        return finished.text, len(llm.calls), worker.calls

    assert run(scenario()) == ("Hello. How are you?", 1, 0)


def test_a_retired_companion_has_saved_and_writes_no_more(tmp_path):
    """A host replaces a companion from synchronous code, when its settings
    change. The successor reads the stored state at once, and nothing the old
    one still had under way may overwrite it."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(WARM, gate=gate)
        retired = companion(tmp_path, {"emotion": worker}, emotion_every=1)
        await retired.reply("thank you for last night", conversation_id="a")
        await until(lambda: worker.calls)
        retired.retire()

        successor = companion(tmp_path)
        await successor.reply("hello", conversation_id="a")
        gate.set()
        for _ in range(50):
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await retired.reply("still there?", conversation_id="a")
        await retired.close()
        await successor.close()
        return companion(tmp_path).snapshot()

    snapshot = run(scenario())
    assert snapshot.trust == pytest.approx(50.6)
    assert snapshot.emotion == "neutral"


def test_a_companion_belongs_to_the_event_loop_of_its_first_turn(tmp_path):
    current = companion(tmp_path)
    assert current.usable_in_running_loop() is True

    async def first():
        await current.reply("hi", conversation_id="a")
        return current.usable_in_running_loop()

    async def later():
        return current.usable_in_running_loop()

    assert run(first()) is True
    assert run(later()) is False


def test_a_long_life_does_not_fill_the_memory(tmp_path):
    """Measured over 800 turns before this: 28 KiB kept per turn, in the
    records of finished background work, each with a copy of the conversation,
    in the ledger of events, in the coordinator's decisions and in timers."""

    async def scenario():
        workers = {
            "emotion": Worker(NEUTRAL),
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}),
        }
        current = companion(
            tmp_path, workers, emotion_every=1, memory_every=1, records_kept=8
        )
        for number in range(30):
            await current.reply(f"line {number}", conversation_id="a")
            await current.settle()
        loop = asyncio.get_running_loop()
        kept = {
            "tasks": current._tasks.tracked_tasks,
            "handles": len(current._background.handles()),
            "decisions": current._commits.remembered_decisions,
            "ledger": len(current.runtime.memory_manager.ledger.list_for_character("mei:a")),
            "timers": sum(1 for handle in loop._scheduled if not handle.cancelled()),
        }
        await current.close()
        return kept

    kept = run(scenario())
    assert kept["tasks"] <= 8
    assert kept["handles"] <= 8
    assert kept["decisions"] <= 8
    assert kept["ledger"] <= 8
    assert kept["timers"] <= 2


def test_the_event_loop_can_shut_down_without_close(tmp_path):
    import threading

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(NEUTRAL, gate=gate)
        current = companion(tmp_path, {"emotion": worker}, emotion_every=1)
        await current.reply("hello", conversation_id="a")
        await until(lambda: worker.calls)

    thread = threading.Thread(target=lambda: asyncio.run(scenario()), daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()


def test_a_memory_the_background_wrote_with_a_line_break_is_shown_and_taken_back_as_one_line(tmp_path):
    """Only the host's own edits went through the one-line form; a memory the
    background extraction wrote with a line break came back as two lines and
    the page could not take it back."""
    from ai_character_engine.memory.models import MemoryRecord

    async def scenario():
        current = companion(tmp_path)
        current._memory_store.add(
            MemoryRecord(
                character_id=current._scope("a"),
                summary="The user has a cat\ncalled Bun",
                importance=0.8,
            )
        )
        shown = current.memories("a")
        current.rewrite_memories("a", [], edited_from=shown)
        left = current.memories("a")
        await current.close()
        return shown, left

    assert run(scenario()) == (["The user has a cat called Bun"], [])


def test_interrupting_an_older_turn_of_the_conversation_at_hand_leaves_the_newest_reply(tmp_path):
    """Two replies of one conversation, both finished. A late interruption
    of the first cannot mean the second: the first was heard whole before
    the second began."""

    async def scenario():
        llm = Scripted("Hello. How are you?", "Fine. Tell me more.", "Go on then.")
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("second", conversation_id="a", turn_id="w2")
        current.interrupt("Hel", conversation_id="a", turn_id="w1")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert sent.count("Hello. How are you?") == 1
    assert sent.count("Fine. Tell me more.") == 1
    assert not any("[Interrupted by user]" in content for content in sent)


def test_interrupting_a_kept_conversation_touches_only_that_turns_reply(tmp_path):
    """The newest message of a kept conversation may be a later turn's
    record, cut short while she spoke. A late interruption of the reply
    before it cuts that reply, wherever it is, and not the record."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        llm.gate.clear()
        llm.started.clear()
        cut = asyncio.ensure_future(current.reply("second", conversation_id="a", turn_id="w2"))
        await llm.started.wait()
        current.interrupt("Hel", conversation_id="a", turn_id="w2")
        with pytest.raises(TurnInterrupted):
            await cut
        llm.gate.set()
        await current.reply("elsewhere", conversation_id="b")
        current.interrupt("Hello.", conversation_id="a", turn_id="w1")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert sent.index("Hello. [Interrupted by user]") < sent.index("second")
    assert sent.index("Hel [Interrupted by user]") > sent.index("second")
    assert "Hello. How are you?" not in sent


def test_a_hosts_own_model_call_waits_for_her_reply_like_her_background_work(tmp_path):
    """A host has model calls of its own beside hers, on the same local model:
    a memory of its own to tidy, a translation. Made through the companion,
    such a call gives way to the reply once and takes its turn with the
    background workers instead of slowing the reply down."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        started: list[str] = []

        async def tidy():
            started.append("tidy" if not llm.started.is_set() or llm.gate.is_set() else "tidy-during-reply")
            return "tidied"

        talking = asyncio.ensure_future(current.reply("hello", conversation_id="a"))
        await llm.started.wait()
        aside = asyncio.ensure_future(current.aside(tidy))
        for _ in range(20):
            await asyncio.sleep(0)
        during = list(started)
        llm.gate.set()
        await talking
        result = await aside
        await current.close()
        return during, result, started

    assert run(scenario()) == ([], "tidied", ["tidy"])


def test_a_hosts_own_model_call_is_refused_once_she_is_closed(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        await current.close()

        async def tidy():
            return "tidied"

        with pytest.raises(CompanionClosed):
            await current.aside(tidy)

    run(scenario())


def test_a_reply_the_host_did_not_use_is_taken_back_with_the_words_it_answered(tmp_path):
    """A host may throw a reply away, one that repeats what she just said, and
    ask the same question again with a hint. Kept, the exchange would stand
    twice in the conversation, once with a reply nobody heard."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("tell me more", conversation_id="a", turn_id="w2")
        current.take_back("a")
        await current.reply("tell me more, differently", conversation_id="a", turn_id="w3")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "tell me more" not in sent
    assert sent.count("Hello. How are you?") == 1
    assert sent.index("first") < sent.index("tell me more, differently")


def test_taking_back_touches_only_the_newest_exchange_and_only_once(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("second", conversation_id="a", turn_id="w2")
        current.take_back("a")
        current.take_back("a")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "first" in sent
    assert "second" not in sent
    assert sent.count("Hello. How are you?") == 1


def test_a_kept_conversation_can_have_its_newest_exchange_taken_back(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("second", conversation_id="a", turn_id="w2")
        await current.reply("elsewhere", conversation_id="b")
        current.take_back("a")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return [message.content for message in llm.calls[-1]]

    sent = run(scenario())
    assert "first" in sent
    assert "second" not in sent


# --- what she keeps in mind ----------------------------------------------------


def _goal(character_id, objective, urgency, *, updated_at=None, conversation="a"):
    """A short-term goal of the conversation ``conversation``: from 1.2.0 one
    is in mind only in the conversation it came from."""
    from ai_character_engine.goals.models import (
        GoalEvidenceRef,
        GoalHorizon,
        GoalRecord,
        MotivationKind,
        MotivationSignal,
    )

    record = GoalRecord(
        character_id,
        objective,
        GoalHorizon.SHORT_TERM,
        urgency,
        0.9,
        (
            MotivationSignal(
                kind=MotivationKind.EXPLICIT_REQUEST,
                strength=0.9,
                evidence=GoalEvidenceRef(source_type="event", source_id="e1", excerpt="please"),
                rationale="The user asked",
            ),
        ),
        metadata={"conversation_id": conversation},
    )
    if updated_at is not None:
        from dataclasses import replace

        record = replace(record, updated_at=updated_at)
    return record


def _goal_lines(messages):
    return [
        line
        for message in messages
        for line in message.content.splitlines()
        if line.startswith("- goal: ")
    ]


def test_only_the_few_goals_that_matter_most_are_in_her_mind(tmp_path):
    """Every active goal went into the prompt, bounded by tokens alone. A 9B
    model's goals are many and uneven; a few, the most pressing first, keep
    her focused and the prompt short."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        store = current.runtime.goal_manager.store
        for number, urgency in enumerate((0.2, 0.9, 0.5, 0.7, 0.3)):
            store.add_goal(_goal("mei", f"Goal number {number}", urgency))
        await current.reply("hello", conversation_id="a")
        shown = current.snapshot().goals
        await current.close()
        return _goal_lines(llm.calls[-1]), shown

    lines, shown = run(scenario())
    assert lines == ["- goal: Goal number 1", "- goal: Goal number 3", "- goal: Goal number 2"]
    assert list(shown) == ["Goal number 1", "Goal number 3", "Goal number 2"]


def test_how_many_goals_she_keeps_in_mind_is_a_setting(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, goals_shown=1)
        store = current.runtime.goal_manager.store
        for number, urgency in enumerate((0.2, 0.9)):
            store.add_goal(_goal("mei", f"Goal number {number}", urgency))
        await current.reply("hello", conversation_id="a")
        await current.close()
        return _goal_lines(llm.calls[-1])

    assert run(scenario()) == ["- goal: Goal number 1"]


def test_a_goal_untouched_for_too_long_leaves_the_prompt_not_only_the_snapshot(tmp_path):
    from datetime import UTC, datetime, timedelta

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        store = current.runtime.goal_manager.store
        store.add_goal(_goal("mei", "Fresh goal", 0.5))
        store.add_goal(
            _goal("mei", "Stale goal", 0.9, updated_at=datetime.now(UTC) - timedelta(days=30))
        )
        await current.reply("hello", conversation_id="a")
        await current.close()
        return _goal_lines(llm.calls[-1])

    assert run(scenario()) == ["- goal: Fresh goal"]


def test_how_many_thoughts_she_keeps_in_mind_is_a_setting(tmp_path):
    from ai_character_engine.long_term_cognition.models import ReflectionRecord

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, thoughts_shown=1)
        store = current.runtime.long_term_cognition.store
        for number in range(3):
            store.add_reflection(
                ReflectionRecord(
                    "mei", f"Thought number {number}", 0.9, metadata={"conversation_id": "a"}
                )
            )
        await current.reply("hello", conversation_id="a")
        await current.close()
        return [
            line
            for message in llm.calls[-1]
            for line in message.content.splitlines()
            if line.startswith("- thought: ")
        ]

    assert run(scenario()) == ["- thought: Thought number 2"]


# --- taking back, host calls and the goals in mind: edge cases -------------------


def test_taking_back_after_a_turn_that_failed_leaves_the_exchange_that_was_heard(tmp_path):
    """A turn that ends in an error records nothing; the newest reply is still
    the one before it, which the user heard whole. A host that asks again
    after the error must not lose that exchange."""
    from ai_character_engine.host.bridge import HostBridgeError

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        llm.parts = ("",)
        with pytest.raises(HostBridgeError):
            await current.reply("second", conversation_id="a", turn_id="w2")
        taken = current.take_back("a")
        llm.parts = ("Hello. ", "How are you?")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return taken, [message.content for message in llm.calls[-1]]

    taken, sent = run(scenario())
    assert taken is False
    assert "first" in sent
    assert sent.count("Hello. How are you?") == 1


def test_taking_back_after_a_remark_kept_out_of_memory_takes_nothing(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("(a remark of her own)", conversation_id="a", skip_memory=True)
        taken = current.take_back("a")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return taken, [message.content for message in llm.calls[-1]]

    taken, sent = run(scenario())
    assert taken is False
    assert "first" in sent


def test_taking_back_a_remark_she_made_on_her_own_leaves_the_exchange_before_it(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("(say something)", conversation_id="a", proactive=True, turn_id="w2")
        taken = current.take_back("a")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return taken, [message.content for message in llm.calls[-1]]

    taken, sent = run(scenario())
    assert taken is True
    assert "first" in sent
    assert sent.count("Hello. How are you?") == 1


def test_taking_back_leaves_no_note_behind_for_the_removed_messages(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        await current.reply("second", conversation_id="a", turn_id="w2")
        current.take_back("a")
        active_ok = all(
            any(anchor is message for message in current.runtime.history)
            for anchor, _ in current.runtime.context_notes
        )
        await current.reply("third", conversation_id="a", turn_id="w3")
        await current.reply("elsewhere", conversation_id="b")
        current.take_back("a")
        history, notes = current._kept["a"]
        kept_ok = all(any(anchor is message for message in history) for anchor, _ in notes)
        await current.close()
        return active_ok, kept_ok

    assert run(scenario()) == (True, True)


def test_taking_back_waits_for_no_turn_and_refuses_while_one_runs(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        llm.gate.clear()
        llm.started.clear()
        talking = asyncio.ensure_future(current.reply("second", conversation_id="a", turn_id="w2"))
        await llm.started.wait()
        taken = current.take_back("a")
        llm.gate.set()
        await talking
        await current.close()
        return taken

    assert run(scenario()) is False


def test_a_reply_the_host_replaced_can_still_be_taken_back_or_cut(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="w1")
        current.replace_reply("Hello, how are you?")
        taken = current.take_back("a")
        await current.reply("go on", conversation_id="a")
        await current.close()
        return taken, [message.content for message in llm.calls[-1]]

    taken, sent = run(scenario())
    assert taken is True
    assert "first" not in sent


def test_a_hosts_own_call_made_during_the_reply_still_goes_after_her_workers(tmp_path):
    async def scenario():
        Worker.reset()
        workers = {
            "emotion": Worker(WARM, name="emotion"),
            "memory": Worker({"items": [], "confidence": 0.5, "evidence": []}, name="memory"),
        }
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, workers, llm=llm, emotion_every=1, memory_every=1)

        async def tidy():
            Worker.order.append("aside")
            return "tidied"

        talking = asyncio.ensure_future(current.reply("hello", conversation_id="a"))
        await llm.started.wait()
        aside = asyncio.ensure_future(current.aside(tidy))
        for _ in range(5):
            await asyncio.sleep(0)
        llm.gate.set()
        await talking
        await aside
        await current.settle()
        await current.close()
        return list(Worker.order)

    assert run(scenario()) == ["emotion", "memory", "aside"]


def test_a_hosts_own_call_still_waiting_when_she_retires_is_refused(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        made = []

        async def tidy():
            made.append(1)
            return "tidied"

        talking = asyncio.ensure_future(current.reply("hello", conversation_id="a"))
        await llm.started.wait()
        aside = asyncio.ensure_future(current.aside(tidy))
        for _ in range(5):
            await asyncio.sleep(0)
        current.retire()
        llm.gate.set()
        await talking
        with pytest.raises(CompanionClosed):
            await aside
        await current.close()
        return made

    assert run(scenario()) == []


def test_a_goal_pushed_out_by_a_more_pressing_one_is_not_said_to_be_given_up(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, goals_shown=2)
        store = current.runtime.goal_manager.store
        store.add_goal(_goal("mei", "Goal A", 0.5))
        store.add_goal(_goal("mei", "Goal B", 0.4))
        await current.reply("hello", conversation_id="a")
        store.add_goal(_goal("mei", "Goal C", 0.9))
        await current.reply("again", conversation_id="a")
        await current.close()
        return [
            line
            for message in llm.calls[-1]
            for line in message.content.splitlines()
            if "goal" in line
        ]

    lines = run(scenario())
    assert "- goal: Goal C" in lines
    assert not any(line.startswith("- no longer a goal:") for line in lines)

# --- what she keeps in mind, and taking back after unusual turns ------------------


def _roles(history):
    return [(m.role, m.content[:30]) for m in history]


def test_taking_back_after_a_turn_interrupted_before_she_began_takes_nothing(tmp_path):
    """Turn a2 is interrupted before she began (nothing heard). It is the
    conversation's last turn and it failed; take_back must refuse."""

    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        llm.gate.set()
        current = companion(tmp_path, llm=llm)
        await current.reply("first", conversation_id="a", turn_id="a1")
        llm.gate.clear()
        llm.started.clear()
        b = asyncio.ensure_future(current.reply("other", conversation_id="b", turn_id="b1"))
        await llm.started.wait()
        a2 = asyncio.ensure_future(current.reply("second", conversation_id="a", turn_id="a2"))
        for _ in range(5):
            await asyncio.sleep(0)
        current.interrupt("", conversation_id="a", turn_id="a2")
        llm.gate.set()
        await b
        with pytest.raises(TurnInterrupted):
            await a2
        before = _roles(current.runtime.history)
        taken = current.take_back("a")
        after = _roles(current.runtime.history)
        await current.close()
        return taken, before, after

    taken, before, after = run(scenario())
    assert taken is False, (before, after)


def test_taking_back_a_reply_that_used_a_tool_leaves_the_exchange_before_it(tmp_path):
    class ToolUsing(Foreground):
        async def stream_generate(self, messages, *, tools=None):
            self.calls.append(list(messages))
            if messages[-1].role == "tool":
                yield LLMStreamChunk(text="It is 12:34.")
                yield LLMStreamChunk(final=True, response=LLMResponse(text="It is 12:34.", model="f"))
                return
            if "time" not in messages[-1].content and "time" not in str(messages[-2:]):
                yield LLMStreamChunk(text="Hi.")
                yield LLMStreamChunk(final=True, response=LLMResponse(text="Hi.", model="f"))
                return
            yield LLMStreamChunk(text="Let me check. ")
            yield LLMStreamChunk(
                final=True,
                response=LLMResponse(text="Let me check. ", tool_calls=(ToolCall("one", "clock", {}),), model="f"),
            )

    async def scenario():
        current = companion(tmp_path, llm=ToolUsing())
        current.tools.register(
            ToolDefinition("clock", "Read the clock", {"type": "object", "properties": {}}),
            lambda: "12:34",
        )
        await current.reply("hello", conversation_id="a")
        h0 = _roles(current.runtime.history)
        await current.reply("what time is it", conversation_id="a")
        h1 = _roles(current.runtime.history)
        taken = current.take_back("a")
        h2 = _roles(current.runtime.history)
        await current.close()
        return h0, h1, taken, h2

    h0, h1, taken, h2 = run(scenario())
    assert taken and h2 == h0, (h1, h2)


def test_taking_back_a_reply_about_a_picture_leaves_the_exchange_before_it(tmp_path):
    from ai_character_engine.vision import VisionPipeline
    from ai_character_engine.vision.models import ImageInput, VisionAnalysis, VisionFrame
    from ai_character_engine.vision.providers import CallableVisionProvider
    from tests.test_vision import PNG

    async def look(image, prompt):
        return VisionAnalysis(text="A red square.", provider="fake")

    async def scenario():
        llm = Foreground()
        current = CharacterCompanion(
            character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
            llm=llm,
            background_llm={},
            vision=VisionPipeline(provider=CallableVisionProvider(look)),
        )
        await current.reply("hi", conversation_id="a")
        h0 = _roles(current.runtime.history)
        frame = VisionFrame(image=ImageInput.from_bytes(PNG, mime_type="image/png"))
        await current.reply("what is this", conversation_id="a", frames=(frame,))
        taken = current.take_back("a")
        h2 = _roles(current.runtime.history)
        notes = current.runtime.context_notes
        ok = all(any(a is m for m in current.runtime.history) for a, _ in notes)
        await current.close()
        return h0, taken, h2, ok

    h0, taken, h2, ok = run(scenario())
    assert taken and h2 == h0 and ok


def test_the_goals_in_the_conversation_are_the_goals_in_mind(tmp_path):
    """Notes stay in the conversation. A goal pushed out of mind by a more
    pressing one was still standing there as a goal: the prompt carried every
    goal ever announced, not goals_shown of them."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, goals_shown=1)
        store = current.runtime.goal_manager.store
        store.add_goal(_goal("mei", "Goal A", 0.5))
        await current.reply("hello", conversation_id="a")
        store.add_goal(_goal("mei", "Goal B", 0.9))
        await current.reply("again", conversation_id="a")
        store.add_goal(_goal("mei", "Goal C", 0.95))
        await current.reply("more", conversation_id="a")
        shown = current.snapshot().goals
        await current.close()
        return shown, [
            line
            for message in llm.calls[-1]
            for line in message.content.splitlines()
            if line.startswith(("- goal: ", "- no longer a goal: ", "- set aside for now: "))
        ]

    shown, lines = run(scenario())
    in_mind = {}
    for line in lines:
        if line.startswith("- goal: "):
            in_mind[line.removeprefix("- goal: ")] = True
        else:
            in_mind[line.split(": ", 1)[1]] = False
    assert [goal for goal, held in in_mind.items() if held] == list(shown) == ["Goal C"]
    assert not any(line.startswith("- no longer a goal:") for line in lines)


def test_a_hosts_own_call_does_not_wait_for_work_that_never_ends(tmp_path):
    async def scenario():
        current = companion(tmp_path, foreground_patience_seconds=0.2)
        stuck = asyncio.ensure_future(asyncio.Event().wait())
        current._pending.add(stuck)

        async def tidy():
            return "tidied"

        result = await asyncio.wait_for(current.aside(tidy), timeout=2)
        current._pending.discard(stuck)
        stuck.cancel()
        await current.close()
        return result

    assert run(scenario()) == "tidied"


def test_a_hosts_own_call_queued_behind_a_worker_is_refused_once_she_retires(tmp_path):
    async def scenario():
        Worker.reset()
        gate = asyncio.Event()
        workers = {"memory": Worker({"items": [], "confidence": 0.5, "evidence": []}, gate=gate, name="memory")}
        current = companion(tmp_path, workers, memory_every=1, foreground_patience_seconds=0.2)
        await current.reply("hello", conversation_id="a")
        for _ in range(50):
            await asyncio.sleep(0)
        made = []

        async def tidy():
            made.append(1)
            return "x"

        aside = asyncio.ensure_future(current.aside(tidy))
        await asyncio.sleep(0.4)  # past the patience: now queued in ModelAccess behind the worker
        current.retire()
        gate.set()
        try:
            result = await aside
        except CompanionClosed:
            result = "closed"
        await current.close()
        return result, made, list(Worker.order)

    result, made, order = run(scenario())
    assert result == "closed" and made == [], (result, made, order)


# --- what she said on her own ------------------------------------------------------


def test_a_remark_she_made_on_her_own_stays_in_the_conversation(tmp_path):
    """A host prompts her to speak up with a long instruction it does not want
    kept. Kept out of memory, the remark was forgotten too: she repeated her
    remarks, and when the user answered one she said it again word for word.
    remember_remark keeps what she said, after a short line saying why."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("I am building a time machine", conversation_id="a")
        await current.remember_remark("a", "Mind the load on Amadeus.")
        await current.reply("ok, checking", conversation_id="a")
        await current.close()
        return llm.calls[-1]

    sent = run(scenario())
    contents = [message.content for message in sent]
    at = contents.index("Mind the load on Amadeus.")
    assert sent[at].role == "assistant"
    assert sent[at - 1].role == "event"
    assert contents.index("ok, checking") > at
    assert contents.count("Mind the load on Amadeus.") == 1


def test_a_remark_goes_to_the_conversation_it_was_made_in(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hello", conversation_id="a")
        await current.reply("hello", conversation_id="b")
        await current.remember_remark("a", "Still there?")
        await current.reply("yes", conversation_id="a")
        in_a = [message.content for message in llm.calls[-1]]
        await current.reply("and here?", conversation_id="b")
        in_b = [message.content for message in llm.calls[-1]]
        await current.close()
        return in_a, in_b

    in_a, in_b = run(scenario())
    assert "Still there?" in in_a
    assert "Still there?" not in in_b


def test_a_remark_waits_for_the_reply_under_way(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        talking = asyncio.ensure_future(current.reply("hello", conversation_id="a"))
        await llm.started.wait()
        remark = asyncio.ensure_future(current.remember_remark("a", "By the way."))
        for _ in range(5):
            await asyncio.sleep(0)
        early = remark.done()
        llm.gate.set()
        await talking
        await remark
        history = [message.content for message in current.runtime.history]
        await current.close()
        return early, history

    early, history = run(scenario())
    assert early is False
    assert history[-1] == "By the way."
    assert history.index("Hello. How are you?") < history.index("By the way.")


def test_a_remark_can_be_taken_back_like_a_reply(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hello", conversation_id="a")
        before = [message.content for message in current.runtime.history]
        await current.remember_remark("a", "By the way.")
        taken = current.take_back("a")
        after = [message.content for message in current.runtime.history]
        await current.close()
        return taken, before, after

    taken, before, after = run(scenario())
    assert taken is True
    assert after == before


def test_an_empty_remark_is_not_kept(tmp_path):
    async def scenario():
        current = companion(tmp_path)
        await current.reply("hello", conversation_id="a")
        before = list(current.runtime.history)
        await current.remember_remark("a", "   ")
        after = list(current.runtime.history)
        await current.close()
        return before, after

    before, after = run(scenario())
    assert after == before


# --- speaking up on her own --------------------------------------------------------


def test_she_speaks_up_on_her_own_and_remembers_what_she_said(tmp_path):
    async def scenario():
        llm = Scripted("Hello. How are you?", "Mind the load on Amadeus.", "Good.")
        current = companion(tmp_path, llm=llm)
        await current.reply("I am building a time machine", conversation_id="a")
        result = await current.speak_up("a")
        spoken_prompt = llm.calls[-1]
        await current.reply("ok", conversation_id="a")
        await current.close()
        return result.text, spoken_prompt, llm.calls[-1]

    text, spoken_prompt, after = run(scenario())
    assert text == "Mind the load on Amadeus."
    # She was told to speak up, as the newest message, not as the user's words.
    assert spoken_prompt[-1].role == "event"
    assert "Speak up on your own" in spoken_prompt[-1].content
    # What stays is what she said, after a short event; not the instruction.
    contents = [message.content for message in after]
    assert not any("Speak up on your own" in content for content in contents)
    at = contents.index("Mind the load on Amadeus.")
    assert after[at].role == "assistant" and after[at - 1].role == "event"


def test_speaking_up_brings_what_she_wants_to_mind(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        current.runtime.goal_manager.store.add_goal(_goal("mei", "Ask how the drawing went", 0.8))
        await current.speak_up("a")
        await current.close()
        return "\n".join(message.content for message in llm.calls[-1])

    assert "- goal: Ask how the drawing went" in run(scenario())


def test_what_the_host_suggests_is_for_that_remark_only(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a", notes=["Topics the user likes: astronomy"])
        during = "\n".join(message.content for message in llm.calls[-1])
        await current.reply("hi", conversation_id="a")
        after = "\n".join(message.content for message in llm.calls[-1])
        await current.close()
        return during, after

    during, after = run(scenario())
    assert "Topics the user likes: astronomy" in during
    assert "Topics the user likes: astronomy" not in after


def test_a_host_that_keeps_the_remark_itself_can_say_so(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hello", conversation_id="a")
        before = [message.content for message in current.runtime.history]
        await current.speak_up("a", keep=False)
        after = [message.content for message in current.runtime.history]
        await current.close()
        return before, after

    before, after = run(scenario())
    assert after == before


def test_a_remark_cut_short_keeps_what_was_heard_and_never_the_instruction(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        speaking = asyncio.ensure_future(current.speak_up("a", turn_id="p1"))
        await llm.started.wait()
        current.interrupt("Hello.", conversation_id="a", turn_id="p1")
        with pytest.raises(TurnInterrupted):
            await speaking
        history = [(message.role, message.content) for message in current.runtime.history]
        await current.close()
        return history

    history = run(scenario())
    assert history[-1] == ("assistant", "Hello. [Interrupted by user]")
    assert history[-2][0] == "event"
    assert not any("Speak up on your own" in content for _, content in history)
    assert not any(role == "user" for role, _ in history)


def test_a_remark_the_host_cancelled_and_reported_afterwards_is_kept_as_hers(tmp_path):
    async def scenario():
        llm = Foreground(gate=asyncio.Event())
        current = companion(tmp_path, llm=llm)
        speaking = asyncio.ensure_future(current.speak_up("a"))
        await llm.started.wait()
        speaking.cancel()
        with pytest.raises(asyncio.CancelledError):
            await speaking
        current.interrupt("Hello.", conversation_id="a")
        history = [(message.role, message.content) for message in current.runtime.history]
        await current.close()
        return history

    history = run(scenario())
    assert history[-1] == ("assistant", "Hello. [Interrupted by user]")
    assert not any(role == "user" for role, _ in history)


def test_a_remark_interrupted_while_it_is_played_is_cut_to_what_was_heard(tmp_path):
    async def scenario():
        llm = Scripted("Hello. How are you?", "Mind the load. Or else.")
        current = companion(tmp_path, llm=llm)
        await current.reply("hello", conversation_id="a")
        await current.speak_up("a", turn_id="p1")
        current.interrupt("Mind the load.", conversation_id="a", turn_id="p1")
        history = [(message.role, message.content) for message in current.runtime.history]
        await current.close()
        return history

    history = run(scenario())
    assert history[-1] == ("assistant", "Mind the load. [Interrupted by user]")
    assert history[1] == ("assistant", "Hello. How are you?")


def test_a_host_can_ask_her_to_speak_up_in_its_own_words(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a", instruction="對方已經有一段時間沒講話了。請你自然地開口。")
        await current.close()
        return llm.calls[-1][-1].content

    event = run(scenario())
    assert "對方已經有一段時間沒講話了。請你自然地開口。" in event
    assert "Speak up on your own" not in event


class Scripted(Foreground):
    """Answers each call with the next of the given replies."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else "…"
        self.started.set()
        yield LLMStreamChunk(text=text)
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="scripted"))


def test_she_never_says_the_same_thing_on_her_own_twice(tmp_path):
    """A small model copied its previous remark word for word, even with the
    remark in the conversation and a rule against it. The engine checks what
    she is about to say against what she said, and asks again."""

    async def scenario():
        llm = Scripted(
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Mind the load on Amadeus, or your time machine breaks too!",
            "How is Bun doing tonight?",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        heard: list[str] = []
        result = await current.speak_up("a", on_text_delta=heard.append)
        kept = [message.content for message in current.runtime.history if message.role == "assistant"]
        await current.close()
        return result.text, "".join(heard), kept, len(llm.calls)

    text, heard, kept, calls = run(scenario())
    assert text == "How is Bun doing tonight?"
    assert heard == "How is Bun doing tonight?"
    assert kept == [
        "Mind the load on Amadeus, or your time machine breaks too.",
        "How is Bun doing tonight?",
    ]
    assert calls == 3


def test_she_stays_quiet_rather_than_repeat_herself(tmp_path):
    async def scenario():
        llm = Scripted(*(["Mind the load on Amadeus."] * 5))
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        heard: list[str] = []
        result = await current.speak_up("a", on_text_delta=heard.append)
        kept = [message.content for message in current.runtime.history if message.role == "assistant"]
        await current.close()
        return result.text, heard, kept

    text, heard, kept = run(scenario())
    assert text == ""
    assert heard == []
    assert kept == ["Mind the load on Amadeus."]


def test_a_remark_that_repeats_a_reply_she_gave_is_not_said_either(tmp_path):
    async def scenario():
        llm = Scripted("Hello. How are you?", "Hello. How are you?", "What are you reading?")
        current = companion(tmp_path, llm=llm)
        await current.reply("hi", conversation_id="a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "What are you reading?"


# --- not repeating herself in a reply -------------------------------------------------


def test_a_sentence_she_already_said_is_not_said_again_in_a_reply(tmp_path):
    """When the user answered her remark, a small model said the remark again
    word for word before answering. A sentence that repeats one of her latest
    lines is not passed on, and is not kept."""

    async def scenario():
        llm = Scripted(
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Mind the load on Amadeus, or your time machine breaks too. Good, go check it now.",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        heard: list[str] = []
        await current.reply("ok, checking", conversation_id="a", on_text_delta=heard.append)
        kept = current.runtime.history[-1].content
        await current.close()
        return "".join(heard), kept

    heard, kept = run(scenario())
    assert heard.strip() == "Good, go check it now."
    assert kept.strip() == "Good, go check it now."


def test_a_short_word_she_says_often_is_not_a_repetition(tmp_path):
    async def scenario():
        llm = Scripted("Hmm. I see what you mean.", "Hmm. Then try the other way round.")
        current = companion(tmp_path, llm=llm)
        await current.reply("it failed", conversation_id="a")
        heard: list[str] = []
        await current.reply("again", conversation_id="a", on_text_delta=heard.append)
        await current.close()
        return "".join(heard)

    assert run(scenario()) == "Hmm. Then try the other way round."


def test_a_reply_that_would_be_all_repetition_is_asked_again(tmp_path):
    async def scenario():
        llm = Scripted(
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Then go and look at the cooling first.",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        heard: list[str] = []
        result = await current.reply("ok", conversation_id="a", on_text_delta=heard.append)
        await current.close()
        return result.text, "".join(heard)

    text, heard = run(scenario())
    assert heard == "Then go and look at the cooling first."
    assert text == "Then go and look at the cooling first."


def test_a_remark_leaves_out_a_sentence_she_already_said(tmp_path):
    async def scenario():
        llm = Scripted(
            "The typhoon turned. Better look at the weather map than sulk over the failed run.",
            "Put the failure aside for now. Better look at the weather map than sulk over the failed run.",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        heard: list[str] = []
        result = await current.speak_up("a", on_text_delta=heard.append)
        kept = current.runtime.history[-1].content
        await current.close()
        return result.text, "".join(heard), kept

    text, heard, kept = run(scenario())
    assert text.strip() == "Put the failure aside for now."
    assert heard.strip() == "Put the failure aside for now."
    assert kept.strip() == "Put the failure aside for now."


def test_a_short_remark_of_her_own_is_still_a_remark(tmp_path):
    async def scenario():
        llm = Scripted("喂，還醒著嗎？")
        current = companion(tmp_path, llm=llm)
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "喂，還醒著嗎？"


def test_a_sentence_with_a_new_opening_on_an_old_one_is_still_a_repetition(tmp_path):
    async def scenario():
        llm = Scripted(
            "Stop dreaming and look at the latest typhoon map, that is what saves lives.",
            "Stop building that machine and look at the latest typhoon map, that is what saves lives.",
            "How is Bun taking the rain?",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "How is Bun taking the rain?"


def test_an_old_sentence_retold_with_small_changes_is_still_a_repetition(tmp_path):
    """Most of the words kept, in order, with a few swapped along the way: no
    long run is copied and the whole is not nearly identical, yet it is the
    same sentence again."""

    async def scenario():
        llm = Scripted(
            "Since you agree, put the failed run aside; I just saw Tokyo Banana went on sale"
            " here as custard puffs, and that sounds tastier than the formula we mixed."
            " Weren't you annoyed? Have a bite and swap the taste of failure for something"
            " sweet.",
            "Put the failures aside, I saw Tokyo Banana went on sale here too as custard"
            " puffs, tastier than the formula we mixed.",
            "How is Bun taking the rain?",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "How is Bun taking the rain?"


# --- what she said about herself ---------------------------------------------------

SELF_LINE = "- you said about yourself: "


def _listed_for_her(messages):
    return messages[1].content.split("Character lines to extract from:")[1].split("\n\n")[0]


def about_herself(*facts):
    """A self-memory worker that reports each (summary, quote) whose quote is
    among the lines it was given to read."""

    def answer(messages):
        listed = _listed_for_her(messages)
        items = [
            {
                "summary": summary,
                "kind": "preference",
                "importance": 0.6,
                "confidence": 0.9,
                "evidence": quote,
            }
            for summary, quote in facts
            if quote in listed
        ]
        return {"items": items, "confidence": 0.9, "evidence": []}

    return answer


TEA = ("Mei drinks jasmine tea when she works late", "I drink jasmine tea when I work late")
PIANO = ("Mei has played the piano since she was six", "I have played the piano since I was six")
HILLS = ("Mei walks in the hills every Sunday", "I walk in the hills every Sunday")


def _self_lines(messages):
    return [
        line
        for message in messages
        for line in message.content.splitlines()
        if line.startswith(SELF_LINE)
    ]


def test_she_remembers_what_she_said_about_herself(tmp_path):
    """She said one thing about herself in one conversation and the opposite
    in the next: nothing kept what she said, only what the user said."""

    async def scenario():
        llm = Scripted("Oh, I drink jasmine tea when I work late.", "Hello again.")
        worker = Worker(about_herself(TEA))
        current = companion(tmp_path, {"self_memory": worker}, llm=llm, self_memory_every=1)
        await current.reply("what do you drink at night", conversation_id="a")
        await current.settle()
        await current.reply("hi", conversation_id="b")
        await current.close()
        return current.self_memories(), _self_lines(llm.calls[-1]), current.memories("a")

    remembered, in_b, about_the_user = run(scenario())
    assert remembered == ["Mei drinks jasmine tea when she works late"]
    # Hers, not the conversation's: another conversation has it too.
    assert in_b == [f"{SELF_LINE}Mei drinks jasmine tea when she works late"]
    assert about_the_user == []


def test_only_her_lines_are_given_to_be_read_each_once(tmp_path):
    asked = []

    def nothing(messages):
        asked.append(_listed_for_her(messages))
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        llm = Scripted("I drink jasmine tea when I work late.", "Mind the cooling.", "Fine.")
        current = companion(
            tmp_path, {"self_memory": Worker(nothing)}, llm=llm, self_memory_every=1
        )
        await current.reply("I have a cat called Bun", conversation_id="a")
        await current.settle()
        # A remark of her own schedules no work; the next turn reads it.
        await current.speak_up("a")
        await current.reply("ok", conversation_id="a")
        await current.settle()
        await current.close()

    run(scenario())
    first, second = asked
    assert "I drink jasmine tea when I work late." in first
    assert "I have a cat called Bun" not in first
    assert "I drink jasmine tea" not in second
    assert "Mind the cooling." in second and "Fine." in second


def test_what_she_said_about_herself_needs_a_quote_of_her_own(tmp_path):
    invented = {
        "items": [
            {
                "summary": "Mei has a cat called Bun",
                "kind": "fact",
                "importance": 0.8,
                "confidence": 0.9,
                "evidence": "I have a cat called Bun",
            },
            {
                "summary": "Mei loves coffee",
                "kind": "preference",
                "importance": 0.8,
                "confidence": 0.9,
                "evidence": "I love coffee",
            },
            {
                "summary": "Mei likes the rain",
                "kind": "preference",
                "importance": 0.8,
                "confidence": 0.9,
                "evidence": "Do you like the rain?",
            },
            {
                "summary": "Mei drinks jasmine tea when she works late",
                "kind": "preference",
                "importance": 0.6,
                "confidence": 0.9,
                "evidence": "I drink jasmine tea when I work late",
            },
        ],
        "confidence": 0.9,
        "evidence": [],
    }

    async def scenario():
        llm = Scripted("I drink jasmine tea when I work late. Do you like the rain?")
        current = companion(
            tmp_path, {"self_memory": Worker(invented)}, llm=llm, self_memory_every=1
        )
        await current.reply("I have a cat called Bun", conversation_id="a")
        await current.settle()
        await current.close()
        return current.self_memories()

    assert run(scenario()) == ["Mei drinks jasmine tea when she works late"]


def test_said_again_about_herself_it_replaces_what_she_held(tmp_path):
    again = ("Mei drinks jasmine tea, when she works late!", "jasmine tea again tonight")

    async def scenario():
        llm = Scripted(
            "I drink jasmine tea when I work late.", "Yes, jasmine tea again tonight."
        )
        worker = Worker(about_herself(TEA, again))
        current = companion(tmp_path, {"self_memory": worker}, llm=llm, self_memory_every=1)
        await current.reply("what do you drink", conversation_id="a")
        await current.settle()
        await current.reply("again?", conversation_id="b")
        await current.settle()
        await current.close()
        store = current.runtime.memory_manager.store
        statuses = [
            (record.summary, record.status) for record in store.list_for_character("mei#self")
        ]
        return current.self_memories(), statuses

    held, statuses = run(scenario())
    # Once, as she said it last.
    assert held == ["Mei drinks jasmine tea, when she works late!"]
    assert statuses == [
        ("Mei drinks jasmine tea when she works late", "forgotten"),
        ("Mei drinks jasmine tea, when she works late!", "active"),
    ]


@pytest.mark.parametrize(
    "first, then",
    [
        (("Mei likes cats.", "I like cats"), ("Mei does not like cats.", "I don't like cats")),
        (("小梅喜歡貓。", "我喜歡貓"), ("小梅不喜歡貓。", "我不喜歡貓")),
        (("Mei's favourite colour is blue.", "blue"), ("Mei's favourite colour is red.", "red")),
        (("Mei likes horror films.", "I like horror"), ("Mei dislikes horror films.", "dislike")),
    ],
)
def test_what_she_says_differently_about_herself_is_not_a_repeat(tmp_path, first, then):
    """Nearly the same words can say the opposite. The newer statement was
    taken for a repeat and never kept, and she was asked to stay consistent
    with the older one."""

    async def scenario():
        llm = Scripted(f"So, {first[1]}.", f"Well, {then[1]}.")
        worker = Worker(about_herself(first, then))
        current = companion(tmp_path, {"self_memory": worker}, llm=llm, self_memory_every=1)
        for text in ("one", "two"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()
        return current.self_memories()

    assert run(scenario()) == [first[0], then[0]]


def test_she_keeps_the_newest_of_what_she_said_about_herself(tmp_path):
    async def scenario():
        llm = Scripted(
            "I drink jasmine tea when I work late.",
            "I have played the piano since I was six.",
            "I walk in the hills every Sunday.",
            "Hello.",
        )
        worker = Worker(about_herself(TEA, PIANO, HILLS))
        current = companion(
            tmp_path,
            {"self_memory": worker},
            llm=llm,
            self_memory_every=1,
            self_memories_kept=2,
        )
        for text in ("one", "two", "three"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.reply("four", conversation_id="a")
        await current.close()
        store = current.runtime.memory_manager.store
        statuses = {
            record.summary: record.status for record in store.list_for_character("mei#self")
        }
        return current.self_memories(), statuses, _self_lines(llm.calls[-1])

    kept, statuses, lines = run(scenario())
    assert kept == [PIANO[0], HILLS[0]]
    assert statuses[TEA[0]] == "forgotten"
    # Pushed out, it no longer stands in the conversation either.
    assert f"{SELF_LINE}{TEA[0]}" not in lines


def test_without_a_self_memory_worker_nothing_is_read(tmp_path):
    async def scenario():
        worker = Worker(about_herself(TEA))
        current = companion(tmp_path, {"self_memory": worker}, self_memory_every=0)
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await current.close()
        return worker.calls

    assert run(scenario()) == 0
    assert CompanionSettings().self_memory_every == 2
    assert CompanionSettings().self_memories_kept == 40
    assert CompanionSettings().self_memories_shown == 12


def test_the_host_can_show_and_edit_what_she_said_about_herself(tmp_path):
    """A memory page, and a host that kept such lines itself before."""

    async def scenario():
        current = companion(tmp_path)
        current.rewrite_self_memories(["A"])
        shown = current.self_memories()
        current.rewrite_self_memories(["A", "B"])  # arrived after the page was shown
        current.rewrite_self_memories(["A", "C"], edited_from=shown)
        edited = current.self_memories()
        current.rewrite_self_memories(["C", "D"])
        await current.close()
        return edited, current.self_memories(), current.memories(None)

    edited, rewritten, about_the_user = run(scenario())
    assert edited == ["A", "B", "C"]
    assert rewritten == ["C", "D"]
    assert about_the_user == []


def test_what_she_said_about_herself_survives_a_restart(tmp_path):
    async def scenario():
        first = companion(tmp_path)
        first.rewrite_self_memories(["Mei walks in the hills every Sunday"])
        await first.close()
        return companion(tmp_path).self_memories()

    assert run(scenario()) == ["Mei walks in the hills every Sunday"]


def test_what_she_said_about_herself_stands_in_the_conversation_until_forgotten(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        current.rewrite_self_memories([TEA[0], PIANO[0]])
        await current.reply("hello", conversation_id="a")
        first = llm.calls[-1]
        current.rewrite_self_memories([PIANO[0]])
        await current.reply("and now", conversation_id="a")
        await current.close()
        return system_context(first), _self_lines(first), _self_lines(llm.calls[-1])

    context, first, after = run(scenario())
    assert first == [f"{SELF_LINE}{TEA[0]}", f"{SELF_LINE}{PIANO[0]}"]
    assert after == [f"{SELF_LINE}{PIANO[0]}"]
    assert "stay consistent with" in context


def test_only_the_newest_of_what_she_said_about_herself_is_in_mind(tmp_path):
    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, self_memories_shown=2)
        current.rewrite_self_memories([TEA[0], PIANO[0], HILLS[0]])
        await current.reply("hello", conversation_id="a")
        await current.close()
        return _self_lines(llm.calls[-1])

    assert run(scenario()) == [f"{SELF_LINE}{PIANO[0]}", f"{SELF_LINE}{HILLS[0]}"]


def test_read_every_second_turn_nothing_she_said_is_skipped(tmp_path):
    asked = []

    def nothing(messages):
        asked.append(_listed_for_her(messages))
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        llm = Scripted("I drink jasmine tea when I work late.", "I walk in the hills on Sundays.")
        current = companion(
            tmp_path, {"self_memory": Worker(nothing)}, llm=llm, self_memory_every=2
        )
        await current.reply("one", conversation_id="a")
        await current.reply("two", conversation_id="a")
        await current.settle()
        await current.close()

    run(scenario())
    (listed,) = asked
    assert "jasmine tea" in listed and "hills" in listed


# --- what she says is checked before it is passed on ---------------------------------


def test_a_sentence_of_nothing_but_punctuation_is_not_passed_on(tmp_path):
    """A host showed "……" between two sentences as an empty subtitle."""

    async def scenario():
        llm = Scripted("Hello. …… How are you?", "……", "……", "Mind the cooling. ……")
        current = companion(tmp_path, llm=llm)
        heard: list[str] = []
        reply = await current.reply("hi", conversation_id="a", on_text_delta=heard.append)
        kept = current.runtime.history[-1].content
        silent: list[str] = []
        quiet = await current.reply("well?", conversation_id="a", on_text_delta=silent.append)
        calls = len(llm.calls)
        remark = await current.speak_up("a")
        await current.close()
        return "".join(heard), reply.text, kept, silent, quiet.text, calls, remark.text

    heard, text, kept, silent, quiet, calls, remark = run(scenario())
    assert heard == "Hello. How are you?"
    assert text == kept == "Hello. How are you?"
    # Nothing but "……" is her silence: nothing is passed on, and she is not
    # asked again for words.
    assert silent == [] and quiet == "" and calls == 2
    # A remark of nothing is no remark: she is asked again.
    assert remark == "Mind the cooling."


def test_a_remark_that_only_acknowledges_is_no_remark(tmp_path):
    """Nobody said anything: there is nothing to say "OK" to."""

    async def scenario():
        llm = Scripted("嗯。", "Mm-hmm.", "How is Bun taking the rain?", "好吧。", "Yeah.", "嗯，對。")
        current = companion(tmp_path, llm=llm)
        spoken = await current.speak_up("a")
        quiet = await current.speak_up("a")
        await current.close()
        return spoken.text, quiet.text

    assert run(scenario()) == ("How is Bun taking the rain?", "")


def test_a_short_call_of_her_own_is_not_an_acknowledgement(tmp_path):
    async def scenario():
        llm = Scripted("Hey.")
        current = companion(tmp_path, llm=llm)
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "Hey."


def test_an_acknowledgement_or_a_question_is_a_fine_reply(tmp_path):
    async def scenario():
        llm = Scripted("好。", "Why?")
        current = companion(tmp_path, llm=llm)
        first = await current.reply("can you wait a minute", conversation_id="a")
        second = await current.reply("I give up", conversation_id="a")
        await current.close()
        return first.text, second.text

    assert run(scenario()) == ("好。", "Why?")


def test_she_does_not_offer_help_like_an_assistant(tmp_path):
    """Small models fall into a support closing at the end of a reply; it
    breaks the character."""

    async def scenario():
        llm = Scripted(
            "The rain stopped. Let me know if you need anything else!",
            "雨停了。如果還有其他問題，隨時告訴我。",
        )
        current = companion(tmp_path, llm=llm)
        heard: list[str] = []
        english = await current.reply(
            "is it raining", conversation_id="a", on_text_delta=heard.append
        )
        chinese = await current.reply("還在下雨嗎", conversation_id="b")
        kept = current.runtime.history[-1].content
        await current.close()
        return "".join(heard), english.text, chinese.text, kept

    heard, english, chinese, kept = run(scenario())
    assert heard.strip() == english == "The rain stopped."
    assert chinese == kept == "雨停了。"


def test_a_reply_that_is_all_assistant_talk_is_asked_again(tmp_path):
    async def scenario():
        llm = Scripted(
            "Is there anything else I can help you with?",
            "Then go and look at the cooling first.",
        )
        current = companion(tmp_path, llm=llm)
        result = await current.reply("ok", conversation_id="a")
        asked_again = llm.calls[-1][-1].content
        await current.close()
        return result.text, asked_again

    text, asked_again = run(scenario())
    assert text == "Then go and look at the cooling first."
    assert asked_again == "ok"


def test_a_remark_that_is_all_assistant_talk_is_asked_again(tmp_path):
    async def scenario():
        llm = Scripted("有什麼我可以幫你的嗎？", "我在想週末去爬山。")
        current = companion(tmp_path, llm=llm)
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "我在想週末去爬山。"


def test_quoting_herself_in_a_remark_is_repeating_herself(tmp_path):
    """With nobody to answer, a small model quoted its own words back and
    reacted to them: an interview with itself."""

    async def scenario():
        llm = Scripted(
            "我今天想喝茉莉花茶。",
            "「想喝茉莉花茶」——我剛剛是這麼說的吧？真是的，我在自言自語什麼呢。",
        )
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "真是的，我在自言自語什麼呢。"


def test_quoting_the_user_in_a_remark_is_not_repeating_herself(tmp_path):
    async def scenario():
        llm = Scripted("那就待在家吧。", "你說「明天要下雨」，記得帶傘。")
        current = companion(tmp_path, llm=llm)
        await current.reply("明天要下雨", conversation_id="a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "你說「明天要下雨」，記得帶傘。"


def test_a_host_can_ask_for_a_remark_without_a_question(tmp_path):
    """The user stays quiet and she asks question after question."""

    async def scenario():
        llm = Scripted(
            "What are you reading?",
            "I finished the drawing. Do you want to see it?",
            "Are you there?",
        )
        current = companion(tmp_path, llm=llm)
        statement = await current.speak_up("a", statement_only=True)
        asked_again = llm.calls[-1][-1].content
        question = await current.speak_up("a")
        await current.close()
        return statement.text, asked_again, question.text

    statement, asked_again, question = run(scenario())
    assert statement == "I finished the drawing."
    assert "What are you reading" not in asked_again
    assert question == "Are you there?"


def test_conversations_taken_in_turns_are_each_read_once(tmp_path):
    """Two windows open, the user writing in each in turn. Reading counted the
    turns of all conversations together and kept one place to read on from:
    every second turn, conversation a was never read at all."""
    hers, theirs = [], []

    def her_lines(messages):
        hers.append(_listed_for_her(messages))
        return {"items": [], "confidence": 0.5, "evidence": []}

    def their_lines(messages):
        theirs.append(messages[1].content.split("User lines to extract from:")[1])
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        llm = Scripted(
            *(f"Reply {conversation}{number}." for number in "123" for conversation in "AB")
        )
        current = companion(
            tmp_path,
            {"self_memory": Worker(her_lines), "memory": Worker(their_lines)},
            llm=llm,
            self_memory_every=2,
            memory_every=2,
        )
        for number in "123":
            for conversation in "ab":
                await current.reply(f"line {conversation}{number}", conversation_id=conversation)
                await current.settle()
        await current.close()

    run(scenario())
    for read in (hers, theirs):
        joined = "\n".join(read)
        for conversation in "ab":
            name = conversation if read is theirs else conversation.upper()
            prefix = "line " if read is theirs else "Reply "
            for number in "12":
                assert joined.count(f"{prefix}{name}{number}") == 1


def test_why_she_is_asked_again_is_not_kept_as_the_users_words(tmp_path):
    """The reason for asking again went into the user's message, which the
    conversation kept, and memory extraction read it as something the user
    said."""
    read = []

    def their_lines(messages):
        read.append(messages[1].content.split("User lines to extract from:")[1])
        return {"items": [], "confidence": 0.5, "evidence": []}

    async def scenario():
        llm = Scripted(
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Is there anything else I can help you with?",
            "Then go and look at the cooling first.",
            "Mind the load on Amadeus, or your time machine breaks too.",
            "And keep the window open tonight.",
            "Mind the load on Amadeus, or your time machine breaks too.",
            "Have you eaten yet?",
        )
        current = companion(tmp_path, {"memory": Worker(their_lines)}, llm=llm, memory_every=1)
        await current.speak_up("a")
        await current.reply("ok", conversation_id="a")
        assistant_retry = llm.calls[-1]
        await current.reply("fine", conversation_id="a")
        repeat_retry = llm.calls[-1]
        await current.speak_up("a")
        remark_retry = llm.calls[-1]
        await current.settle()
        history = [(m.role, m.content) for m in current.runtime.history]
        await current.close()
        return assistant_retry, repeat_retry, remark_retry, history, read

    assistant_retry, repeat_retry, remark_retry, history, read = run(scenario())
    assert ("user", "ok") in history and ("user", "fine") in history
    assert not any("What you were about to say" in content for _, content in history)
    assert not any("What you were about to say" in lines for lines in read)
    # Told for that reply only, in the note of the turn, not as the user's words.
    for call, note in (
        (assistant_retry, NOT_AN_ASSISTANT),
        (repeat_retry, ALREADY_SAID),
        (remark_retry, ALREADY_SAID),
    ):
        assert f"For the next reply only: {note}" in system_context(call)
        assert note not in call[-1].content


def test_what_the_user_never_heard_her_say_is_not_kept(tmp_path):
    """The job reads her reply when the turn ends. Interrupted while it was
    played, or not used by the host, the rest of it was never heard."""

    async def scenario(cut):
        gate = asyncio.Event()
        worker = Worker(about_herself(TEA, PIANO), gate=gate)
        llm = Scripted(
            "I drink jasmine tea when I work late. I have played the piano since I was six."
        )
        current = companion(tmp_path / cut, {"self_memory": worker}, llm=llm, self_memory_every=1)
        await current.reply("tell me about you", conversation_id="a")
        await until(lambda: worker.calls)
        if cut == "interrupted":
            current.interrupt("I drink jasmine tea when I work late.")
        else:
            current.take_back("a")
        gate.set()
        await current.settle()
        await current.close()
        return current.self_memories()

    assert run(scenario("interrupted")) == [TEA[0]]
    assert run(scenario("taken back")) == []


def test_what_she_said_before_the_engine_kept_it_is_older_than_what_it_kept(tmp_path):
    """A host hands over what it kept itself. Dated now, the old lines took
    the places of the newest in her mind and pushed real recent ones out."""

    async def scenario():
        llm = Foreground()
        current = companion(tmp_path, llm=llm, self_memories_kept=3, self_memories_shown=2)
        current.rewrite_self_memories(["Recent one", "Recent two"])
        current.rewrite_self_memories(["Old one", "Old two"], edited_from=[], from_before=True)
        await current.reply("hello", conversation_id="a")
        await current.close()
        return current.self_memories(), _self_lines(llm.calls[-1])

    held, lines = run(scenario())
    # In the order given, before the rest; the oldest goes first.
    assert held == ["Old two", "Recent one", "Recent two"]
    assert lines == [f"{SELF_LINE}Recent one", f"{SELF_LINE}Recent two"]


@pytest.mark.parametrize(
    "said",
    [
        "請隨時保持警惕，他們就在附近。",
        "Let me know if you want to come along tomorrow!",
        "Is there anything else you remember about that night?",
        "店員說：「有什麼可以幫您的嗎？」",
        "如果還有其他問題，隨時告訴我。我先去煮水。那壺茶還熱著。",
        "如果你需要任何幫忙搬家的話，週六我有空。",
        "Let me know if you have any good horror films.",
        "Feel free to ask Alex, he was there too.",
        "Let me know if you need anything from the shop.",
    ],
)
def test_words_like_an_assistants_are_fine_in_character(tmp_path, said):
    """A guard, a friend, a scene retold: such words were taken for an
    assistant's closing and dropped. A closing comes at the end, in her own
    voice."""

    async def scenario():
        llm = Scripted(said, said)
        current = companion(tmp_path, llm=llm)
        heard: list[str] = []
        reply = await current.reply("and then?", conversation_id="a", on_text_delta=heard.append)
        remark = await current.speak_up("b")
        await current.close()
        return "".join(heard), reply.text, remark.text

    assert run(scenario()) == (said, said, said)


@pytest.mark.parametrize(
    "said",
    [
        "祝你程式編寫一切順利！",
        "如果你需要任何幫助，請告訴我。",
        "Let me know if you have any questions.",
        "Feel free to ask me anything.",
        "Let me know if you need anything else!",
    ],
)
def test_an_assistants_closing_at_the_end_is_still_left_out(tmp_path, said):
    async def scenario():
        llm = Scripted(f"這段程式應該沒問題了。{said}")
        current = companion(tmp_path, llm=llm)
        result = await current.reply("好了嗎", conversation_id="a")
        await current.close()
        return result.text

    assert run(scenario()) == "這段程式應該沒問題了。"


def test_naming_again_what_she_mentioned_in_a_reply_is_not_quoting_herself(tmp_path):
    async def scenario():
        llm = Scripted("我最近在追「進擊的巨人」。", "我昨天看了「進擊的巨人」最終季，結局比想像中安靜。")
        current = companion(tmp_path, llm=llm)
        await current.reply("最近在看什麼", conversation_id="a")
        result = await current.speak_up("a")
        await current.close()
        return result.text

    assert run(scenario()) == "我昨天看了「進擊的巨人」最終季，結局比想像中安靜。"


@pytest.mark.parametrize("said", ["*輕輕點頭。*", "好啊！😊", "Fine. *waves*", "See you! 👋"])
def test_a_stage_direction_or_an_emoji_at_the_end_is_kept_whole(tmp_path, said):
    """The closing asterisk of "*輕輕點頭。*" and the emoji after "好啊！" were
    taken for sentences of nothing but punctuation and dropped."""

    async def scenario():
        llm = Scripted(said)
        current = companion(tmp_path, llm=llm)
        heard: list[str] = []
        result = await current.reply("hi", conversation_id="a", on_text_delta=heard.append)
        kept = current.runtime.history[-1].content
        await current.close()
        return "".join(heard), result.text, kept

    assert run(scenario()) == (said, said, said)


def test_her_silence_is_kept_as_silence_not_as_an_empty_line(tmp_path):
    async def scenario():
        llm = Scripted("……")
        current = companion(tmp_path, llm=llm)
        result = await current.reply("say something", conversation_id="a")
        history = [(message.role, message.content) for message in current.runtime.history]
        await current.close()
        return result.text, history

    text, history = run(scenario())
    assert text == ""
    assert history[-1] == ("assistant", "……")


def test_a_reply_with_nothing_left_to_pass_on_is_kept_as_silence(tmp_path):
    async def scenario():
        llm = Scripted(*(["Mind the load on Amadeus, or your time machine breaks too."] * 3))
        current = companion(tmp_path, llm=llm)
        await current.speak_up("a")
        result = await current.reply("ok", conversation_id="a")
        newest = current.runtime.history[-1]
        await current.close()
        return result.text, (newest.role, newest.content)

    assert run(scenario()) == ("", ("assistant", "……"))


def test_a_host_can_tell_what_this_engine_offers():
    """A host that runs on several versions of the engine asks before it
    relies on these."""
    import inspect

    from ai_character_engine import BackgroundCognitionKind, CognitiveRole
    from ai_character_engine.companion import (
        ACKNOWLEDGEMENTS,
        ASSISTANT_SPEAK,
        SELF_MEMORY_LINE,
    )

    assert hasattr(CharacterCompanion, "self_memories")
    assert hasattr(CharacterCompanion, "rewrite_self_memories")
    rewrite = inspect.signature(CharacterCompanion.rewrite_self_memories).parameters
    assert "from_before" in rewrite and "edited_from" in rewrite
    assert "statement_only" in inspect.signature(CharacterCompanion.speak_up).parameters
    settings = CompanionSettings()
    for name in ("self_memory_every", "self_memories_kept", "self_memories_shown"):
        assert hasattr(settings, name)
    assert BackgroundCognitionKind.SELF_MEMORY_EXTRACTION.value == "self_memory_extraction"
    assert CognitiveRole.SELF_MEMORY.value == "self_memory"
    assert SELF_MEMORY_LINE == "- you said about yourself: "
    assert ASSISTANT_SPEAK and ACKNOWLEDGEMENTS


def test_an_asterisk_that_opens_a_stage_direction_starts_the_next_sentence():
    """Taken for a closing mark, the "*" that opens "*笑著點頭*" stayed with
    the sentence before it. Left out with that sentence, it left a lone "*",
    and a host that pairs asterisks then muted everything after it."""
    from ai_character_engine.companion.companion import _sentences

    assert _sentences("好啊！*笑著點頭*") == (["好啊！"], "*笑著點頭*")
    assert _sentences("*輕輕點頭。*") == (["*輕輕點頭。*"], "")


@pytest.mark.parametrize(
    "said, statement_only, kept",
    [
        ("今天好累喔。有什麼可以幫你的嗎？*歪頭看著你*", False, "今天好累喔。*歪頭看著你*"),
        ("你週末有空嗎？*歪頭看著你*我想去看電影。", True, "*歪頭看著你*我想去看電影。"),
    ],
)
def test_a_stage_direction_stays_whole_when_the_sentence_before_it_is_left_out(
    tmp_path, said, statement_only, kept
):
    async def scenario():
        current = companion(tmp_path, llm=Scripted(said))
        result = await current.speak_up("a", statement_only=statement_only)
        await current.close()
        return result.text

    assert run(scenario()) == kept


def _tea_from(quote):
    return {
        "items": [
            {
                "summary": TEA[0],
                "kind": "preference",
                "importance": 0.6,
                "confidence": 0.9,
                "evidence": quote,
            }
        ],
        "confidence": 0.9,
        "evidence": [],
    }


def test_what_she_said_around_a_stage_direction_is_still_hers(tmp_path):
    """The worker reads her lines without their stage directions; the check
    that the line was heard read them with, and found no such quote."""

    async def scenario():
        worker = Worker(_tea_from("I drink jasmine tea when I work late"))
        llm = Scripted("I *smiles* drink jasmine tea when I work late.")
        current = companion(tmp_path, {"self_memory": worker}, llm=llm, self_memory_every=1)
        await current.reply("tea or coffee", conversation_id="a")
        await current.settle()
        await current.close()
        return current.self_memories()

    assert run(scenario()) == [TEA[0]]


def test_what_she_said_before_the_conversation_was_trimmed_is_still_kept(tmp_path):
    """The conversation kept only its latest turns by the time the result
    came in: her line could not be found, but nothing says she was not heard."""

    async def scenario():
        gate = asyncio.Event()
        worker = Worker(about_herself(TEA), gate=gate)
        llm = Scripted("I drink jasmine tea when I work late.", "Fine.", "Sure.")
        current = companion(
            tmp_path,
            {"self_memory": worker},
            llm=llm,
            self_memory_every=1,
            max_history_messages=2,
        )
        await current.reply("tea or coffee", conversation_id="a")
        await until(lambda: worker.calls)
        await current.reply("ok", conversation_id="a")
        await current.reply("ok then", conversation_id="a")
        gate.set()
        await current.settle()
        await current.close()
        return current.self_memories()

    assert run(scenario()) == [TEA[0]]


SAD_MOOD = {"mood": "sad", "intensity": 0.8, "confidence": 0.9, "evidence": ["before he left"]}


def test_her_mood_is_read_from_both_sides_of_the_conversation(tmp_path):
    asked = []

    def judge(messages):
        asked.append(messages[1].content)
        return SAD_MOOD

    async def scenario():
        llm = Scripted("My father used to say that, before he left.")
        current = companion(tmp_path, {"mood": Worker(judge)}, llm=llm, mood_every=1)
        await current.reply("what did your father say", conversation_id="a")
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    snapshot = run(scenario())
    assert (snapshot.emotion, snapshot.mood_intensity) == ("sad", pytest.approx(0.8))
    assert "User: what did your father say" in asked[0]
    assert "Mei: My father used to say that, before he left." in asked[0]


def test_her_mood_is_read_every_second_turn_by_default(tmp_path):
    assert CompanionSettings().mood_every == 2
    judge = Worker(SAD_MOOD)

    async def scenario():
        llm = Scripted("The kettle is on.", "Rain again, of all things.", "The cat is asleep.")
        current = companion(tmp_path, {"mood": judge}, llm=llm, mood_every=2)
        for text in ("hello", "how are you", "and the cat"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()

    run(scenario())
    assert judge.calls == 1


def test_between_readings_of_her_mood_the_users_emotion_still_moves_it(tmp_path):
    async def scenario():
        llm = Scripted("The kettle is on.", "Rain again, of all things.", "The cat is asleep.")
        current = companion(
            tmp_path,
            {"emotion": Worker(warm_only_when_thanked), "mood": Worker(SAD_MOOD)},
            llm=llm,
            emotion_every=1,
            mood_every=2,
        )
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await current.reply("how are you", conversation_id="a")
        await current.settle()
        read = current.snapshot().emotion
        await current.reply("thank you for last night", conversation_id="a")
        await current.settle()
        thanked = current.snapshot().emotion
        await current.close()
        return read, thanked

    # The reading of turn two stands; the unremarkable turns leave it alone
    # and the thanks of turn three moves her again.
    assert run(scenario()) == ("sad", "happy")


def test_the_host_hears_when_her_mood_changes(tmp_path):
    heard = []

    async def scenario():
        llm = Scripted("The kettle is on.", "Rain again, of all things.")
        current = companion(
            tmp_path, {"mood": Worker(SAD_MOOD)}, llm=llm, mood_every=1, clock=lambda: 1000.0
        )
        current.on_mood_change = heard.append
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await current.reply("again", conversation_id="a")
        await current.settle()
        await current.close()

    run(scenario())
    # The second reading said the same at the same time: nothing to tell.
    assert [(s.emotion, s.mood_updated_at) for s in heard] == [("sad", 1000.0)]
    assert isinstance(heard[0], CompanionSnapshot)


def test_a_reading_that_changes_nothing_tells_the_host_nothing(tmp_path):
    heard = []

    async def scenario():
        current = companion(
            tmp_path, {"mood": Worker({**SAD_MOOD, "mood": "開心"})}, mood_every=1
        )
        current.on_mood_change = heard.append
        await current.reply("hello", conversation_id="a")
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    assert run(scenario()).emotion == "neutral"
    assert heard == []


def test_a_failing_listener_does_not_cost_her_the_mood(tmp_path):
    def broken(snapshot):
        raise RuntimeError("the page is gone")

    async def scenario():
        current = companion(tmp_path, {"mood": Worker(SAD_MOOD)}, mood_every=1)
        current.on_mood_change = broken
        await current.reply("hello", conversation_id="a")
        await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    assert run(scenario()).emotion == "sad"


def test_her_mood_keeps_fading_while_she_is_not_running(tmp_path):
    now = [1000.0]

    async def scenario():
        first = companion(tmp_path, {"mood": Worker(SAD_MOOD)}, mood_every=1, clock=lambda: now[0])
        await first.reply("hello", conversation_id="a")
        await first.settle()
        await first.close()
        now[0] = 1300.0
        later = companion(tmp_path, clock=lambda: now[0]).snapshot()
        now[0] = 1900.0
        gone = companion(tmp_path, clock=lambda: now[0]).snapshot()
        return later, gone

    later, gone = run(scenario())
    assert (later.emotion, later.mood_updated_at) == ("sad", 1000.0)
    assert later.mood_intensity == pytest.approx(0.8)
    assert gone.emotion == "neutral"


def test_a_host_can_tell_her_mood_is_there():
    from ai_character_engine.cognition import BackgroundCognitionKind, CognitiveRole

    assert CHARACTER_MOODS[0] == "neutral"
    assert hasattr(CompanionSettings(), "mood_every")
    assert BackgroundCognitionKind.CHARACTER_MOOD.value == "character_mood"
    assert CognitiveRole.MOOD.value == "mood"
    assert "mood_half_life_seconds" in CompanionSnapshot.__dataclass_fields__


def readings(*answers):
    """A mood worker that answers each reading with the next of ``answers``."""
    left = list(answers)
    return Worker(lambda messages: left.pop(0) if len(left) > 1 else left[0])


def test_a_reading_that_keeps_her_mood_tells_the_host_nothing(tmp_path):
    """A neutral reading, or a weaker one of another mood, leaves her mood
    as it was: no change to tell."""
    heard = []

    async def scenario():
        worker = readings(
            SAD_MOOD,
            {**SAD_MOOD, "mood": "neutral", "intensity": 0.0},
            {**SAD_MOOD, "mood": "worried", "intensity": 0.5},
        )
        llm = Scripted("The kettle is on.", "Rain again.", "The cat is asleep.")
        current = companion(tmp_path, {"mood": worker}, llm=llm, mood_every=1, clock=lambda: 1000.0)
        current.on_mood_change = heard.append
        for text in ("hello", "how are you", "and the cat"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        snapshot = current.snapshot()
        await current.close()
        return worker.calls, snapshot

    calls, snapshot = run(scenario())
    assert calls == 3
    assert (snapshot.emotion, snapshot.mood_intensity, snapshot.mood_updated_at) == (
        "sad",
        pytest.approx(0.8),
        1000.0,
    )
    assert [(s.emotion, s.mood_updated_at) for s in heard] == [("sad", 1000.0)]


def test_a_reading_weighs_against_her_mood_as_faded_by_her_settings(tmp_path):
    now = [1000.0]

    async def scenario(name, **settings):
        worker = readings(SAD_MOOD, {**SAD_MOOD, "mood": "worried", "intensity": 0.5})
        llm = Scripted("The kettle is on.", "Rain again.")
        current = companion(
            tmp_path / name, {"mood": worker}, llm=llm, mood_every=1,
            clock=lambda: now[0], **settings,
        )
        now[0] = 1000.0
        await current.reply("hello", conversation_id="a")
        await current.settle()
        now[0] = 1060.0
        await current.reply("how are you", conversation_id="a")
        await current.settle()
        mood = current.snapshot().emotion
        await current.close()
        return mood

    assert run(scenario("default")) == "sad"  # 0.8 a minute ago is still 0.7
    assert run(scenario("short_half_life", mood_half_life_seconds=60.0)) == "worried"  # 0.4 left


def test_the_rules_weigh_against_her_mood_as_faded_by_her_settings(tmp_path):
    now = [1060.0]

    async def scenario(name, **settings):
        current = companion(
            tmp_path / name, {"emotion": Worker({**WARM, "intensity": 0.6})}, emotion_every=1,
            clock=lambda: now[0], **settings,
        )
        current.runtime.state.apply(
            StatePatch(emotion="sad", mood_intensity=1.0, mood_updated_at=1000.0)
        )
        await current.reply("thank you for last night", conversation_id="a")
        await current.settle()
        mood = current.snapshot().emotion
        await current.close()
        return mood

    assert run(scenario("default")) == "sad"
    assert run(scenario("short_half_life", mood_half_life_seconds=10.0)) == "happy"


# --- on a turn her mood is read, the rules leave her mood to that reading ---

WARMLY = {**WARM, "intensity": 0.6}  # notable: happy, as strong as 0.6
HOSTILE = {**WARM, "emotion": "angry", "valence": -0.8, "stance": -1.0, "intensity": 0.8}
APPLIED = "_relationship_applied_observation"


def stored_mood(current):
    state = current.runtime.state
    return state.emotion, state.mood_intensity


async def eventually(condition):
    """Background results go through worker threads: wait in real time."""
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


def test_an_observation_before_the_reading_of_her_mood_does_not_move_it(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        current = companion(
            tmp_path,
            {"emotion": Worker(WARMLY), "mood": Worker({**SAD_MOOD, "intensity": 0.5}, gate=gate)},
            emotion_every=1,
            mood_every=1,
            clock=lambda: 1000.0,
        )
        trust = current.runtime.state.trust
        await current.reply("thank you for last night", conversation_id="a")
        await eventually(lambda: APPLIED in current.runtime.state.custom)
        between = stored_mood(current), current.runtime.state.trust > trust
        gate.set()
        await current.settle()
        after = stored_mood(current)
        await current.close()
        return between, after

    (mood_between, trusted), after = run(scenario())
    # The rules still count the warmth, not her mood: the reading decides it.
    assert mood_between == ("neutral", 0.0) and trusted
    assert after == ("sad", pytest.approx(0.5))


def test_an_observation_after_the_reading_of_her_mood_does_not_override_it(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        current = companion(
            tmp_path,
            {
                "emotion": Worker(WARMLY, gate=gate),
                "mood": Worker({**SAD_MOOD, "mood": "neutral", "intensity": 0.0}),
            },
            emotion_every=1,
            mood_every=1,
            clock=lambda: 1000.0,
        )
        await current.reply("thank you for last night", conversation_id="a")
        # A slow emotion analysis whose hold on the model ran out: the mood
        # is read and committed first.
        for key in list(current._access._holds):
            current._access.release(key)
        await eventually(lambda: "_mood_turn_ended_at" in current.runtime.state.custom)
        gate.set()
        await current.settle()
        after = stored_mood(current), APPLIED in current.runtime.state.custom
        await current.close()
        return after

    assert run(scenario()) == (("neutral", 0.0), True)


@pytest.mark.parametrize(
    ("start", "expected"),
    [(None, ("happy", 0.6)), (("sad", 0.9), ("sad", 0.9))],
    ids=["moves it", "with inertia"],
)
def test_on_a_turn_without_a_reading_of_her_mood_the_rules_move_it(tmp_path, start, expected):
    async def scenario():
        mood = Worker(SAD_MOOD)
        current = companion(
            tmp_path,
            {"emotion": Worker(WARMLY), "mood": mood},
            emotion_every=1,
            mood_every=2,
            clock=lambda: 1000.0,
        )
        if start is not None:
            current.runtime.state.apply(
                StatePatch(emotion=start[0], mood_intensity=start[1], mood_updated_at=1000.0)
            )
        await current.reply("thank you for last night", conversation_id="a")
        await current.settle()
        after = stored_mood(current)
        await current.close()
        return mood.calls, after

    calls, after = run(scenario())
    assert calls == 0
    assert after == (expected[0], pytest.approx(expected[1]))


def test_without_the_mood_worker_the_rules_move_her_mood_on_every_notable_turn(tmp_path):
    async def scenario():
        answers = [WARMLY, HOSTILE]
        current = companion(
            tmp_path,
            {"emotion": Worker(lambda messages: answers.pop(0))},
            emotion_every=1,
            mood_every=0,
            clock=lambda: 1000.0,
        )
        moods = []
        for text in ("thank you for last night", "you are useless"):
            await current.reply(text, conversation_id="a")
            await current.settle()
            moods.append(stored_mood(current))
        await current.close()
        return moods

    first, second = run(scenario())
    assert first == ("happy", pytest.approx(0.6))
    assert second == ("sad", pytest.approx(0.8))


def test_an_observation_moved_past_a_remark_of_hers_still_leaves_her_mood_to_the_reading(tmp_path):
    """She spoke up before the emotion analysis of the turn came back. Its
    observation is moved onto the turn of her remark, which read no mood; it
    is still of the turn whose mood is being read."""

    async def scenario():
        emotion_gate, mood_gate = asyncio.Event(), asyncio.Event()
        current = companion(
            tmp_path,
            {
                "emotion": Worker(WARMLY, gate=emotion_gate),
                "mood": Worker({**SAD_MOOD, "intensity": 0.5}, gate=mood_gate),
            },
            llm=Scripted("The kettle is on.", "Oh, the rain stopped."),
            emotion_every=1,
            mood_every=1,
            clock=lambda: 1000.0,
        )
        await current.reply("thank you for last night", conversation_id="a")
        await current.speak_up("a")
        revision = current._tasks.revision
        emotion_gate.set()
        await eventually(lambda: APPLIED in current.runtime.state.custom)
        observation = current.runtime.state.custom["observed_user_emotion"]
        between = stored_mood(current)
        mood_gate.set()
        await current.settle()
        after = stored_mood(current)
        await current.close()
        return revision, observation, between, after

    revision, observation, between, after = run(scenario())
    assert observation["base_revision"] == revision  # moved onto her remark
    assert between == ("neutral", 0.0)
    assert after == ("sad", pytest.approx(0.5))


# --- a host's state policy is its own ----------------------------------------


def test_companions_sharing_a_host_policy_keep_their_own_clock_and_readings(tmp_path):
    """Each companion wires the policy to its own clock and mood readings; on
    the host's one instance the second companion overwrote the first's."""
    from ai_character_engine.state.relationship import RelationshipStatePolicy

    shared = RelationshipStatePolicy()
    host_clock = shared.clock

    def build(name, now):
        return CharacterCompanion(
            character=CharacterProfile(id=name, name=name, description="A researcher."),
            llm=Foreground(),
            storage_dir=tmp_path / name,
            settings=CompanionSettings(mood_every=0),
            clock=lambda: now,
            state_policy=shared,
        )

    async def scenario():
        first, second = build("a", 1000.0), build("b", 2000.0)
        first._mood_read_at.append(5)
        policies = first.runtime.state_policy, second.runtime.state_policy
        result = (
            [policy.clock() for policy in policies],
            [policy.mood_read_for(5) for policy in policies],
        )
        await first.close()
        await second.close()
        return result

    clocks, read = run(scenario())
    assert clocks == [1000.0, 2000.0]
    assert read == [True, False]
    assert shared.clock is host_clock
    assert shared.mood_read_for(5) is False


# --- the host hears of her mood -----------------------------------------------


def test_the_host_hears_her_mood_change_even_when_saving_the_state_fails(tmp_path):
    heard = []

    async def scenario():
        gate = asyncio.Event()
        current = companion(
            tmp_path, {"mood": Worker(SAD_MOOD, gate=gate)}, mood_every=1, clock=lambda: 1000.0
        )
        current.on_mood_change = heard.append
        await current.reply("hello", conversation_id="a")
        save = current._save_state

        def broken():
            raise OSError("disk full")

        current._save_state = broken
        gate.set()
        await current.settle()
        current._save_state = save
        mood = current.runtime.state.emotion
        await current.close()
        return mood

    assert run(scenario()) == "sad"
    assert [(s.emotion, s.mood_updated_at) for s in heard] == [("sad", 1000.0)]


def test_an_async_listener_is_run_on_the_event_loop(tmp_path):
    heard = []

    async def listener(snapshot):
        await asyncio.sleep(0)
        heard.append(snapshot)

    async def scenario():
        current = companion(
            tmp_path, {"mood": Worker(SAD_MOOD)}, mood_every=1, clock=lambda: 1000.0
        )
        current.on_mood_change = listener
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await until(lambda: heard)
        await current.close()

    run(scenario())
    assert [(s.emotion, s.mood_updated_at) for s in heard] == [("sad", 1000.0)]


def test_a_failing_async_listener_does_not_cost_her_the_mood(tmp_path, caplog):
    async def broken(snapshot):
        raise RuntimeError("the page is gone")

    async def scenario():
        current = companion(tmp_path, {"mood": Worker(SAD_MOOD)}, mood_every=1)
        current.on_mood_change = broken
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await until(lambda: "the page is gone" in caplog.text)
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    with caplog.at_level("WARNING"):
        assert run(scenario()).emotion == "sad"


def test_an_async_listener_without_a_running_loop_is_dropped_with_a_warning(tmp_path, caplog):
    called = []

    async def listener(snapshot):
        called.append(snapshot)

    current = companion(tmp_path)
    current.on_mood_change = listener
    with caplog.at_level("WARNING"):
        current._tell_mood()
    assert called == []
    assert "no event loop is running" in caplog.text
    run(current.close())


# --- reply check: a slip in her reply is pointed out on her next reply ----------------

SLIPPING = ("As an AI, hello. ", "How are you?")
SLIP = {
    "issues": [
        {"kind": "broke_character", "evidence": "As an AI, hello.", "fix": "Open differently."}
    ]
}
REPLY_NOTE = "About your last reply: Open differently."


def slip_once(messages):
    """Finds the slip in the first reply it reads, none after; says yes when
    asked about it again."""
    if '"behind"' in messages[0].content:
        return {"behind": "yes", "word": "AI"}
    slip_once.calls += 1
    return SLIP if slip_once.calls == 1 else {"issues": []}


def new_in(llm, turn):
    """What the prompt of ``turn`` (0-based) added to the one before it."""
    before = llm.calls[turn - 1] if turn else []
    return "\n".join(message.content for message in llm.calls[turn][len(before) :])


async def three_turns(tmp_path, worker, between=None, **settings):
    llm = Foreground(SLIPPING)
    current = companion(tmp_path, {"reply_check": worker}, llm=llm, **settings)
    for number, text in enumerate(("hi", "what are you doing", "and then")):
        await current.reply(text, conversation_id="a")
        await current.settle()
        if between is not None and number == 0:
            between(current)
    await current.close()
    return llm


def test_a_slip_in_her_reply_is_pointed_out_on_her_next_reply_only(tmp_path):
    slip_once.calls = 0
    llm = run(three_turns(tmp_path, Worker(slip_once), reply_check_every=1))
    assert REPLY_NOTE not in new_in(llm, 0)
    assert REPLY_NOTE in new_in(llm, 1)
    assert REPLY_NOTE not in new_in(llm, 2)
    assert slip_once.calls == 3


def test_the_reply_check_runs_every_turn_by_default_and_not_at_zero(tmp_path):
    assert CompanionSettings().reply_check_every == 1
    worker = Worker(SLIP)
    llm = run(three_turns(tmp_path, worker, reply_check_every=0))
    assert worker.calls == 0
    assert all(REPLY_NOTE not in new_in(llm, turn) for turn in range(3))


def test_a_reply_without_slips_leaves_no_trace(tmp_path):
    worker = Worker({"issues": []})
    llm = run(three_turns(tmp_path, worker, reply_check_every=1))
    assert worker.calls == 3
    assert all("About your last reply" not in new_in(llm, turn) for turn in range(3))


def test_a_note_on_a_reply_she_was_cut_off_in_is_dropped(tmp_path):
    slip_once.calls = 0
    llm = run(
        three_turns(
            tmp_path,
            Worker(slip_once),
            between=lambda current: current.interrupt("Hello."),
            reply_check_every=1,
        )
    )
    assert REPLY_NOTE not in new_in(llm, 1)


def test_a_note_on_a_reply_the_host_rewrote_is_dropped(tmp_path):
    slip_once.calls = 0
    llm = run(
        three_turns(
            tmp_path,
            Worker(slip_once),
            between=lambda current: current.replace_reply("Hello. What is new?"),
            reply_check_every=1,
        )
    )
    assert REPLY_NOTE not in new_in(llm, 1)


def test_a_note_is_for_the_conversation_of_the_reply(tmp_path):
    async def scenario():
        slip_once.calls = 0
        llm = Foreground(SLIPPING)
        current = companion(
            tmp_path, {"reply_check": Worker(slip_once)}, llm=llm, reply_check_every=1
        )
        await current.reply("hi", conversation_id="a")
        await current.settle()
        await current.reply("hi", conversation_id="b")
        await current.reply("and you", conversation_id="a")
        await current.close()
        return llm

    llm = run(scenario())
    assert all(REPLY_NOTE not in "\n".join(m.content for m in call) for call in llm.calls)


def test_a_late_note_is_not_given_to_a_later_reply(tmp_path):
    async def scenario():
        slip_once.calls = 0
        gate = asyncio.Event()
        llm = Foreground(SLIPPING)
        current = companion(
            tmp_path, {"reply_check": Worker(slip_once, gate=gate)}, llm=llm, reply_check_every=2
        )
        await current.reply("hi", conversation_id="a")
        # Checked after this turn, every second one; the check is slow.
        await current.reply("what are you doing", conversation_id="a")
        await current.reply("and then", conversation_id="a")
        gate.set()
        await current.settle()
        await current.reply("and after that", conversation_id="a")
        await current.close()
        return llm

    llm = run(scenario())
    assert slip_once.calls >= 1
    assert REPLY_NOTE not in new_in(llm, 2)
    assert REPLY_NOTE not in new_in(llm, 3)


def test_her_reply_is_checked_after_the_users_emotion_and_before_her_mood(tmp_path):
    async def scenario():
        workers = {
            "emotion": Worker(NEUTRAL, name="emotion"),
            "mood": Worker(
                {"mood": "calm", "intensity": 0.3, "confidence": 0.9, "evidence": []},
                name="mood",
            ),
            "reply_check": Worker({"issues": []}, name="reply_check"),
        }
        current = companion(
            tmp_path, workers, emotion_every=1, mood_every=1, reply_check_every=1
        )
        await current.reply("hello", conversation_id="a")
        await current.settle()
        await current.close()
        return list(Worker.order)

    assert run(scenario()) == ["emotion", "reply_check", "mood"]


def test_a_note_on_a_reply_the_host_took_back_is_dropped(tmp_path):
    slip_once.calls = 0
    llm = run(
        three_turns(
            tmp_path,
            Worker(slip_once),
            between=lambda current: current.take_back("a"),
            reply_check_every=1,
        )
    )
    # The exchange left the conversation: the next prompt is no longer longer.
    assert REPLY_NOTE not in "\n".join(message.content for message in llm.calls[1])



class Openings(Foreground):
    """Her replies in turn, each in one part."""

    def __init__(self, *replies):
        super().__init__()
        self.replies = list(replies)

    async def stream_generate(self, messages, *, tools=None):
        self.parts = (self.replies.pop(0) if len(self.replies) > 1 else self.replies[0],)
        async for chunk in super().stream_generate(messages, tools=tools):
            yield chunk


def test_her_opening_said_again_is_pointed_out_without_the_model_seeing_it(tmp_path):
    async def scenario():
        llm = Openings(
            "Hello there friend, the sun is out and the weather is lovely today.",
            "Hello there friend, I finished that long book about old trains.",
            "Fine.",
        )
        current = companion(
            tmp_path, {"reply_check": Worker({"issues": []})}, llm=llm, reply_check_every=1
        )
        for text in ("hi", "how was it", "and then"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()
        return llm

    llm = run(scenario())
    assert "About your last reply" not in new_in(llm, 1)
    assert "About your last reply: Do not open with the same words again." in new_in(llm, 2)


def test_her_opening_said_again_is_pointed_out_without_a_model_for_the_check(tmp_path):
    """A host whose mapping names other workers but no reply_check client
    still gets the opening check: it needs no model."""

    async def scenario():
        llm = Openings(
            "Hello there friend, the sun is out and the weather is lovely today.",
            "Hello there friend, I finished that long book about old trains.",
            "Fine.",
        )
        current = companion(
            tmp_path, {"emotion": Worker(NEUTRAL)}, llm=llm, reply_check_every=1, emotion_every=1
        )
        for text in ("hi", "how was it", "and then"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()
        return llm

    llm = run(scenario())
    assert "About your last reply: Do not open with the same words again." in new_in(llm, 2)


def test_an_empty_mapping_still_turns_all_background_work_off(tmp_path):
    """As in 1.1: background_llm={} runs nothing, the opening check included."""

    async def scenario():
        llm = Openings(
            "Hello there friend, the sun is out and the weather is lovely today.",
            "Hello there friend, I finished that long book about old trains.",
            "Fine.",
        )
        current = companion(tmp_path, {}, llm=llm, reply_check_every=1)
        for text in ("hi", "how was it", "and then"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        background = current._background
        await current.close()
        return llm, background

    llm, background = run(scenario())
    assert background is None
    assert "About your last reply" not in new_in(llm, 2)


# --- how the user has been lately -------------------------------------------------------

from datetime import datetime  # noqa: E402

from ai_character_engine.companion import DiaryEntry, UserState  # noqa: E402

TIRED = "I am so tired, work never ends"
TIRED_STATE = {
    "energy": "low",
    "mood_trend": "down",
    "concerns": [{"concern": "work never ends", "evidence": TIRED}],
    "evidence": [TIRED],
}
LATELY = "- user lately: energy low, mood down; concerns: work never ends"


class Clock:
    """The system clock, moved on at will: what the background writes is
    dated by the system clock, her diary and the user state by hers."""

    def __init__(self):
        self.ahead = 0.0

    def __call__(self):
        import time

        return time.time() + self.ahead


def test_how_the_user_has_been_is_read_every_sixth_turn_and_not_at_zero(tmp_path):
    assert CompanionSettings().user_state_every == 6
    assert CompanionSettings().user_state_ttl_hours == 48.0

    async def scenario(every):
        worker = Worker(TIRED_STATE)
        current = companion(tmp_path, {"user_state": worker}, user_state_every=every)
        for text in ("one", "two", "three", "four"):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()
        return worker.calls

    assert run(scenario(2)) == 2
    assert run(scenario(0)) == 0


async def tired_user(tmp_path, clock=None, *, llm=None, then=(), **settings):
    llm = llm or Foreground()
    def once(messages):
        # How the user was, read from the turn that showed it; nothing after.
        return TIRED_STATE if TIRED in messages[1].content else {"energy": "unknown"}

    current = companion(
        tmp_path, {"user_state": Worker(once)}, llm=llm, clock=clock,
        user_state_every=1, **settings
    )
    await current.reply(TIRED, conversation_id="a")
    await current.settle()
    for step in then:
        step(current)
        await current.reply("and now?", conversation_id="a")
        await current.settle()
    snapshot = current.snapshot()
    await current.close()
    return llm, snapshot


def test_how_the_user_has_been_reaches_her_next_reply_and_the_snapshot(tmp_path):
    llm, snapshot = run(tired_user(tmp_path, then=[lambda current: None]))
    assert LATELY in new_in(llm, 1).splitlines()
    assert snapshot.user_state == UserState(
        energy="low",
        mood_trend="down",
        concerns=("work never ends",),
        evidence=(TIRED,),
        updated_at=snapshot.user_state.updated_at,
    )


def test_how_the_user_has_been_holds_across_conversations_and_a_restart(tmp_path):
    run(tired_user(tmp_path))

    async def later():
        llm = Foreground()
        current = companion(tmp_path, llm=llm)
        await current.reply("hello", conversation_id="b")
        snapshot = current.snapshot()
        await current.close()
        return llm, snapshot

    llm, snapshot = run(later())
    assert snapshot.user_state.concerns == ("work never ends",)
    assert LATELY in new_in(llm, 0).splitlines()


def test_how_the_user_has_been_is_forgotten_after_its_time(tmp_path):
    clock = Clock()

    def hours_later(hours):
        def step(current):
            clock.ahead = hours * 3600

        return step

    llm, snapshot = run(tired_user(tmp_path, clock, then=[hours_later(47), hours_later(49)]))
    assert LATELY in "\n".join(message.content for message in llm.calls[1])
    assert snapshot.user_state is None
    assert LATELY not in "\n".join(message.content for message in llm.calls[2])


def test_how_the_user_seemed_on_each_turn_is_read_into_the_user_state(tmp_path):
    async def scenario():
        prompts = []

        def state(messages):
            prompts.append(messages[1].content)
            return TIRED_STATE

        current = companion(
            tmp_path,
            {"emotion": Worker(WARM), "user_state": Worker(state)},
            emotion_every=1,
            user_state_every=2,
        )
        for text in ("thank you", TIRED):
            await current.reply(text, conversation_id="a")
            await current.settle()
        await current.close()
        return prompts

    (prompt,) = run(scenario())
    assert "- glad (valence 0.9, stance 1.0)" in prompt


# --- her diary ----------------------------------------------------------------------

BLACK_TEA = "I like strong black tea. "
DAWN = "I am Dawn and I work at a print shop"


def her_tea(messages):
    if BLACK_TEA.strip() not in messages[1].content.split("Character lines to extract from:")[1]:
        return {"items": [], "confidence": 0.5, "evidence": []}
    return {
        "items": [
            {"summary": "Mei likes strong black tea.", "kind": "taste", "importance": 0.6,
             "confidence": 0.9, "evidence": BLACK_TEA.strip()}
        ],
        "confidence": 0.9,
        "evidence": [],
    }


def dawns_work(messages):
    if DAWN not in messages[1].content.split("User lines to extract from:")[1]:
        return {"items": [], "confidence": 0.5, "evidence": []}
    return {
        "items": [
            {"summary": "Dawn works at a print shop.", "kind": "fact", "importance": 0.8,
             "confidence": 0.9, "evidence": DAWN}
        ],
        "confidence": 0.9,
        "evidence": [],
    }


ENTRY = (
    "Dawn told me she works at a print shop. I told her about my strong black tea. "
    "It was a quiet day."
)


class DiaryWorker(Worker):
    def __init__(self, text=ENTRY, evidence=("Dawn works at a print shop.",)):
        self.prompts = []
        super().__init__(self.answer, name="diary")
        self.text, self.evidence = text, list(evidence)

    def answer(self, messages):
        self.prompts.append(messages[1].content)
        return {"text": self.text, "evidence": self.evidence}


async def a_day(tmp_path, diary, clock=None, *, llm=None, texts=(DAWN, "nice"), **settings):
    current = companion(
        tmp_path,
        {"memory": Worker(dawns_work), "self_memory": Worker(her_tea), "diary": diary},
        llm=llm or Foreground((BLACK_TEA, "How are you?")),
        clock=clock,
        memory_every=1,
        self_memory_every=1,
        **settings,
    )
    for text in texts:
        await current.reply(text, conversation_id="a")
        await current.settle()
    return current


def test_her_diary_is_written_when_the_host_asks_and_kept(tmp_path):
    clock = Clock()

    async def scenario():
        diary = DiaryWorker()
        current = await a_day(tmp_path, diary, clock)
        entry = await current.write_diary()
        kept = current.diary()
        await current.close()
        return diary, entry, kept

    diary, entry, kept = run(scenario())
    (prompt,) = diary.prompts
    assert "- Dawn works at a print shop." in prompt
    assert "- Mei likes strong black tea." in prompt
    assert entry == DiaryEntry(
        date=datetime.fromtimestamp(clock()).date().isoformat(),
        text=ENTRY,
        evidence=("Dawn works at a print shop.",),
        conversation_ids=("a",),
        until=entry.until,
    )
    assert kept == (entry,)
    assert (tmp_path / "engine" / "diary.jsonl").is_file()

    async def after_a_restart():
        current = companion(tmp_path)
        kept = current.diary()
        await current.close()
        return kept

    assert run(after_a_restart()) == (entry,)


def test_without_a_conversation_there_is_no_diary(tmp_path):
    async def scenario():
        diary = DiaryWorker()
        current = companion(tmp_path, {"diary": diary})
        entry = await current.write_diary()
        await current.close()
        return diary, entry

    diary, entry = run(scenario())
    assert entry is None
    assert diary.calls == 0


def test_an_entry_resting_on_nothing_of_the_day_is_not_kept(tmp_path):
    async def scenario():
        current = await a_day(tmp_path, DiaryWorker(evidence=["We went to the sea."]))
        entry = await current.write_diary()
        kept = current.diary()
        await current.close()
        return entry, kept

    assert run(scenario()) == (None, ())


def test_her_diary_is_written_on_its_own_once_a_day(tmp_path):
    assert CompanionSettings().diary_every_hours == 24.0
    clock = Clock()

    async def scenario():
        diary = DiaryWorker()
        current = await a_day(tmp_path, diary, clock)
        written = [len(current.diary())]
        clock.ahead = 23 * 3600
        await current.reply("still here", conversation_id="a")
        await current.settle()
        written.append(len(current.diary()))
        clock.ahead = 25 * 3600
        await current.reply("good morning", conversation_id="a")
        await current.settle()
        written.append(len(current.diary()))
        await current.reply("and again", conversation_id="a")
        await current.settle()
        written.append(len(current.diary()))
        await current.close()
        return diary, written

    diary, written = run(scenario())
    assert written == [0, 0, 1, 1]
    assert diary.calls == 1


def test_no_day_without_a_conversation_and_none_on_its_own_at_zero(tmp_path):
    clock = Clock()

    async def scenario(hours):
        diary = DiaryWorker()
        current = await a_day(tmp_path / str(hours), diary, clock, diary_every_hours=hours)
        clock.ahead = 25 * 3600
        await current.reply("good morning", conversation_id="a")
        await current.settle()
        written = len(current.diary())
        await current.close()
        clock.ahead = 0
        return written

    assert run(scenario(0)) == 0
    assert run(scenario(24.0)) == 1


def test_the_start_of_her_last_entry_is_in_her_system_prompt(tmp_path):
    long_entry = "Dawn came by. She works at a print shop. We had tea. It rained."

    async def scenario(in_context):
        llm = Foreground((BLACK_TEA, "How are you?"))
        current = await a_day(
            tmp_path / str(in_context), DiaryWorker(long_entry), llm=llm,
            diary_in_context=in_context,
        )
        await current.write_diary()
        await current.reply("hello again", conversation_id="b")
        await current.close()
        return llm.calls[-1][0].content

    system = run(scenario(True))
    assert "in your own words:\nDawn came by. She works at a print shop." in system
    assert "We had tea" not in system
    assert "in your own words" not in run(scenario(False))


def test_her_mood_through_the_day_is_part_of_what_happened(tmp_path):
    clock = Clock()

    async def scenario():
        diary = DiaryWorker()
        current = companion(
            tmp_path,
            {"mood": Worker({"mood": "happy", "intensity": 0.8, "confidence": 0.9,
                             "evidence": []}), "diary": diary},
            clock=clock,
            mood_every=1,
        )
        await current.reply(DAWN, conversation_id="a")
        await current.settle()
        felt_at = clock()
        # Her day ends where the entry is written, that moment excluded; on
        # Windows before Python 3.13 the system clock ticks every 15.6 ms, so
        # without this the entry could be written at the moment she felt it.
        clock.ahead += 1
        await current.write_diary()
        await current.close()
        return diary.prompts[0], felt_at

    prompt, felt_at = run(scenario())
    assert f"- {datetime.fromtimestamp(felt_at):%H:%M} happy" in prompt


@pytest.mark.parametrize(
    "setting", ["user_state_every", "user_state_ttl_hours", "diary_every_hours"]
)
def test_the_diary_and_user_state_settings_are_never_negative(setting):
    with pytest.raises(ValueError, match=setting):
        CompanionSettings(**{setting: -1})


def test_what_her_days_were_made_of_is_let_go_after_eight_days(tmp_path):
    clock = Clock()

    async def scenario():
        current = companion(tmp_path, clock=clock)
        await current.reply("hello", conversation_id="a")
        await current.close()
        clock.ahead = 9 * 86400
        later = companion(tmp_path, clock=clock)
        await later.reply("hello again", conversation_id="a")
        await later.close()

    run(scenario())
    lines = (tmp_path / "engine" / "diary_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["conversation"] for line in lines] == ["a"]
    assert json.loads(lines[0])["at"] > clock() - 86400


def test_her_remarks_the_user_has_not_answered_are_counted(tmp_path):
    """She spoke up and nobody answered: the next time she speaks up she must
    not take her own last remark for the user's question."""

    async def scenario():
        llm = Foreground(("Hello there.",))
        current = companion(tmp_path, llm=llm)
        await current.reply("hi", conversation_id="a")
        counts = [current.unanswered_remarks("a")]
        llm.parts = ("Shall we talk about games?",)
        await current.speak_up("a")
        counts.append(current.unanswered_remarks("a"))
        await current.remember_remark("a", "Anyone there?")
        counts.append(current.unanswered_remarks("a"))
        counts.append(current.unanswered_remarks("b"))
        await current.reply("sorry, I was away", conversation_id="a")
        counts.append(current.unanswered_remarks("a"))
        await current.close()
        return counts

    assert run(scenario()) == [0, 1, 2, 0, 0]


def test_speaking_up_again_unanswered_she_is_told_her_last_remark_was_hers(tmp_path):
    async def scenario():
        llm = Foreground(("Hello there.",))
        current = companion(tmp_path, llm=llm)
        await current.reply("hi", conversation_id="a")
        llm.parts = ("Shall we talk about games?",)
        await current.speak_up("a")
        first = llm.calls[-1][-1].content
        llm.parts = ("I found a new game.",)
        await current.speak_up("a")
        second = llm.calls[-1][-1].content
        await current.close()
        return first, second

    first, second = run(scenario())
    assert "not answered" not in first
    assert "1 time" in second and "not answered" in second
