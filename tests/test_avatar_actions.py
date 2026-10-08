"""The face and the gesture of each line she says (reply_actions)."""

from __future__ import annotations

import asyncio

import pytest

from ai_character_engine.companion import AvatarChoices, CompanionSettings, LineActions, ReplyActions
from ai_character_engine.companion.avatar_actions import read_pick
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.state.models import StatePatch
from tests.test_companion import Foreground, Worker, companion, run

CHOICES = AvatarChoices(
    expressions=("joy", "sadness", "anger"),
    motions={"nod": "agreeing", "wave": ""},
    mood_faces={"happy": "joy", "sad": "sadness"},
)


class Model:
    """A background model: a fixed answer (or a function of the messages), late if asked."""

    def __init__(self, answer='{"expression":"joy","motion":"nod","intensity":0.5}', *, delay=0.0, fail=False):
        self.answer = answer
        self.delay = delay
        self.fail = fail
        self.asked: list[list] = []

    async def generate(self, messages, *, tools=None):
        self.asked.append(list(messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model is down")
        answer = self.answer(messages) if callable(self.answer) else self.answer
        return LLMResponse(text=answer)


def actions(model, *, choices=CHOICES, mood=("neutral", 0.0), timeout=6.0):
    return ReplyActions(client=model, choices=choices, mood=lambda: mood, timeout_seconds=timeout)


# --- reading an answer --------------------------------------------------------------------------


def test_an_answer_is_kept_to_the_hosts_lists():
    assert read_pick('{"expression":"joy","motion":"nod","intensity":0.4}', CHOICES) == LineActions(
        "joy", "nod", 0.4
    )
    assert read_pick('{"expression":"smug","motion":"bow"}', CHOICES) == LineActions(None, None, 1.0)


def test_spelling_case_brackets_and_null_words_are_forgiven():
    assert read_pick('{"expression":" [JOY] ","motion":"None","intensity":"0.3"}', CHOICES) == LineActions(
        "joy", None, 0.3
    )


def test_intensity_stays_between_zero_and_one():
    assert read_pick('{"expression":"joy","intensity":7}', CHOICES).intensity == 1.0
    assert read_pick('{"expression":"joy","intensity":-1}', CHOICES).intensity == 0.0
    assert read_pick('{"expression":"joy","intensity":"lots"}', CHOICES).intensity == 1.0


def test_a_fenced_answer_with_its_thinking_and_a_repeat_is_read():
    raw = '<think>joy?</think>```json\n{"expression":"anger"}\n```\n{"expression":"joy"}'
    assert read_pick(raw, CHOICES).expression == "anger"


def test_an_answer_that_is_not_json_is_nothing():
    assert read_pick("joy, probably", CHOICES) is None
    assert read_pick('["joy"]', CHOICES) is None


# --- picking ------------------------------------------------------------------------------------


def test_a_line_is_picked_with_the_one_before_it_and_her_mood():
    model = Model()
    picker = actions(model, mood=("sad", 0.42))

    async def two():
        first = await picker.pick("You came!")
        second = await picker.pick("I am tired today.")
        return first, second

    first, second = asyncio.run(two())
    assert first == LineActions("joy", "nod", 0.5) and second is not None
    asked = model.asked[1][-1].content
    assert "Previous line: You came!" in asked and "Line: I am tired today." in asked
    assert "Her mood: sad (0.42)" in asked
    assert "nod (agreeing)" in asked and "wave" in asked


def test_one_line_at_a_time_and_nothing_queues():
    model = Model(delay=0.2)
    picker = actions(model)

    async def together():
        return await asyncio.gather(picker.pick("You came!"), picker.pick("Look at this."))

    first, second = asyncio.run(together())
    assert first is not None and second is None
    assert len(model.asked) == 1


def test_a_slow_model_is_given_up_on():
    assert asyncio.run(actions(Model(delay=1.0), timeout=0.05).pick("You came!")) is None


def test_a_failing_model_is_a_line_without_a_face():
    assert asyncio.run(actions(Model(fail=True)).pick("You came!")) is None


def test_nothing_to_pick_from_or_no_model_asks_nothing():
    model = Model()
    assert asyncio.run(actions(model, choices=AvatarChoices()).pick("You came!")) is None
    assert asyncio.run(actions(None).pick("You came!")) is None
    assert asyncio.run(actions(model).pick("   ")) is None
    assert model.asked == []


def test_the_voice_is_the_first_face_picked_else_her_mood_else_nothing():
    picker = actions(Model(), mood=("sad", 0.8))
    assert picker.voice() == "sadness"
    asyncio.run(picker.pick("You came!"))
    assert picker.voice() == "joy"
    assert actions(Model(), mood=("calm", 0.8)).voice() is None


def test_choices_are_copied_and_mood_names_lowered():
    motions = {"nod": None}
    choices = AvatarChoices(expressions=["joy"], motions=motions, mood_faces={"Happy": "joy"})
    motions["wave"] = "waving"
    assert choices.expressions == ("joy",) and choices.motions == {"nod": ""}
    assert choices.mood_faces == {"happy": "joy"}


# --- from the companion -------------------------------------------------------------------------


def test_her_actions_model_is_the_actions_client_of_a_mapping(tmp_path):
    model = Model()
    mei = companion(tmp_path, {"actions": model})
    picked = run(mei.reply_actions(CHOICES).pick("You came!"))
    assert picked.expression == "joy" and len(model.asked) == 1


def test_a_mapping_without_an_actions_client_picks_nothing(tmp_path):
    mei = companion(tmp_path, {"mood": Worker({"emotion": "happy"})})
    picker = mei.reply_actions(CHOICES)
    assert not picker.can_pick
    assert run(picker.pick("You came!")) is None


def test_one_background_client_or_none_at_all_serves_too(tmp_path):
    single = Model()
    mei = companion(tmp_path, single, emotion_every=0)
    assert run(mei.reply_actions(CHOICES).pick("You came!")) is not None and single.asked

    own = Model('{"expression":"anger"}')
    mei = only_her_model(tmp_path / "solo", own)
    assert run(mei.reply_actions(CHOICES).pick("You came!")).expression == "anger"


def only_her_model(path, model):
    """A companion given no background model: her own model serves it."""
    from ai_character_engine import CharacterProfile
    from ai_character_engine.companion import CharacterCompanion

    class Talking(Foreground):
        async def generate(self, messages, *, tools=None):
            return await model.generate(messages, tools=tools)

    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=Talking(),
        storage_dir=path,
        settings=CompanionSettings(
            emotion_every=0, memory_every=0, goal_every=0, reflection_every=0, summary_every=0,
            self_memory_every=0, mood_every=0, reply_check_every=0,
        ),
    )


def test_a_line_is_picked_while_she_is_still_speaking(tmp_path):
    """Her background work waits until she has spoken; a pick does not."""
    gate = asyncio.Event()
    talking = Foreground(gate=gate)
    mei = companion(tmp_path, {"actions": Model()}, llm=talking)

    async def go():
        speaking = asyncio.create_task(mei.reply("hi"))
        await talking.started.wait()
        picked = await asyncio.wait_for(mei.reply_actions(CHOICES).pick("Hello."), timeout=1.0)
        gate.set()
        await speaking
        return picked

    assert run(go()) is not None


def test_the_mood_she_is_in_gives_the_voice(tmp_path):
    mei = companion(tmp_path, {"actions": Model()})
    mei.runtime.state.apply(StatePatch(emotion="sad", mood_intensity=0.9, mood_updated_at=mei._clock()))
    assert mei.reply_actions(CHOICES).voice() == "sadness"


def test_a_closed_companion_picks_nothing(tmp_path):
    mei = companion(tmp_path, {"actions": Model()})
    run(mei.close())
    assert run(mei.reply_actions(CHOICES).pick("You came!")) is None


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError, match="actions_timeout_seconds"):
        CompanionSettings(actions_timeout_seconds=0)


def test_a_model_of_its_own_for_the_picks_comes_before_the_background_one(tmp_path):
    from ai_character_engine import CharacterProfile
    from ai_character_engine.companion import CharacterCompanion

    background, picker = Model('{"expression":"sadness"}'), Model('{"expression":"anger"}')
    mei = CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=Foreground(),
        background_llm=background,
        actions_llm=picker,
        storage_dir=tmp_path,
        settings=CompanionSettings(
            emotion_every=0, memory_every=0, goal_every=0, reflection_every=0, summary_every=0,
            self_memory_every=0, mood_every=0, reply_check_every=0,
        ),
    )
    assert run(mei.reply_actions(CHOICES).pick("You came!")).expression == "anger"
    assert background.asked == []


# --- lining the prompt up with the model server's cache ---------------------------------------


class Counting(Model):
    """A model server that counts a character as a token, plus the end of its chat template."""

    END = 9

    def __init__(self, *args, usage=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage = usage

    async def generate(self, messages, *, tools=None):
        response = await super().generate(messages, tools=tools)
        if not self.usage:
            return response
        tokens = sum(len(m.content) for m in messages) + self.END
        return LLMResponse(text=response.text, input_tokens=tokens)


def shared_part(messages):
    """What every line's request has in common: all but the mood, the line before and the line."""
    system, user = messages
    return len(system.content) + len(user.content.split("Her mood:")[0])


def test_the_fixed_part_is_padded_so_a_cache_block_ends_inside_it():
    model = Counting()
    picker = ReplyActions(
        client=model, choices=CHOICES, mood=lambda: ("sad", 0.42), timeout_seconds=6.0, cache_block=128
    )

    async def lines():
        await picker.pick("You came!")
        await asyncio.sleep(0)  # the probes run beside the first line
        await picker.calibrated
        await picker.pick("I am tired today.")
        await picker.pick("Shall we go?")

    asyncio.run(lines())
    asked = [m for m in model.asked if "Line: " in m[-1].content]
    for messages in asked[1:]:
        tokens = sum(len(m.content) for m in messages) + Counting.END
        assert tokens // 128 * 128 <= shared_part(messages) < tokens  # the block before the line is kept
    assert len(model.asked) == len(asked) + 2  # two probes, once


def test_a_server_that_does_not_count_tokens_gets_no_padding():
    model = Counting(usage=False)
    picker = ReplyActions(client=model, choices=CHOICES, mood=lambda: ("sad", 0.42), timeout_seconds=6.0, cache_block=128)

    async def lines():
        await picker.pick("You came!")
        await picker.calibrated
        await picker.pick("I am tired today.")

    asyncio.run(lines())
    last = model.asked[-1][-1].content
    assert last.startswith("Expressions:") and "\n." not in last and " . ." not in last


def test_without_a_cache_block_nothing_is_probed():
    model = Counting()
    asyncio.run(actions(model).pick("You came!"))
    assert len(model.asked) == 1


def test_the_companion_probes_once_for_the_same_avatar(tmp_path):
    model = Counting()
    mei = companion(tmp_path, {"actions": model}, actions_cache_block=128)

    async def two_replies():
        first = mei.reply_actions(CHOICES)
        await first.pick("You came!")
        await first.calibrated
        await mei.reply_actions(CHOICES).pick("Again!")

    run(two_replies())
    assert len(model.asked) == 4  # two probes, two lines


def test_the_cache_block_is_not_negative():
    with pytest.raises(ValueError, match="actions_cache_block"):
        CompanionSettings(actions_cache_block=-1)


class LongEnd(Counting):
    END = 40  # a chat template that closes with more tokens than the old margin


def test_a_long_chat_template_end_still_leaves_the_block_inside_the_fixed_part():
    model = LongEnd()
    picker = ReplyActions(
        client=model, choices=CHOICES, mood=lambda: ("sad", 0.42), timeout_seconds=6.0, cache_block=128
    )

    async def lines():
        await picker.pick("Hi.")
        await picker.calibrated
        await picker.pick("Hm.")

    asyncio.run(lines())
    messages = model.asked[-1]
    tokens = sum(len(m.content) for m in messages) + LongEnd.END
    assert tokens // 128 * 128 <= shared_part(messages)


def test_a_probe_that_fails_is_tried_again_later(monkeypatch):
    from ai_character_engine.companion import avatar_actions

    monkeypatch.setattr(avatar_actions, "PROBE_RETRY_SECONDS", 0.0)

    class Flaky(Counting):
        failures = 1

        async def generate(self, messages, *, tools=None):
            if "Line: " not in messages[-1].content and Flaky.failures:
                Flaky.failures -= 1
                raise RuntimeError("model is loading")
            return await super().generate(messages, tools=tools)

    model = Flaky()
    picker = ReplyActions(client=model, choices=CHOICES, mood=lambda: ("sad", 0.42), timeout_seconds=6.0, cache_block=128)

    async def lines():
        await picker.pick("You came!")
        await picker.calibrated  # failed
        await picker.pick("I am tired today.")
        await picker.calibrated  # tried again
        await picker.pick("Shall we go?")

    asyncio.run(lines())
    assert " . ." in model.asked[-1][-1].content


def test_a_line_skipped_by_the_host_is_still_the_line_before_the_next():
    model = Model()
    picker = actions(model)
    picker.skip("You came!")
    asyncio.run(picker.pick("I am tired today."))
    assert "Previous line: You came!" in model.asked[0][-1].content
