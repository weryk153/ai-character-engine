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
