"""Her mood: eight words of her own, how strongly she feels it, and since when.

The engine had a mood word that never faded and had no strength. A host that
shows her face between replies needs both, and needs them to survive a
restart of the host."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

import ai_character_engine
from ai_character_engine import CharacterProfile
from ai_character_engine.companion import CHARACTER_MOODS, CharacterCompanion, CompanionSettings
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from ai_character_engine.session.serialization import state_from_dict, state_to_dict
from ai_character_engine.state.models import CharacterState, StatePatch


class Talker:
    """Her model: answers each call with the next of the given lines."""

    def __init__(self, *lines):
        self.lines = list(lines) or ["Mm."]
        self.calls = []

    def _next(self):
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        return LLMResponse(text=self._next(), model="talker")

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        text = self._next()
        yield LLMStreamChunk(text=text)
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="talker"))


def make(tmp_path, *, llm=None, clock=None, workers=None, **settings):
    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=llm or Talker(),
        background_llm=workers or {},
        storage_dir=tmp_path / "engine",
        settings=CompanionSettings(**settings),
        clock=clock,
    )


OLD_STATE = {
    "emotion": "hurt",
    "energy": 100.0,
    "trust": 61.0,
    "favorability": 55.0,
    "relationship_stage": "acquaintance",
    "custom": {},
}


def test_her_moods_are_eight_words_in_a_fixed_order():
    assert CHARACTER_MOODS == (
        "neutral", "happy", "sad", "angry", "surprised", "embarrassed", "calm", "worried",
    )


def test_the_vocabulary_is_not_part_of_the_frozen_root_api():
    assert "CHARACTER_MOODS" not in ai_character_engine.__all__


def test_a_fresh_state_has_no_mood_to_speak_of():
    state = CharacterState()
    assert (state.emotion, state.mood_intensity, state.mood_updated_at) == ("neutral", 0.0, None)


def test_neutral_has_no_intensity():
    assert CharacterState(emotion="neutral", mood_intensity=0.9).mood_intensity == 0.0


def test_a_mood_keeps_its_intensity_and_time():
    state = CharacterState()
    state.apply(StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0))
    assert (state.emotion, state.mood_intensity, state.mood_updated_at) == ("sad", 0.8, 1000.0)


def test_an_intensity_beyond_one_is_one():
    state = CharacterState()
    state.apply(StatePatch(emotion="happy", mood_intensity=3.0, mood_updated_at=1.0))
    assert state.mood_intensity == 1.0


def test_a_mood_set_without_intensity_is_of_middling_strength_as_of_now():
    state = CharacterState()
    before = time.time()
    state.apply(StatePatch(emotion="happy"))
    assert state.mood_intensity == 0.5
    assert before <= state.mood_updated_at <= time.time()


def test_a_patch_without_a_mood_leaves_it_alone():
    state = CharacterState()
    state.apply(StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0))
    state.apply(StatePatch(trust_delta=1.0))
    assert (state.emotion, state.mood_intensity, state.mood_updated_at) == ("sad", 0.8, 1000.0)


def test_restore_brings_back_the_mood():
    state = CharacterState()
    state.apply(StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0))
    saved = state.snapshot()
    state.apply(StatePatch(emotion="happy", mood_intensity=0.2, mood_updated_at=2000.0))
    state.restore(saved)
    assert (state.emotion, state.mood_intensity, state.mood_updated_at) == ("sad", 0.8, 1000.0)


def test_the_mood_survives_saving():
    state = CharacterState()
    state.apply(StatePatch(emotion="worried", mood_intensity=0.6, mood_updated_at=1234.5))
    loaded = state_from_dict(json.loads(json.dumps(state_to_dict(state.snapshot()))))
    assert (loaded.emotion, loaded.mood_intensity, loaded.mood_updated_at) == ("worried", 0.6, 1234.5)


def test_a_state_saved_before_moods_faded_reads_as_no_mood_yet():
    loaded = state_from_dict(dict(OLD_STATE))
    assert (loaded.mood_intensity, loaded.mood_updated_at) == (0.0, None)
    assert (loaded.trust, loaded.favorability) == (61.0, 55.0)


@pytest.mark.parametrize("garbage", ["high", None, float("nan"), True, [0.5]])
def test_an_unreadable_saved_mood_is_no_mood(garbage):
    loaded = state_from_dict(
        {"emotion": "sad", "mood_intensity": garbage, "mood_updated_at": "yesterday"}
    )
    assert (loaded.mood_intensity, loaded.mood_updated_at) == (0.0, None)


def test_a_state_file_from_before_moods_loads_with_her_mood_as_of_now(tmp_path):
    async def scenario():
        engine = tmp_path / "engine"
        engine.mkdir()
        (engine / "state.json").write_text(json.dumps(OLD_STATE), encoding="utf-8")
        current = make(tmp_path, clock=lambda: 5000.0)
        state = current.runtime.state
        found = (state.trust, state.mood_intensity, state.mood_updated_at)
        await current.close()
        return found

    assert asyncio.run(scenario()) == (61.0, 0.0, 5000.0)
