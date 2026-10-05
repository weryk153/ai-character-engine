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
from ai_character_engine.avatar import AvatarRuntime, EmotionExpressionPolicy, ExpressionScheduler
from ai_character_engine.companion import (
    CHARACTER_MOODS,
    CharacterCompanion,
    CompanionSettings,
    CompanionSnapshot,
)
from ai_character_engine.context.builder import ContextBuilder, is_turn_context
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from ai_character_engine.session.serialization import state_from_dict, state_to_dict
from ai_character_engine.state.mood import MOOD_SYNONYMS, blend_mood, effective_mood
from ai_character_engine.state.models import CharacterState, StatePatch
from ai_character_engine.voice import AudioChunk, AudioFormat
from tests.fakes import system_context

CHARACTER = CharacterProfile(id="mei", name="Mei", description="A researcher.")


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


def fades(*, mood="sad", intensity=0.8, updated_at=1000.0, now):
    return effective_mood(
        mood, intensity, updated_at, now=now, half_life_seconds=300.0, floor=0.15
    )


@pytest.mark.parametrize(("elapsed", "expected"), [(0, 0.8), (300, 0.4), (600, 0.2)])
def test_her_mood_halves_every_half_life(elapsed, expected):
    mood, strength = fades(now=1000.0 + elapsed)
    assert mood == "sad"
    assert strength == pytest.approx(expected)


def test_below_the_floor_she_is_neutral_again():
    assert fades(now=1900.0) == ("neutral", 0.0)  # 0.8 / 8 = 0.1


def test_neutral_does_not_fade_into_anything():
    assert fades(mood="neutral", intensity=1.0, now=1000.0) == ("neutral", 0.0)


def test_a_time_ahead_of_now_does_not_make_her_mood_stronger():
    assert fades(mood="happy", intensity=0.6, updated_at=2000.0, now=1000.0) == ("happy", 0.6)


def test_a_mood_of_unknown_time_is_as_of_now():
    assert fades(mood="happy", intensity=0.6, updated_at=None, now=5000.0) == ("happy", 0.6)


@pytest.mark.parametrize("half_life", [0.0, -60.0, float("nan"), float("inf")])
def test_a_half_life_that_is_no_length_of_time_means_no_fading(half_life):
    """CompanionSettings refuses these, but a builder's or a background
    runtime's attribute can still be set to one."""
    assert effective_mood(
        "sad", 0.8, 1000.0, now=1900.0, half_life_seconds=half_life, floor=0.15
    ) == ("sad", 0.8)


def test_a_builder_set_to_a_zero_half_life_still_builds():
    builder = fading_builder([1900.0])
    builder.mood_half_life_seconds = 0.0
    assert "- emotion: sad" in note(built(builder, SAD))


def test_how_her_mood_fades_is_a_setting():
    settings = CompanionSettings()
    assert (settings.mood_half_life_seconds, settings.mood_floor) == (300.0, 0.15)
    with pytest.raises(ValueError, match="mood_half_life_seconds"):
        CompanionSettings(mood_half_life_seconds=0)
    with pytest.raises(ValueError, match="mood_floor"):
        CompanionSettings(mood_floor=1.5)


def fading_builder(now, **options):
    builder = ContextBuilder(**options)
    builder.mood_half_life_seconds = 300.0
    builder.mood_floor = 0.15
    builder.clock = lambda: now[0]
    return builder


def built(builder, state):
    return builder.build_for_event(
        character=CHARACTER, history=[], event=CharacterEvent.user_message("hi"), state=state
    )


def note(messages):
    return "\n".join(m.content for m in messages if is_turn_context(m))


SAD = CharacterState(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0)


def test_the_note_gives_her_mood_as_it_stands_now():
    now = [1000.0]
    builder = fading_builder(now)
    fresh = note(built(builder, SAD))
    now[0] = 1900.0
    faded = note(built(builder, SAD))
    assert "- emotion: sad" in fresh
    assert "- emotion: neutral" in faded


def test_her_mood_stands_in_the_note_of_the_turn_never_in_the_system_prompt():
    """A change of mood adds a line to the conversation; in the system prompt
    it would make the server read the whole conversation again."""
    messages = built(fading_builder([1000.0]), SAD)
    assert "- emotion:" not in messages[0].content
    assert "- emotion: sad" in note(messages)


def test_the_turn_placement_reads_her_mood_faded_too():
    messages = built(fading_builder([1900.0], context_placement="turn"), SAD)
    assert "- emotion: neutral" in note(messages)


def test_a_builder_not_told_how_moods_fade_fades_her_mood_like_every_other_reader():
    """A bare CharacterRuntime has no companion to set the builder: its
    "- emotion:" line must fade as the live face does, or tone and face
    disagree."""
    from ai_character_engine.state.mood import DEFAULT_MOOD_FLOOR, DEFAULT_MOOD_HALF_LIFE_SECONDS

    builder = ContextBuilder()
    assert (builder.mood_half_life_seconds, builder.mood_floor) == (
        DEFAULT_MOOD_HALF_LIFE_SECONDS,
        DEFAULT_MOOD_FLOOR,
    )
    builder.clock = lambda: 1000.0
    assert "- emotion: sad" in note(built(builder, SAD))
    builder.clock = lambda: 1900.0
    assert "- emotion: neutral" in note(built(builder, SAD))


def test_the_mood_defaults_are_named_once():
    from ai_character_engine.state.mood import DEFAULT_MOOD_FLOOR, DEFAULT_MOOD_HALF_LIFE_SECONDS

    assert (DEFAULT_MOOD_HALF_LIFE_SECONDS, DEFAULT_MOOD_FLOOR) == (300.0, 0.15)
    settings = CompanionSettings()
    assert (settings.mood_half_life_seconds, settings.mood_floor) == (
        DEFAULT_MOOD_HALF_LIFE_SECONDS,
        DEFAULT_MOOD_FLOOR,
    )


def test_the_snapshot_says_how_her_mood_fades(tmp_path):
    async def scenario():
        now = [1000.0]
        current = make(tmp_path, clock=lambda: now[0])
        current.runtime.state.apply(
            StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0)
        )
        first = current.snapshot()
        now[0] = 1300.0
        later = current.snapshot()
        now[0] = 1900.0
        gone = current.snapshot()
        await current.close()
        return first, later, gone

    first, later, gone = asyncio.run(scenario())
    assert (first.emotion, first.mood_intensity, first.mood_updated_at) == ("sad", 0.8, 1000.0)
    assert first.mood_half_life_seconds == 300.0
    # The intensity as it was set: a host fades it itself between snapshots.
    assert (later.emotion, later.mood_intensity) == ("sad", 0.8)
    assert (gone.emotion, gone.mood_intensity) == ("neutral", 0.0)


def test_a_snapshot_still_builds_from_positions():
    snapshot = CompanionSnapshot("calm", 50.0, 50.0, "stranger")
    assert (snapshot.mood_intensity, snapshot.mood_updated_at, snapshot.mood_half_life_seconds) == (
        0.0,
        None,
        300.0,
    )


def test_the_snapshot_gives_the_floor_below_which_her_mood_is_gone(tmp_path):
    from ai_character_engine.state.mood import DEFAULT_MOOD_FLOOR

    assert CompanionSnapshot("calm", 50.0, 50.0, "stranger").mood_floor == DEFAULT_MOOD_FLOOR
    assert list(CompanionSnapshot.__dataclass_fields__)[-1] == "mood_floor"

    async def scenario():
        current = make(tmp_path, mood_floor=0.3)
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    assert asyncio.run(scenario()).mood_floor == 0.3


def test_a_mood_saved_with_a_time_still_to_come_is_as_of_loading(tmp_path):
    """Another machine's clock, or milliseconds read as seconds: left as it
    was, her mood would not fade until that time came."""

    async def scenario():
        engine = tmp_path / "engine"
        engine.mkdir()
        saved = {**OLD_STATE, "emotion": "sad", "mood_intensity": 0.8,
                 "mood_updated_at": 1000.0 * 1000}
        (engine / "state.json").write_text(json.dumps(saved), encoding="utf-8")
        now = [1000.0]
        current = make(tmp_path, clock=lambda: now[0])
        loaded = current.snapshot()
        now[0] = 1900.0
        later = current.snapshot()
        await current.close()
        return loaded, later

    loaded, later = asyncio.run(scenario())
    assert (loaded.emotion, loaded.mood_updated_at) == ("sad", 1000.0)
    assert later.emotion == "neutral"


def test_a_state_file_from_before_moods_reads_as_neutral(tmp_path):
    async def scenario():
        engine = tmp_path / "engine"
        engine.mkdir()
        (engine / "state.json").write_text(json.dumps(OLD_STATE), encoding="utf-8")
        current = make(tmp_path, clock=lambda: 5000.0)
        snapshot = current.snapshot()
        await current.close()
        return snapshot

    snapshot = asyncio.run(scenario())
    assert (snapshot.emotion, snapshot.mood_intensity, snapshot.trust) == ("neutral", 0.0, 61.0)


def test_her_reply_hears_her_mood_as_it_stands(tmp_path):
    async def scenario():
        now = [1000.0]
        llm = Talker("The kettle is on.", "Rain again, of all things.")
        current = make(tmp_path, llm=llm, clock=lambda: now[0])
        current.runtime.state.apply(
            StatePatch(emotion="sad", mood_intensity=0.8, mood_updated_at=1000.0)
        )
        await current.reply("hello", conversation_id="a")
        first = system_context(llm.calls[-1])
        now[0] = 1900.0
        await current.reply("still there?", conversation_id="a")
        later = system_context(llm.calls[-1])
        await current.close()
        return first, later

    first, later = asyncio.run(scenario())
    assert "- emotion: sad" in first
    assert "- emotion: neutral" not in first
    assert "- emotion: neutral" in later


@pytest.mark.parametrize(
    ("mood", "expression"),
    [
        ("happy", "happy"),
        ("sad", "sad"),
        ("angry", "angry"),
        ("surprised", "surprised"),
        ("embarrassed", "happy"),
        ("calm", "relaxed"),
        ("worried", "sad"),
    ],
)
def test_each_of_her_moods_has_a_face(mood, expression):
    cue = EmotionExpressionPolicy().cue(mood, start_ms=0, duration_ms=100)
    assert cue is not None and cue.expression == expression


def test_neutral_makes_no_face():
    assert EmotionExpressionPolicy().cue("neutral", start_ms=0, duration_ms=100) is None


@pytest.mark.parametrize("old", ["hurt", "concerned"])
def test_the_moods_the_rules_used_to_name_still_have_a_face(old):
    assert EmotionExpressionPolicy().cue(old, start_ms=0, duration_ms=100).expression == "sad"


def test_a_faint_mood_makes_a_faint_face():
    cue = EmotionExpressionPolicy().cue("sad", start_ms=0, duration_ms=100, intensity=0.5)
    assert cue.weight == pytest.approx(0.325)


def test_a_mood_with_no_strength_makes_no_face():
    assert EmotionExpressionPolicy().cue("sad", start_ms=0, duration_ms=100, intensity=0.0) is None


def test_the_scheduler_and_the_avatar_pass_on_how_strong_her_mood_is():
    cues = ExpressionScheduler().resolve(
        start_ms=0, duration_ms=50, emotion="happy", emotion_intensity=0.5
    )
    assert cues[0].weight == pytest.approx(0.325)
    avatar = AvatarRuntime()
    avatar.begin_turn("turn")
    bundle = avatar.feed_audio(
        text="あ",
        chunk=AudioChunk(b"\x00\x00" * 800, AudioFormat()),
        segment_sequence=0,
        chunk_index=0,
        duration_ms=50,
        emotion="sad",
        emotion_intensity=0.5,
    )
    assert bundle.expressions[0].weight == pytest.approx(0.325)


# --- inertia: a new reading moves her mood only when it is at least as strong ---


@pytest.mark.parametrize(
    ("current", "reading", "expected"),
    [
        # A neutral reading leaves her mood to fade by itself.
        (("sad", 0.8), ("neutral", 0.0), None),
        (("neutral", 0.0), ("neutral", 0.0), None),
        # The same mood again: as strong as the stronger of the two, as of now.
        (("sad", 0.4), ("sad", 0.7), ("sad", 0.7)),
        (("sad", 0.7), ("sad", 0.4), ("sad", 0.7)),
        # Another mood takes over only when it is at least as strong.
        (("sad", 0.8), ("worried", 0.5), None),
        (("sad", 0.5), ("worried", 0.5), ("worried", 0.5)),
        (("sad", 0.5), ("happy", 0.9), ("happy", 0.9)),
        # From neutral, any mood that is felt at all.
        (("neutral", 0.0), ("happy", 0.2), ("happy", 0.2)),
        (("neutral", 0.0), ("happy", 0.0), None),
    ],
)
def test_a_new_reading_blends_into_her_mood(current, reading, expected):
    assert blend_mood(*current, *reading) == expected


def test_a_faded_mood_gives_way_to_a_weaker_reading_than_it_started_as():
    """Blended with her mood as it stands now: 0.8 sad five minutes ago is
    0.4 now, and a 0.5 worried reading takes over."""
    faded = effective_mood("sad", 0.8, 1000.0, now=1300.0)
    assert faded == ("sad", pytest.approx(0.4))
    assert blend_mood(*faded, "worried", 0.5) == ("worried", 0.5)
    assert blend_mood(*faded, "worried", 0.3) is None


def test_the_same_mood_raises_a_faded_one_back_up():
    faded = effective_mood("sad", 0.8, 1000.0, now=1300.0)
    assert blend_mood(*faded, "sad", 0.3) == ("sad", pytest.approx(0.4))


def test_a_mood_faded_below_the_floor_is_neutral_and_gives_way_to_anything():
    faded = effective_mood("sad", 0.8, 1000.0, now=1900.0)
    assert blend_mood(*faded, "calm", 0.16) == ("calm", 0.16)


# --- synonyms: near words the model answers with stand for one of the eight ---


@pytest.mark.parametrize(
    ("word", "mood"),
    [
        ("joyful", "happy"), ("excited", "happy"), ("amused", "happy"),
        ("relieved", "calm"), ("at ease", "calm"), ("content", "calm"),
        ("upset", "sad"), ("lonely", "sad"), ("down", "sad"),
        ("annoyed", "angry"), ("frustrated", "angry"), ("furious", "angry"),
        ("anxious", "worried"), ("scared", "worried"), ("nervous", "worried"),
        ("shy", "embarrassed"), ("flustered", "embarrassed"), ("awkward", "embarrassed"),
        ("shocked", "surprised"), ("amazed", "surprised"),
    ],
)
def test_near_words_stand_for_one_of_her_moods(word, mood):
    assert MOOD_SYNONYMS[word] == mood


def test_every_synonym_stands_for_a_mood_of_hers_and_is_not_one_itself():
    assert set(MOOD_SYNONYMS.values()) <= set(CHARACTER_MOODS) - {"neutral"}
    assert not set(MOOD_SYNONYMS) & set(CHARACTER_MOODS)
    assert all(word == word.strip().lower() for word in MOOD_SYNONYMS)
