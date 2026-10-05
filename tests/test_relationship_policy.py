"""How the character's own state moves with the way the user treats it.

The engine records what it observes about the user (emotion, valence, stance)
but leaves the character's own state to the host. RelationshipStatePolicy is a
ready-made answer for hosts that do not want to invent one.
"""
from __future__ import annotations

import pytest

from ai_character_engine import CharacterEvent, CharacterProfile, CharacterRuntime
from ai_character_engine.state.models import CharacterState
from ai_character_engine.state.mood import CHARACTER_MOODS, MOOD_TURN_KEY
from ai_character_engine.state.relationship import (
    APPLIED_OBSERVATION_KEY,
    RelationshipRules,
    RelationshipStatePolicy,
    relationship_patch,
)
from tests.fakes import FakeLLMClient


def state(*, trust=50.0, favorability=50.0, stage="stranger", observed=None, applied=None, judged=None):
    custom = {}
    if observed is not None:
        custom["observed_user_emotion"] = observed
    if applied is not None:
        custom[APPLIED_OBSERVATION_KEY] = applied
    if judged is not None:
        custom[MOOD_TURN_KEY] = judged
    return CharacterState(
        trust=trust, favorability=favorability, relationship_stage=stage, custom=custom
    ).snapshot()


def observed(**overrides):
    base = {
        "emotion": "glad",
        "intensity": 1.0,
        "confidence": 1.0,
        "valence": 0.0,
        "stance": 0.0,
        "proposal_id": "p1",
    }
    base.update(overrides)
    return base


def test_every_turn_builds_a_little_trust():
    patch = relationship_patch(state(), count_turn=True)

    assert patch.trust_delta == pytest.approx(0.3)
    assert patch.favorability_delta == 0
    assert patch.emotion is None


def test_nothing_to_do_between_turns_without_a_new_observation():
    assert relationship_patch(state(), count_turn=False) is None


def test_warmth_toward_the_character_raises_favorability_and_trust():
    patch = relationship_patch(state(observed=observed(stance=1.0, valence=0.9)), count_turn=True)

    assert patch.favorability_delta == pytest.approx(4.0)
    assert patch.trust_delta == pytest.approx(0.3 + 2.0)
    assert patch.emotion == "happy"
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p1"}


def test_hostility_costs_more_trust_than_warmth_earns():
    patch = relationship_patch(state(observed=observed(stance=-1.0, valence=-0.8)), count_turn=True)

    assert patch.favorability_delta == pytest.approx(-4.0)
    assert patch.trust_delta == pytest.approx(0.3 - 3.0)
    assert patch.emotion == "sad"


def test_someone_hurting_makes_the_character_concerned_even_when_they_are_warm():
    """Measured: "my boss shouted at me all day, I just want to talk to you" is
    stance 1, valence -0.4. They came in trust; the character should worry,
    not be pleased."""
    patch = relationship_patch(
        state(observed=observed(stance=1.0, valence=-0.4, intensity=0.8, confidence=0.9)),
        count_turn=True,
    )

    assert patch.emotion == "worried"
    assert patch.favorability_delta > 0
    assert patch.trust_delta > 0.3


def test_a_bad_day_that_is_not_about_the_character_earns_trust_not_dislike():
    patch = relationship_patch(
        state(observed=observed(stance=0.0, valence=-0.6, intensity=0.8)), count_turn=True
    )

    assert patch.favorability_delta == 0
    assert patch.trust_delta == pytest.approx(0.3 + 0.8)
    assert patch.emotion == "worried"


def test_the_same_observation_is_applied_only_once():
    patch = relationship_patch(
        state(observed=observed(stance=1.0), applied="p1"), count_turn=True
    )

    assert patch.favorability_delta == 0
    assert patch.trust_delta == pytest.approx(0.3)
    assert patch.custom_updates == {}


def test_an_observation_without_scores_is_consumed_without_changing_anything():
    old_worker = {"emotion": "frustrated", "intensity": 0.8, "confidence": 0.9, "proposal_id": "p2"}

    patch = relationship_patch(state(observed=old_worker), count_turn=False)

    assert patch.favorability_delta == 0 and patch.trust_delta == 0 and patch.emotion is None
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p2"}


@pytest.mark.parametrize("garbage", ["high", None, float("nan"), True, [1]])
def test_scores_that_cannot_be_read_count_as_missing(garbage):
    patch = relationship_patch(
        state(observed=observed(stance=garbage, valence=garbage)), count_turn=True
    )

    assert patch.favorability_delta == 0
    assert patch.emotion is None


@pytest.mark.parametrize(
    ("trust", "favorability", "expected"),
    [(50.0, 50.0, None), (60.0, 56.0, "acquaintance"), (70.0, 70.0, "friend"), (90.0, 80.0, "close")],
)
def test_the_stage_follows_the_average_of_trust_and_favorability(trust, favorability, expected):
    patch = relationship_patch(state(trust=trust, favorability=favorability), count_turn=True)

    assert patch.relationship_stage == expected


def test_the_stage_does_not_drop_on_a_small_dip():
    patch = relationship_patch(state(trust=66.0, favorability=66.0, stage="friend"), count_turn=True)

    assert patch.relationship_stage is None


def test_the_stage_drops_once_the_dip_is_clear():
    patch = relationship_patch(state(trust=60.0, favorability=62.0, stage="friend"), count_turn=True)

    assert patch.relationship_stage == "acquaintance"


def test_a_host_can_tune_the_numbers():
    rules = RelationshipRules(trust_per_turn=1.0, favorability_per_warmth=10.0)

    patch = relationship_patch(state(observed=observed(stance=1.0)), count_turn=True, rules=rules)

    assert patch.favorability_delta == pytest.approx(10.0)
    assert patch.trust_delta == pytest.approx(1.0 + 2.0)


async def test_as_a_state_policy_it_moves_the_character_on_every_user_turn():
    runtime = CharacterRuntime(
        character=CharacterProfile("c", "C", "test"),
        llm=FakeLLMClient("reply"),
        state_policy=RelationshipStatePolicy(),
    )

    await runtime.run_turn("hello")
    await runtime.process_event(CharacterEvent(type="clock_tick", source="host", content="noon"))

    assert runtime.state.trust == pytest.approx(50.3)


def test_her_mood_takes_its_strength_from_the_observation():
    patch = relationship_patch(
        state(observed=observed(stance=1.0, valence=0.9, intensity=0.7)),
        count_turn=False,
        now=1000.0,
    )
    assert (patch.emotion, patch.mood_intensity, patch.mood_updated_at) == ("happy", 0.7, 1000.0)


def test_an_observation_without_intensity_makes_a_middling_mood():
    without = observed(stance=-1.0, valence=-0.8)
    del without["intensity"]
    patch = relationship_patch(state(observed=without), count_turn=False, now=1.0)
    assert (patch.emotion, patch.mood_intensity) == ("sad", 0.5)


@pytest.mark.parametrize(("stance", "valence"), [(-1.0, -1.0), (1.0, -0.6), (0.0, -0.6), (1.0, 1.0)])
def test_the_rules_speak_her_vocabulary(stance, valence):
    patch = relationship_patch(
        state(observed=observed(stance=stance, valence=valence)), count_turn=False, now=1.0
    )
    assert patch.emotion in CHARACTER_MOODS
    assert patch.emotion not in {"hurt", "concerned"}


def test_a_reading_of_her_mood_from_the_same_turn_is_not_replaced_by_the_rules():
    """The mood worker read both sides of the turn; the observation read the
    user alone. What the user did still counts for trust."""
    patch = relationship_patch(
        state(observed=observed(stance=1.0, valence=0.9, turn_ended_at=100.0), judged=100.0),
        count_turn=False,
        now=1.0,
    )
    assert patch.emotion is None
    assert patch.mood_intensity is None
    assert patch.favorability_delta == pytest.approx(4.0)
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p1"}


def test_a_newer_turn_moves_her_mood_again():
    patch = relationship_patch(
        state(observed=observed(stance=1.0, valence=0.9, turn_ended_at=101.0), judged=100.0),
        count_turn=False,
        now=1.0,
    )
    assert patch.emotion == "happy"
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p1", MOOD_TURN_KEY: 101.0}


def test_the_policy_dates_a_mood_by_its_clock():
    policy = RelationshipStatePolicy()
    policy.clock = lambda: 4242.0
    from ai_character_engine.events.models import CharacterEvent

    patch = policy.on_event(
        CharacterEvent.user_message("hi"),
        state(observed=observed(stance=1.0, valence=0.9)),
    )
    assert patch.mood_updated_at == 4242.0


@pytest.mark.parametrize(("stance", "valence"), [(0.0, 0.0), (0.2, 0.9), (-0.2, -0.2), (0.1, 0.5)])
def test_an_unremarkable_observation_leaves_her_mood_as_it_was(stance, valence):
    """Only a notable observation moves her mood. Setting calm on every other
    turn flipped her resting face between neutral and calm between readings."""
    before = CharacterState(
        emotion="embarrassed",
        mood_intensity=0.8,
        mood_updated_at=500.0,
        custom={
            "observed_user_emotion": observed(
                stance=stance, valence=valence, intensity=0.6, turn_ended_at=101.0
            ),
            MOOD_TURN_KEY: 100.0,
        },
    )

    patch = relationship_patch(before.snapshot(), count_turn=True, now=1000.0)

    assert (patch.emotion, patch.mood_intensity, patch.mood_updated_at) == (None, None, None)
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p1"}
    assert patch.favorability_delta == pytest.approx(4.0 * stance * 0.6)
    before.apply(patch)
    assert (before.emotion, before.mood_intensity, before.mood_updated_at) == (
        "embarrassed",
        0.8,
        500.0,
    )
    assert before.custom[MOOD_TURN_KEY] == 100.0


def feeling(mood, strength, updated_at, **observation):
    """A state in which she feels ``mood`` and a new observation came in."""
    return CharacterState(
        emotion=mood,
        mood_intensity=strength,
        mood_updated_at=updated_at,
        custom={"observed_user_emotion": observed(turn_ended_at=101.0, **observation)},
    ).snapshot()


def test_the_rules_leave_a_stronger_mood_of_another_kind():
    patch = relationship_patch(
        feeling("sad", 0.9, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        count_turn=False,
        now=1000.0,
    )
    assert (patch.emotion, patch.mood_intensity, patch.mood_updated_at) == (None, None, None)
    # Not her mood's turn either: a reading of this turn may still set it.
    assert patch.custom_updates == {APPLIED_OBSERVATION_KEY: "p1"}
    assert patch.favorability_delta > 0


def test_the_rules_replace_a_mood_that_is_no_stronger():
    patch = relationship_patch(
        feeling("sad", 0.5, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        count_turn=False,
        now=2000.0,
    )
    assert (patch.emotion, patch.mood_intensity, patch.mood_updated_at) == ("happy", 0.6, 2000.0)
    assert patch.custom_updates[MOOD_TURN_KEY] == 101.0


def test_the_rules_compare_with_her_mood_as_it_has_faded():
    patch = relationship_patch(
        feeling("sad", 0.9, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        count_turn=False,
        now=1300.0,  # 0.45 left
    )
    assert (patch.emotion, patch.mood_intensity) == ("happy", 0.6)


def test_the_rules_fade_her_mood_by_the_half_life_they_are_given():
    patch = relationship_patch(
        feeling("sad", 0.9, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        count_turn=False,
        now=1060.0,
        half_life_seconds=60.0,
    )
    assert patch.emotion == "happy"


def test_the_same_mood_from_the_rules_is_the_stronger_of_the_two():
    patch = relationship_patch(
        feeling("happy", 0.9, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        count_turn=False,
        now=1000.0,
    )
    assert (patch.emotion, patch.mood_intensity, patch.mood_updated_at) == ("happy", 0.9, 1000.0)


def test_the_policy_fades_her_mood_by_its_settings():
    from ai_character_engine.events.models import CharacterEvent

    def mood_set(half_life):
        policy = RelationshipStatePolicy()
        policy.clock = lambda: 1060.0
        if half_life is not None:
            policy.mood_half_life_seconds = half_life
        return policy.on_event(
            CharacterEvent.user_message("hi"),
            feeling("sad", 0.9, 1000.0, stance=1.0, valence=0.9, intensity=0.6),
        ).emotion

    assert mood_set(None) is None  # five-minute half-life: still 0.78 sad
    assert mood_set(60.0) == "happy"  # 0.45 sad left
