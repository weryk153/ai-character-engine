"""How the character's own state moves with the way the user treats it.

The engine records what it observes about the user (emotion, valence, stance)
but leaves the character's own state to the host. RelationshipStatePolicy is a
ready-made answer for hosts that do not want to invent one.
"""
from __future__ import annotations

import pytest

from ai_character_engine import CharacterEvent, CharacterProfile, CharacterRuntime
from ai_character_engine.state.models import CharacterState
from ai_character_engine.state.relationship import (
    APPLIED_OBSERVATION_KEY,
    RelationshipRules,
    RelationshipStatePolicy,
    relationship_patch,
)
from tests.fakes import FakeLLMClient


def state(*, trust=50.0, favorability=50.0, stage="stranger", observed=None, applied=None):
    custom = {}
    if observed is not None:
        custom["observed_user_emotion"] = observed
    if applied is not None:
        custom[APPLIED_OBSERVATION_KEY] = applied
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
    assert patch.emotion == "hurt"


def test_someone_hurting_makes_the_character_concerned_even_when_they_are_warm():
    """Measured: "my boss shouted at me all day, I just want to talk to you" is
    stance 1, valence -0.4. They came in trust; the character should worry,
    not be pleased."""
    patch = relationship_patch(
        state(observed=observed(stance=1.0, valence=-0.4, intensity=0.8, confidence=0.9)),
        count_turn=True,
    )

    assert patch.emotion == "concerned"
    assert patch.favorability_delta > 0
    assert patch.trust_delta > 0.3


def test_a_bad_day_that_is_not_about_the_character_earns_trust_not_dislike():
    patch = relationship_patch(
        state(observed=observed(stance=0.0, valence=-0.6, intensity=0.8)), count_turn=True
    )

    assert patch.favorability_delta == 0
    assert patch.trust_delta == pytest.approx(0.3 + 0.8)
    assert patch.emotion == "concerned"


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
