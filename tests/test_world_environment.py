from __future__ import annotations

from pathlib import Path

import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.multi_character import MultiCharacterRuntime
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.world import (
    ExplicitWorldPerceptionPolicy,
    InMemoryWorldObservationStore,
    InMemoryWorldStore,
    JsonlWorldObservationStore,
    JsonlWorldStore,
    WorldChangeKind,
    WorldEvent,
    WorldFactChange,
    WorldObservation,
    WorldPatch,
    WorldPerceptionDeniedError,
    WorldPerceptionProjection,
    WorldPerceptionProjectionError,
    WorldPerceptionScope,
    WorldRevisionConflictError,
    WorldRuntime,
    WorldStateSnapshot,
)
from tests.fakes import FakeLLMClient


def character_runtime(character_id: str) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(id=character_id, name=character_id.title(), description="test"),
        llm=FakeLLMClient(f"reply-{character_id}"),
        memory_scope_id=f"memory:{character_id}",
        cognition_scope_id=f"cognition:{character_id}",
        goal_scope_id=f"goal:{character_id}",
    )


def test_perception_scope_is_small_and_explicit():
    assert [item.value for item in WorldPerceptionScope] == ["hidden", "direct", "public"]


def test_world_patch_rejects_negative_expected_revision():
    with pytest.raises(ValueError, match="expected_revision"):
        WorldPatch(expected_revision=-1)


def test_world_patch_rejects_set_delete_overlap():
    with pytest.raises(ValueError, match="set and delete"):
        WorldPatch(set_values={"door": "open"}, delete_keys=("door",))


def test_world_patch_defensively_copies_input_values():
    nested = {"items": [1]}
    patch = WorldPatch(set_values={"inventory": nested})
    nested["items"].append(2)
    assert patch.set_values == {"inventory": {"items": [1]}}


def test_direct_event_requires_explicit_observers():
    with pytest.raises(ValueError, match="requires observer"):
        WorldEvent("door_opened", "door opened", 0, 1, perception_scope=WorldPerceptionScope.DIRECT)


def test_public_and_hidden_events_reject_observer_allowlists():
    for scope in (WorldPerceptionScope.PUBLIC, WorldPerceptionScope.HIDDEN):
        with pytest.raises(ValueError, match="must not define observers"):
            WorldEvent("x", "x", 0, 1, perception_scope=scope, observer_character_ids=("a",))


def test_event_observable_keys_must_come_from_its_own_changes():
    with pytest.raises(ValueError, match="observable_keys"):
        WorldEvent("x", "x", 0, 1, observable_keys=("secret",))


def test_world_starts_at_revision_zero_with_no_values():
    world = WorldRuntime()
    assert world.snapshot() == WorldStateSnapshot()
    assert world.events() == ()


def test_apply_event_updates_canonical_world_state_and_revision():
    world = WorldRuntime()
    event = world.apply_event(
        type="weather_changed",
        content="Rain started.",
        patch=WorldPatch(set_values={"weather.kind": "rain"}, expected_revision=0),
    )
    assert event.base_revision == 0
    assert event.revision == 1
    assert event.changed_keys == ("weather.kind",)
    assert world.snapshot().revision == 1
    assert world.snapshot().values == {"weather.kind": "rain"}


def test_delete_is_canonical_change_and_removes_key():
    world = WorldRuntime()
    world.apply_event(type="spawn", patch=WorldPatch(set_values={"object.key": {"x": 1}}))
    event = world.apply_event(type="despawn", patch=WorldPatch(delete_keys=("object.key",)))
    assert world.snapshot().values == {}
    assert event.changes[0].kind is WorldChangeKind.DELETE
    assert event.changes[0].existed_before is True
    assert event.changes[0].before_value == {"x": 1}


def test_deleting_missing_key_is_audited_but_safe():
    world = WorldRuntime()
    event = world.apply_event(type="cleanup", patch=WorldPatch(delete_keys=("missing",)))
    assert event.changes[0].existed_before is False
    assert world.snapshot().values == {}


def test_stale_expected_revision_is_rejected_before_commit():
    world = WorldRuntime()
    world.apply_event(type="one", content="one")
    with pytest.raises(WorldRevisionConflictError, match="stale"):
        world.apply_event(type="two", content="two", patch=WorldPatch(expected_revision=0))
    assert world.snapshot().revision == 1


def test_world_store_rejects_non_contiguous_direct_commit():
    store = InMemoryWorldStore()
    event = WorldEvent("x", "x", base_revision=1, revision=2)
    with pytest.raises(WorldRevisionConflictError):
        store.commit(WorldStateSnapshot(revision=2), event)


def test_canonical_event_without_state_patch_still_advances_world_revision():
    world = WorldRuntime()
    first = world.apply_event(type="bell", content="A bell rang.")
    second = world.apply_event(type="bell", content="It rang again.")
    assert (first.revision, second.revision) == (1, 2)
    assert world.snapshot().values == {}


def test_duplicate_world_event_id_is_rejected():
    world = WorldRuntime()
    world.apply_event(type="one", content="one", event_id="same")
    with pytest.raises(ValueError, match="duplicate"):
        world.apply_event(type="two", content="two", event_id="same")


def test_snapshot_is_a_defensive_copy():
    world = WorldRuntime()
    world.apply_event(type="x", patch=WorldPatch(set_values={"nested": {"items": [1]}}))
    snapshot = world.snapshot()
    snapshot.values["nested"]["items"].append(2)
    assert world.snapshot().values == {"nested": {"items": [1]}}


def test_hidden_world_event_cannot_be_observed_by_default_policy():
    world = WorldRuntime()
    event = world.apply_event(type="secret", content="hidden", perception_scope=WorldPerceptionScope.HIDDEN)
    with pytest.raises(WorldPerceptionDeniedError):
        world.observe_event(event.id, character_id="a")


def test_direct_world_event_is_visible_only_to_explicit_observer():
    world = WorldRuntime()
    event = world.apply_event(
        type="whisper",
        content="A nearby sound.",
        perception_scope=WorldPerceptionScope.DIRECT,
        observer_character_ids=("a",),
    )
    assert world.observe_event(event.id, character_id="a").character_id == "a"
    with pytest.raises(WorldPerceptionDeniedError):
        world.observe_event(event.id, character_id="b")


def test_public_means_eligible_not_automatically_delivered_or_observed():
    store = InMemoryWorldObservationStore()
    world = WorldRuntime(observation_store=store)
    event = world.apply_event(type="rain", content="It rains.", perception_scope=WorldPerceptionScope.PUBLIC)
    assert store.list_for_character("a") == []
    assert store.list_for_character("b") == []
    world.observe_event(event.id, character_id="a")
    assert len(store.list_for_character("a")) == 1
    assert store.list_for_character("b") == []


def test_observation_defaults_to_changed_facts_only():
    world = WorldRuntime()
    event = world.apply_event(
        type="room_changed",
        content="The room changed.",
        patch=WorldPatch(set_values={"room.light": "on", "room.temp": 20}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    observation = world.observe_event(event.id, character_id="a")
    assert [item.key for item in observation.changes] == ["room.light", "room.temp"]


def test_observable_keys_can_hide_part_of_one_world_event():
    world = WorldRuntime()
    event = world.apply_event(
        type="room_changed",
        content="Something changed.",
        patch=WorldPatch(set_values={"room.light": "on", "safe.code": "1234"}),
        perception_scope=WorldPerceptionScope.PUBLIC,
        observable_keys=("room.light",),
    )
    observation = world.observe_event(event.id, character_id="a")
    assert [item.key for item in observation.changes] == ["room.light"]


class NarrowPolicy:
    name = "narrow-test"

    def project(self, *, character_id, event):
        return WorldPerceptionProjection(content=f"{character_id} saw light", fact_keys=("room.light",))


class InventingPolicy:
    name = "bad-test"

    def project(self, *, character_id, event):
        return WorldPerceptionProjection(content="invent", fact_keys=("not.in.event",))


class CharacterSpecificPolicy:
    name = "character-specific-test"

    def project(self, *, character_id, event):
        if character_id == "a":
            return WorldPerceptionProjection(content="A saw the lamp.", fact_keys=("room.light",))
        if character_id == "b":
            return WorldPerceptionProjection(content="B only heard the alarm.", fact_keys=())
        return None


def test_custom_policy_may_narrow_but_not_rewrite_canonical_fact_values():
    world = WorldRuntime()
    event = world.apply_event(
        type="room_changed",
        content="change",
        patch=WorldPatch(set_values={"room.light": "on", "room.temp": 20}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    observation = world.observe_event(event.id, character_id="a", policy=NarrowPolicy())
    assert observation.content == "a saw light"
    assert observation.changes[0].after_value == "on"
    assert observation.perception_policy == "narrow-test"


def test_custom_policy_cannot_invent_world_fact_keys():
    world = WorldRuntime()
    event = world.apply_event(
        type="room_changed",
        patch=WorldPatch(set_values={"room.light": "on"}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    with pytest.raises(WorldPerceptionProjectionError, match="non-observable"):
        world.observe_event(event.id, character_id="a", policy=InventingPolicy())


def test_two_characters_can_have_different_observations_of_same_canonical_event():
    world = WorldRuntime()
    event = world.apply_event(
        type="alarm",
        content="alarm",
        patch=WorldPatch(set_values={"room.light": "red", "alarm.active": True}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    a = world.observe_event(event.id, character_id="a", policy=CharacterSpecificPolicy())
    b = world.observe_event(event.id, character_id="b", policy=CharacterSpecificPolicy())
    assert a.content != b.content
    assert [x.key for x in a.changes] == ["room.light"]
    assert b.changes == ()
    assert world.snapshot().values == {"room.light": "red", "alarm.active": True}


def test_character_event_is_non_authoritative_observation_with_world_provenance():
    world = WorldRuntime()
    event = world.apply_event(
        type="door",
        content="The door opened.",
        patch=WorldPatch(set_values={"door.main": "open"}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    observation = world.observe_event(event.id, character_id="a")
    character_event = world.character_event(observation)
    assert character_event.type == "world_observation"
    assert character_event.source == "world"
    assert character_event.payload["authoritative"] is False
    assert character_event.payload["canonical_world_provenance"] is True
    assert character_event.payload["memory_evidence_type"] == "event_observation"
    assert character_event.payload["world_event_id"] == event.id
    assert 'door.main' in character_event.content


def test_deleted_world_fact_is_rendered_as_deleted_in_character_event():
    world = WorldRuntime()
    world.apply_event(type="spawn", patch=WorldPatch(set_values={"object": 1}))
    event = world.apply_event(
        type="despawn",
        content="It disappeared.",
        patch=WorldPatch(delete_keys=("object",)),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    obs = world.observe_event(event.id, character_id="a")
    assert "<deleted>" in world.character_event(obs).content


def test_observation_creation_never_mutates_canonical_world_state():
    world = WorldRuntime()
    event = world.apply_event(
        type="x", patch=WorldPatch(set_values={"x": 1}), perception_scope=WorldPerceptionScope.PUBLIC
    )
    before = world.snapshot()
    world.observe_event(event.id, character_id="a", metadata={"note": "local"})
    assert world.snapshot() == before


@pytest.mark.asyncio
async def test_delivering_world_observation_to_character_runtime_does_not_mutate_world_truth():
    world = WorldRuntime()
    event = world.apply_event(
        type="weather", content="Rain.", patch=WorldPatch(set_values={"weather": "rain"}),
        perception_scope=WorldPerceptionScope.PUBLIC,
    )
    multi = MultiCharacterRuntime({"a": character_runtime("a")})
    observation = world.observe_event(event.id, character_id="a")
    before = world.snapshot()
    result = await multi.process_event("a", world.character_event(observation))
    assert result.event.type == "world_observation"
    assert world.snapshot() == before


@pytest.mark.asyncio
async def test_world_event_is_not_telepathically_injected_into_other_characters():
    world = WorldRuntime()
    world.apply_event(type="weather", content="Rain.", perception_scope=WorldPerceptionScope.PUBLIC)
    multi = MultiCharacterRuntime({"a": character_runtime("a"), "b": character_runtime("b")})
    assert multi.runtime_for("a").history == []
    assert multi.runtime_for("b").history == []


def test_world_observation_store_rejects_duplicate_ids():
    observation = WorldObservation("a", "event", 1, "saw it", id="same")
    store = InMemoryWorldObservationStore((observation,))
    with pytest.raises(ValueError, match="duplicate"):
        store.add(observation)


def test_world_jsonl_roundtrip_restores_state_and_event_history(tmp_path: Path):
    path = tmp_path / "world.jsonl"
    world = WorldRuntime(store=JsonlWorldStore(path))
    first = world.apply_event(
        type="weather", content="Rain.", patch=WorldPatch(set_values={"weather": "rain"}),
        perception_scope=WorldPerceptionScope.PUBLIC, metadata={"zone": "outside"},
    )
    world.apply_event(type="time", patch=WorldPatch(set_values={"time.hour": 12}))
    loaded = WorldRuntime(store=JsonlWorldStore(path))
    assert loaded.snapshot() == WorldStateSnapshot(revision=2, values={"weather": "rain", "time.hour": 12})
    assert loaded.event(first.id).metadata == {"zone": "outside"}
    assert len(loaded.events()) == 2


def test_world_observation_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    store = JsonlWorldObservationStore(path)
    observation = WorldObservation(
        character_id="a",
        world_event_id="e1",
        world_revision=2,
        content="saw rain",
        changes=(WorldFactChange("weather", WorldChangeKind.SET, after_value="rain"),),
        perception_policy="test",
        metadata={"sensor": "window"},
    )
    store.add(observation)
    loaded = JsonlWorldObservationStore(path).get(observation.id)
    assert loaded == observation


def test_jsonl_world_store_rejects_snapshot_revision_without_history(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"record_type":"world_snapshot","revision":1,"values":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="event history"):
        JsonlWorldStore(path)


def test_explicit_default_policy_is_provider_and_host_neutral():
    policy = ExplicitWorldPerceptionPolicy()
    event = WorldEvent(
        "x", "x", 0, 1, perception_scope=WorldPerceptionScope.PUBLIC,
        changes=(WorldFactChange("x", WorldChangeKind.SET, after_value=1),),
    )
    projection = policy.project(character_id="any-character", event=event)
    assert projection is not None and projection.fact_keys == ("x",)


def test_world_package_does_not_import_character_cognition_authority_or_multi_character():
    root = Path(__file__).resolve().parents[1] / "src" / "ai_character_engine" / "world"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for forbidden in (
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.multi_character",
        "ai_character_engine.avatar",
        "ai_character_engine.voice",
    ):
        assert forbidden not in source


def test_world_runtime_has_no_auto_delivery_api():
    assert not hasattr(WorldRuntime, "broadcast")
    assert not hasattr(WorldRuntime, "auto_deliver")
    assert not hasattr(WorldRuntime, "sync_beliefs")


def test_public_exports_and_version():
    import ai_character_engine as ace

    assert ace.__version__ == "1.0.0"
    assert ace.WorldRuntime is WorldRuntime
    assert ace.WorldPerceptionScope.PUBLIC.value == "public"


def test_docs_describe_world_perception_boundary():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "World truth is not character knowledge" in guide
    assert "authoritative=false" in guide


def test_offline_example_runs_without_provider():
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, str(root / "tests/scenarios/world_environment.py")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "world revision: 1" in proc.stdout
    assert "alice observed: world_observation" in proc.stdout
    assert "bob denied: True" in proc.stdout
    assert "authoritative: False" in proc.stdout
