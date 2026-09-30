from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.compatibility import (
    ContractSurface,
    build_public_api_manifest,
    compare_public_api_manifests,
    current_contract_versions,
    load_public_api_manifest,
)
from ai_character_engine.goals import JsonlGoalStore
from ai_character_engine.long_term_cognition import JsonlLongTermCognitionStore
from ai_character_engine.memory import JsonlMemoryStore, MemoryRecord
from ai_character_engine.multi_character import (
    JsonlCharacterExchangeStore,
    JsonlSharedCognitionStore,
)
from ai_character_engine.persistence import (
    CURRENT_PERSISTENCE_SCHEMA_VERSIONS,
    LEGACY_UNVERSIONED_SCHEMA_VERSION,
    PERSISTENCE_FIXTURE_SCHEMA_VERSION,
    PERSISTENCE_SCHEMA_CONTRACT_VERSION,
    SCHEMA_VERSION_FIELD,
    PersistenceFileFormat,
    PersistenceFixtureError,
    PersistenceMigrationError,
    PersistenceMigrationPathError,
    PersistenceMigrationRegistry,
    PersistenceReplayFixture,
    PersistenceSurface,
    UnsupportedPersistenceSchemaError,
    canonical_persistence_json,
    current_persistence_schema_version,
    infer_persistence_surface,
    load_persistence_replay_fixture,
    migrate_persistence_file,
    migrate_persistence_payload,
    persistence_payload_fingerprint,
    save_persistence_replay_fixture,
    verify_persistence_fixture_directory,
    verify_persistence_replay_fixture,
)
from ai_character_engine.persistence.__main__ import main as persistence_main
from ai_character_engine.session import JsonFileRelationshipStore, JsonFileSessionStore
from ai_character_engine.world import JsonlWorldObservationStore, JsonlWorldStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "persistence" / "v0.45"
PERSISTENCE = ROOT / "src" / "ai_character_engine" / "persistence"
SEALED_V045 = ROOT / "tests/fixtures/api" / "public_api_v0.45_sealed.json"


def fixture_path(surface: PersistenceSurface) -> Path:
    matches = []
    for path in FIXTURES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["surface"] == surface.value:
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


def fixture_payload(surface: PersistenceSurface) -> dict:
    return json.loads(fixture_path(surface).read_text(encoding="utf-8"))


def legacy_payload(surface: PersistenceSurface) -> dict:
    return fixture_payload(surface)["payload"]


def expected_payload(surface: PersistenceSurface) -> dict:
    return fixture_payload(surface)["expected_payload"]


def test_persistence_contract_versions_are_explicit_and_independent():
    assert PERSISTENCE_SCHEMA_CONTRACT_VERSION == 1
    assert PERSISTENCE_FIXTURE_SCHEMA_VERSION == 1
    assert LEGACY_UNVERSIONED_SCHEMA_VERSION == 0
    assert SCHEMA_VERSION_FIELD == "_schema_version"
    assert set(CURRENT_PERSISTENCE_SCHEMA_VERSIONS) == set(PersistenceSurface)
    assert set(CURRENT_PERSISTENCE_SCHEMA_VERSIONS.values()) == {1}
    assert current_contract_versions()[ContractSurface.PERSISTENCE_SCHEMA] == 1


@pytest.mark.parametrize("surface", list(PersistenceSurface))
def test_sealed_fixture_migrates_exactly_to_the_current_schema(surface):
    fixture = load_persistence_replay_fixture(fixture_path(surface))
    assert fixture.engine_version == "0.45.0"
    assert fixture.source_schema_version == 0
    result = migrate_persistence_payload(surface, fixture.payload)
    assert result.source_version == 0
    assert result.target_version == 1
    assert result.migrated
    assert len(result.steps) == 1
    assert dict(result.payload) == dict(fixture.expected_payload)
    assert result.payload[SCHEMA_VERSION_FIELD] == 1


def test_migration_does_not_mutate_caller_payload():
    payload = legacy_payload(PersistenceSurface.MEMORY)
    original = json.loads(json.dumps(payload))
    migrate_persistence_payload(PersistenceSurface.MEMORY, payload)
    assert payload == original
    assert SCHEMA_VERSION_FIELD not in payload


def test_current_schema_is_a_noop_with_empty_step_trace():
    payload = expected_payload(PersistenceSurface.MEMORY)
    result = migrate_persistence_payload(PersistenceSurface.MEMORY, payload)
    assert not result.migrated
    assert result.steps == ()
    assert dict(result.payload) == payload


def test_future_schema_fails_closed():
    payload = expected_payload(PersistenceSurface.MEMORY)
    payload[SCHEMA_VERSION_FIELD] = 99
    with pytest.raises(UnsupportedPersistenceSchemaError, match="newer"):
        migrate_persistence_payload(PersistenceSurface.MEMORY, payload)


def test_downgrade_is_rejected():
    payload = expected_payload(PersistenceSurface.MEMORY)
    with pytest.raises(UnsupportedPersistenceSchemaError, match="downgrades"):
        migrate_persistence_payload(PersistenceSurface.MEMORY, payload, target_version=0)


def test_declared_source_version_must_match_embedded_schema():
    payload = expected_payload(PersistenceSurface.MEMORY)
    with pytest.raises(PersistenceMigrationError, match="disagrees"):
        migrate_persistence_payload(PersistenceSurface.MEMORY, payload, source_version=0)


def test_missing_required_semantic_field_is_not_guessed():
    payload = legacy_payload(PersistenceSurface.MEMORY)
    payload.pop("summary")
    with pytest.raises(PersistenceMigrationError, match="summary"):
        migrate_persistence_payload(PersistenceSurface.MEMORY, payload)


def test_mismatched_typed_record_fails_closed():
    payload = legacy_payload(PersistenceSurface.GOAL)
    payload["record_type"] = "belief"
    with pytest.raises(PersistenceMigrationError, match="expected 'goal'"):
        migrate_persistence_payload(PersistenceSurface.GOAL, payload)


def test_infer_surface_only_for_typed_records():
    assert infer_persistence_surface(legacy_payload(PersistenceSurface.GOAL)) is PersistenceSurface.GOAL
    with pytest.raises(PersistenceMigrationError, match="explicit surface"):
        infer_persistence_surface(legacy_payload(PersistenceSurface.MEMORY))


def test_canonical_fingerprint_is_key_order_independent():
    a = {"b": 2, "a": {"y": 2, "x": 1}}
    b = {"a": {"x": 1, "y": 2}, "b": 2}
    assert canonical_persistence_json(a) == canonical_persistence_json(b)
    assert persistence_payload_fingerprint(a) == persistence_payload_fingerprint(b)


def test_custom_registry_supports_explicit_multi_step_chain():
    versions = dict(CURRENT_PERSISTENCE_SCHEMA_VERSIONS)
    versions[PersistenceSurface.MEMORY] = 2
    registry = PersistenceMigrationRegistry(current_versions=versions)
    registry.register(
        PersistenceSurface.MEMORY,
        0,
        1,
        lambda payload: {**payload, "id": payload.get("id", "m"), "character_id": "c", "summary": "s", "created_at": "2026-01-01T00:00:00+00:00"},
    )
    registry.register(PersistenceSurface.MEMORY, 1, 2, lambda payload: {**payload, "marker": "v2"})
    result = registry.migrate(PersistenceSurface.MEMORY, {})
    assert result.target_version == 2
    assert [step.to_version for step in result.steps] == [1, 2]
    assert result.payload["marker"] == "v2"
    assert result.payload[SCHEMA_VERSION_FIELD] == 2


def test_custom_registry_missing_step_fails_closed():
    versions = dict(CURRENT_PERSISTENCE_SCHEMA_VERSIONS)
    versions[PersistenceSurface.MEMORY] = 2
    registry = PersistenceMigrationRegistry(current_versions=versions)
    registry.register(
        PersistenceSurface.MEMORY,
        0,
        1,
        lambda payload: {**payload, "id": "m", "character_id": "c", "summary": "s", "created_at": "2026-01-01T00:00:00+00:00"},
    )
    with pytest.raises(PersistenceMigrationPathError, match="no migration"):
        registry.migrate(PersistenceSurface.MEMORY, {})


def test_duplicate_registry_edge_is_rejected():
    registry = PersistenceMigrationRegistry()
    registry.register(PersistenceSurface.MEMORY, 0, 1, lambda payload: payload)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(PersistenceSurface.MEMORY, 0, 1, lambda payload: payload)


def test_fixture_directory_verifies_all_sealed_surfaces():
    results = verify_persistence_fixture_directory(FIXTURES)
    assert len(results) == len(PersistenceSurface) == 11
    assert all(result.passed for result in results)
    assert {result.surface for result in results} == set(PersistenceSurface)


def test_fixture_verification_detects_expected_payload_drift():
    raw = fixture_payload(PersistenceSurface.MEMORY)
    raw["expected_payload"]["summary"] = "wrong"
    fixture = PersistenceReplayFixture(
        fixture_id=raw["fixture_id"],
        engine_version=raw["engine_version"],
        surface=PersistenceSurface(raw["surface"]),
        source_schema_version=raw["source_schema_version"],
        payload=raw["payload"],
        expected_payload=raw["expected_payload"],
    )
    result = verify_persistence_replay_fixture(fixture)
    assert not result.passed
    assert result.issues == ("migrated_payload_mismatch",)


def test_fixture_save_load_roundtrip(tmp_path):
    original = load_persistence_replay_fixture(fixture_path(PersistenceSurface.GOAL))
    path = tmp_path / "fixture.json"
    save_persistence_replay_fixture(original, path)
    loaded = load_persistence_replay_fixture(path)
    assert loaded.to_dict() == original.to_dict()


def test_fixture_directory_requires_real_json_fixtures(tmp_path):
    with pytest.raises(PersistenceFixtureError, match="contains no"):
        verify_persistence_fixture_directory(tmp_path)


def test_explicit_json_migration_keeps_source_untouched(tmp_path):
    source = tmp_path / "old.json"
    output = tmp_path / "new.json"
    raw = legacy_payload(PersistenceSurface.MEMORY)
    source.write_text(json.dumps(raw), encoding="utf-8")
    before = source.read_bytes()
    count = migrate_persistence_file(
        source,
        output,
        file_format=PersistenceFileFormat.JSON,
        surface=PersistenceSurface.MEMORY,
    )
    assert count == 1
    assert source.read_bytes() == before
    assert json.loads(output.read_text())[SCHEMA_VERSION_FIELD] == 1


def test_explicit_json_array_migration(tmp_path):
    source = tmp_path / "sessions.json"
    output = tmp_path / "sessions-v1.json"
    source.write_text(json.dumps([legacy_payload(PersistenceSurface.SESSION)]), encoding="utf-8")
    count = migrate_persistence_file(
        source,
        output,
        file_format=PersistenceFileFormat.JSON_ARRAY,
        surface=PersistenceSurface.SESSION,
    )
    assert count == 1
    assert json.loads(output.read_text())[0][SCHEMA_VERSION_FIELD] == 1


def test_typed_jsonl_auto_migration(tmp_path):
    source = tmp_path / "typed.jsonl"
    output = tmp_path / "typed-v1.jsonl"
    rows = [legacy_payload(PersistenceSurface.REFLECTION), legacy_payload(PersistenceSurface.BELIEF)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    count = migrate_persistence_file(source, output, file_format="jsonl", surface=None)
    assert count == 2
    migrated = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row[SCHEMA_VERSION_FIELD] for row in migrated] == [1, 1]


def test_untyped_jsonl_auto_migration_fails(tmp_path):
    source = tmp_path / "memory.jsonl"
    output = tmp_path / "memory-v1.jsonl"
    source.write_text(json.dumps(legacy_payload(PersistenceSurface.MEMORY)) + "\n", encoding="utf-8")
    with pytest.raises(PersistenceMigrationError, match="explicit surface"):
        migrate_persistence_file(source, output, file_format="jsonl", surface=None)


def test_migration_refuses_same_input_output_path(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PersistenceMigrationError, match="different"):
        migrate_persistence_file(path, path, file_format="json", surface=PersistenceSurface.MEMORY)


def test_cli_schema_versions_is_machine_readable(capsys):
    assert persistence_main(["schema-versions"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {surface.value: 1 for surface in PersistenceSurface}


def test_cli_verify_fixtures_passes(capsys):
    assert persistence_main(["verify-fixtures", str(FIXTURES), "--fail-on-violations"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert len(payload["fixtures"]) == 11


def test_cli_migrate_jsonl_writes_current_schema(tmp_path, capsys):
    source = tmp_path / "memory.jsonl"
    output = tmp_path / "memory-v1.jsonl"
    source.write_text(json.dumps(legacy_payload(PersistenceSurface.MEMORY)) + "\n", encoding="utf-8")
    assert persistence_main([
        "migrate", str(source), str(output), "--format", "jsonl", "--surface", "memory"
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["migrated_records"] == 1
    assert json.loads(output.read_text())[SCHEMA_VERSION_FIELD] == 1


def test_jsonl_memory_store_loads_legacy_without_rewriting_then_writes_v1(tmp_path):
    path = tmp_path / "memory.jsonl"
    raw = legacy_payload(PersistenceSurface.MEMORY)
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    original = path.read_bytes()
    store = JsonlMemoryStore(path)
    assert store.list_for_character("char-a")[0].summary == raw["summary"]
    assert path.read_bytes() == original
    store.add(MemoryRecord(character_id="char-a", summary="new memory", id="mem-new"))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert SCHEMA_VERSION_FIELD not in lines[0]
    assert lines[1][SCHEMA_VERSION_FIELD] == 1


def test_long_term_cognition_store_replays_legacy_reflection_and_belief(tmp_path):
    path = tmp_path / "cognition.jsonl"
    rows = [legacy_payload(PersistenceSurface.REFLECTION), legacy_payload(PersistenceSurface.BELIEF)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = path.read_bytes()
    store = JsonlLongTermCognitionStore(path)
    assert store.list_reflections("char-a")[0].id == "ref-v045-1"
    assert store.list_beliefs("char-a")[0].claim.object == "concise"
    assert path.read_bytes() == before


def test_goal_store_replays_legacy_goal_without_disk_mutation(tmp_path):
    path = tmp_path / "goals.jsonl"
    path.write_text(json.dumps(legacy_payload(PersistenceSurface.GOAL)) + "\n", encoding="utf-8")
    before = path.read_bytes()
    store = JsonlGoalStore(path)
    goal = store.list_goals("char-a")[0]
    assert goal.objective == "Answer the pending question"
    assert goal.support_count == 1
    assert path.read_bytes() == before


def test_world_store_replays_legacy_snapshot_and_event(tmp_path):
    path = tmp_path / "world.jsonl"
    snapshot = legacy_payload(PersistenceSurface.WORLD_SNAPSHOT)
    snapshot["revision"] = 1
    snapshot["values"] = {"weather": "rain"}
    event = legacy_payload(PersistenceSurface.WORLD_EVENT)
    event["changes"] = [{"key": "weather", "kind": "set", "before_value": None, "after_value": "rain", "existed_before": False}]
    path.write_text(json.dumps(snapshot) + "\n" + json.dumps(event) + "\n", encoding="utf-8")
    before = path.read_bytes()
    store = JsonlWorldStore(path)
    assert store.snapshot().revision == 1
    assert store.snapshot().values == {"weather": "rain"}
    assert store.list_events()[0].id == "world-v045-1"
    assert path.read_bytes() == before


def test_world_observation_store_replays_legacy_record(tmp_path):
    path = tmp_path / "obs.jsonl"
    path.write_text(json.dumps(legacy_payload(PersistenceSurface.WORLD_OBSERVATION)) + "\n", encoding="utf-8")
    before = path.read_bytes()
    store = JsonlWorldObservationStore(path)
    assert store.list_for_character("char-a")[0].world_event_id == "world-v045-1"
    assert path.read_bytes() == before


def test_multi_character_stores_replay_legacy_records(tmp_path):
    shared_path = tmp_path / "shared.jsonl"
    exchange_path = tmp_path / "exchange.jsonl"
    shared_path.write_text(json.dumps(legacy_payload(PersistenceSurface.SHARED_COGNITION)) + "\n", encoding="utf-8")
    exchange_path.write_text(json.dumps(legacy_payload(PersistenceSurface.CHARACTER_EXCHANGE)) + "\n", encoding="utf-8")
    shared = JsonlSharedCognitionStore(shared_path)
    exchanges = JsonlCharacterExchangeStore(exchange_path)
    assert shared.list_for_owner("char-a")[0].content == "It is raining."
    assert exchanges.list_for_character("char-a")[0].recipient_character_id == "char-b"


def test_session_and_relationship_stores_replay_legacy_json(tmp_path):
    session_path = tmp_path / "sessions.json"
    relationship_path = tmp_path / "relationships.json"
    session_path.write_text(json.dumps([legacy_payload(PersistenceSurface.SESSION)]), encoding="utf-8")
    relationship_path.write_text(json.dumps([legacy_payload(PersistenceSurface.RELATIONSHIP)]), encoding="utf-8")
    session_before = session_path.read_bytes()
    relationship_before = relationship_path.read_bytes()
    sessions = JsonFileSessionStore(session_path)
    relationships = JsonFileRelationshipStore(relationship_path)
    assert sessions.get("session-v045-1").version == 1
    rel = relationships.get(user_id="user-a", character_id="char-a")
    assert rel is not None and rel.trust == 50.0 and rel.relationship_stage == "stranger"
    assert session_path.read_bytes() == session_before
    assert relationship_path.read_bytes() == relationship_before


def test_session_store_new_write_stamps_schema_version(tmp_path):
    from ai_character_engine.session import SessionRecord

    path = tmp_path / "sessions.json"
    store = JsonFileSessionStore(path)
    now = datetime(2026, 9, 1, tzinfo=UTC)
    store.put(SessionRecord(user_id="u", character_id="c", id="s", created_at=now, last_activity=now))
    payload = json.loads(path.read_text())
    assert payload[0][SCHEMA_VERSION_FIELD] == 1


def test_relationship_store_new_write_stamps_schema_version(tmp_path):
    from ai_character_engine.session import RelationshipSnapshot

    path = tmp_path / "relationships.json"
    store = JsonFileRelationshipStore(path)
    store.put(RelationshipSnapshot(user_id="u", character_id="c"))
    payload = json.loads(path.read_text())
    assert payload[0][SCHEMA_VERSION_FIELD] == 1


def test_persistence_package_has_no_runtime_or_authority_imports():
    forbidden = {
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
        "ai_character_engine.session",
    }
    observed: set[str] = set()
    for path in PERSISTENCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                observed.add(node.module)
            elif isinstance(node, ast.Import):
                observed.update(alias.name for alias in node.names)
    assert forbidden.isdisjoint(observed)


def test_persistence_package_has_no_product_renderer_provider_or_broker_vocabulary():
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in PERSISTENCE.glob("*.py"))
    for token in ("vrm", "live2d", "openai", "anthropic", "gemini", "redis", "kafka", "sqs"):
        assert token not in text


def test_persistence_public_api_is_exported_from_root():
    expected = {
        "PERSISTENCE_SCHEMA_CONTRACT_VERSION",
        "PERSISTENCE_FIXTURE_SCHEMA_VERSION",
        "PersistenceSurface",
        "PersistenceMigrationRegistry",
        "PersistenceMigrationResult",
        "PersistenceReplayFixture",
        "PersistenceReplayResult",
        "migrate_persistence_payload",
        "migrate_persistence_file",
        "verify_persistence_replay_fixture",
        "verify_persistence_fixture_directory",
        "persistence_payload_fingerprint",
    }
    assert expected.issubset(set(ace.__all__))
    for name in expected:
        assert hasattr(ace, name)


def test_package_and_contract_versions_are_aligned():
    from ai_character_engine.distributed import DISTRIBUTED_PROTOCOL_VERSION
    from ai_character_engine.extensions import EXTENSION_API_VERSION
    from ai_character_engine.compatibility import PUBLIC_API_CONTRACT_VERSION
    from ai_character_engine_vrm import __version__ as vrm_version

    assert (ace.__version__, vrm_version) == ("1.0.0", "1.0.0")
    assert (
        PUBLIC_API_CONTRACT_VERSION,
        EXTENSION_API_VERSION,
        DISTRIBUTED_PROTOCOL_VERSION,
        PERSISTENCE_SCHEMA_CONTRACT_VERSION,
    ) == (1, 1, 1, 1)


def test_sealed_api_upgrade_only_adds_symbols():
    baseline = load_public_api_manifest(SEALED_V045)
    sealed_v046 = load_public_api_manifest(ROOT / "tests/fixtures/api" / "public_api_v0.46_sealed.json")
    report = compare_public_api_manifests(baseline, sealed_v046)
    assert baseline.engine_version == "0.45.0"
    assert sealed_v046.engine_version == "0.46.0"
    assert report.compatible and report.breaking == ()
    assert {issue.code for issue in report.issues} == {"symbol_added"}
    assert len(report.issues) == 29
