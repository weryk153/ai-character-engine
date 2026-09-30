from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.compatibility import (
    PUBLIC_API_CONTRACT_VERSION,
    ApiStability,
    ApiSymbolKind,
    CompatibilitySeverity,
    ContractSurface,
    ContractVersionRequirement,
    MigrationRule,
    ParameterContract,
    PublicApiManifest,
    PublicApiSymbol,
    build_public_api_manifest,
    check_contract_requirements,
    compare_public_api_manifests,
    current_contract_versions,
    load_public_api_manifest,
    public_export_names,
    save_public_api_manifest,
)
from ai_character_engine.compatibility.__main__ import main as compatibility_main
from ai_character_engine.distributed import DISTRIBUTED_PROTOCOL_VERSION
from ai_character_engine.extensions import EXTENSION_API_VERSION


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/api" / "public_api_v1_candidate.json"
COMPAT = ROOT / "src" / "ai_character_engine" / "compatibility"


def parameter(name: str, kind: str = "positional_or_keyword", required: bool = True) -> ParameterContract:
    return ParameterContract(name=name, kind=kind, required=required)


def symbol(
    name: str = "Thing",
    *,
    params: tuple[ParameterContract, ...] = (),
    stability: ApiStability = ApiStability.CANDIDATE,
    deprecated_since: str | None = None,
    replacement: str | None = None,
    kind: ApiSymbolKind = ApiSymbolKind.FUNCTION,
) -> PublicApiSymbol:
    return PublicApiSymbol(
        module="pkg",
        name=name,
        import_path=f"pkg.{name}",
        kind=kind,
        parameters=params,
        stability=stability,
        deprecated_since=deprecated_since,
        replacement=replacement,
    )


def manifest(version: str, *symbols: PublicApiSymbol, contract: int = 1) -> PublicApiManifest:
    return PublicApiManifest(version, contract, tuple(symbols))


def test_public_api_contract_version_is_independent_from_engine_semver():
    assert PUBLIC_API_CONTRACT_VERSION == 1
    assert ace.__version__ == "1.0.0"


def test_existing_versioned_extension_and_distributed_contracts_do_not_follow_engine_semver():
    assert EXTENSION_API_VERSION == 1
    assert DISTRIBUTED_PROTOCOL_VERSION == 1


def test_current_contract_matrix_exposes_only_explicit_versioned_surfaces():
    versions = current_contract_versions()
    assert versions == {
        ContractSurface.PUBLIC_API: 1,
        ContractSurface.EXTENSION_API: 1,
        ContractSurface.DISTRIBUTED_PROTOCOL: 1,
        ContractSurface.PERSISTENCE_SCHEMA: 1,
    }


def test_contract_requirement_check_accepts_supported_versions():
    result = check_contract_requirements(
        ContractVersionRequirement(ContractSurface.PUBLIC_API, 1),
        ContractVersionRequirement(ContractSurface.EXTENSION_API, 1),
        ContractVersionRequirement(ContractSurface.DISTRIBUTED_PROTOCOL, 1),
        ContractVersionRequirement(ContractSurface.PERSISTENCE_SCHEMA, 1),
    )
    assert result.compatible
    assert result.mismatches == ()
    assert result.engine_version == "1.0.0"


def test_contract_requirement_check_reports_mismatch_without_mutating_any_runtime():
    result = check_contract_requirements(ContractVersionRequirement(ContractSurface.PUBLIC_API, 2))
    assert not result.compatible
    assert result.mismatches == ("public_api: required 2, supported 1",)


def test_contract_requirement_rejects_non_positive_version():
    with pytest.raises(ValueError, match="version"):
        ContractVersionRequirement(ContractSurface.PUBLIC_API, 0)


def test_public_api_manifest_is_deterministic_and_sorted():
    first = build_public_api_manifest()
    second = build_public_api_manifest()
    assert first.to_dict() == second.to_dict()
    keys = [item.key for item in first.symbols]
    assert keys == sorted(keys)


def test_public_api_manifest_tracks_documented_import_path_not_implementation_module():
    current = build_public_api_manifest()
    item = next(item for item in current.symbols if item.name == "CharacterRuntime")
    assert item.import_path == "ai_character_engine.CharacterRuntime"


def test_public_api_manifest_covers_every_root___all___export_exactly_once():
    current = build_public_api_manifest()
    names = tuple(item.name for item in current.symbols)
    assert names == public_export_names()
    assert len(names) == len(set(names)) == len(ace.__all__)


def test_public_api_manifest_contains_key_long_lived_engine_surfaces():
    names = {item.name for item in build_public_api_manifest().symbols}
    assert {
        "CharacterRuntime",
        "MemoryManager",
        "LongTermCognitionManager",
        "GoalManager",
        "WorldRuntime",
        "MultiCharacterRuntime",
        "PluginManager",
        "DistributedWorker",
        "ProductionHardeningRuntime",
    } <= names


def test_baseline_manifest_exists_and_is_v043_candidate_contract():
    baseline = load_public_api_manifest(BASELINE)
    assert baseline.engine_version == "1.0.0"
    assert baseline.contract_version == 1
    assert baseline.metadata["status"] == "v1 candidate"
    assert len(baseline.symbols) == len(ace.__all__)


def test_baseline_manifest_matches_current_runtime_without_breaking_changes():
    baseline = load_public_api_manifest(BASELINE)
    current = build_public_api_manifest()
    report = compare_public_api_manifests(baseline, current)
    assert report.compatible
    assert report.breaking == ()
    assert report.issues == ()


def test_manifest_json_roundtrip_is_lossless(tmp_path):
    original = build_public_api_manifest(metadata={"x": "y"})
    path = tmp_path / "api.json"
    save_public_api_manifest(original, path)
    assert load_public_api_manifest(path).to_dict() == original.to_dict()


def test_manifest_rejects_duplicate_symbols():
    one = symbol()
    with pytest.raises(ValueError, match="duplicate"):
        PublicApiManifest("0.43.0", 1, (one, one))


def test_manifest_rejects_private_symbol_names():
    with pytest.raises(ValueError, match="private"):
        symbol("_internal")


def test_parameter_contract_rejects_empty_name_and_kind():
    with pytest.raises(ValueError):
        ParameterContract("", "positional_or_keyword", True)
    with pytest.raises(ValueError):
        ParameterContract("x", "", True)


def test_removed_candidate_symbol_is_breaking():
    before = manifest("0.43.0", symbol())
    after = manifest("0.44.0")
    report = compare_public_api_manifests(before, after)
    assert not report.compatible
    assert report.breaking[0].code == "symbol_removed"


def test_migration_rule_does_not_make_undeclared_candidate_removal_safe():
    before = manifest("0.43.0", symbol())
    after = manifest("0.44.0")
    rule = MigrationRule("pkg:Thing", "pkg:Replacement", "0.43.0", "0.44.0")
    assert not compare_public_api_manifests(before, after, migrations=(rule,)).compatible


def test_deprecated_symbol_can_be_removed_only_after_declared_window():
    old = symbol(stability=ApiStability.DEPRECATED, deprecated_since="0.43.0")
    before = manifest("0.43.0", old)
    after = manifest("0.44.0")
    rule = MigrationRule("pkg:Thing", "pkg:Replacement", "0.43.0", "0.44.0")
    report = compare_public_api_manifests(before, after, migrations=(rule,))
    assert report.compatible
    assert report.warnings[0].code == "symbol_removed"


def test_deprecated_symbol_cannot_be_removed_before_remove_in():
    old = symbol(stability=ApiStability.DEPRECATED, deprecated_since="0.43.0")
    rule = MigrationRule("pkg:Thing", "pkg:Replacement", "0.43.0", "1.0.0")
    report = compare_public_api_manifests(manifest("0.43.0", old), manifest("0.44.0"), migrations=(rule,))
    assert not report.compatible


def test_deprecated_removal_requires_matching_deprecated_since():
    old = symbol(stability=ApiStability.DEPRECATED, deprecated_since="0.43.0")
    rule = MigrationRule("pkg:Thing", "pkg:Replacement", "0.42.0", "0.44.0")
    report = compare_public_api_manifests(manifest("0.43.0", old), manifest("0.44.0"), migrations=(rule,))
    assert not report.compatible


def test_duplicate_migration_rules_are_rejected():
    before = manifest("0.43.0", symbol(stability=ApiStability.DEPRECATED, deprecated_since="0.43.0"))
    after = manifest("0.44.0")
    rule = MigrationRule("pkg:Thing", None, "0.43.0", "0.44.0")
    with pytest.raises(ValueError, match="duplicate migration"):
        compare_public_api_manifests(before, after, migrations=(rule, rule))


def test_migration_rule_requires_namespaced_symbol_paths():
    with pytest.raises(ValueError, match="old_path"):
        MigrationRule("Thing", None, "0.43.0")
    with pytest.raises(ValueError, match="replacement"):
        MigrationRule("pkg:Thing", "Replacement", "0.43.0")


def test_symbol_kind_change_is_breaking():
    before = manifest("0.43.0", symbol(kind=ApiSymbolKind.FUNCTION))
    after = manifest("0.44.0", symbol(kind=ApiSymbolKind.CLASS))
    assert any(issue.code == "symbol_kind_changed" for issue in compare_public_api_manifests(before, after).breaking)


def test_removing_existing_parameter_is_breaking():
    before = manifest("0.43.0", symbol(params=(parameter("a"), parameter("b", required=False))))
    after = manifest("0.44.0", symbol(params=(parameter("a"),)))
    report = compare_public_api_manifests(before, after)
    assert any("removed parameter 'b'" in issue.message for issue in report.breaking)


def test_adding_required_parameter_is_breaking():
    before = manifest("0.43.0", symbol(params=(parameter("a"),)))
    after = manifest("0.44.0", symbol(params=(parameter("a"), parameter("b"))))
    report = compare_public_api_manifests(before, after)
    assert any("added required parameter 'b'" in issue.message for issue in report.breaking)


def test_optional_parameter_becoming_required_is_breaking():
    before = manifest("0.43.0", symbol(params=(parameter("a", required=False),)))
    after = manifest("0.44.0", symbol(params=(parameter("a", required=True),)))
    assert not compare_public_api_manifests(before, after).compatible


def test_parameter_kind_change_is_breaking():
    before = manifest("0.43.0", symbol(params=(parameter("a", "positional_or_keyword"),)))
    after = manifest("0.44.0", symbol(params=(parameter("a", "keyword_only"),)))
    assert not compare_public_api_manifests(before, after).compatible


def test_inserting_optional_positional_before_existing_parameter_is_breaking():
    before = manifest("0.43.0", symbol(params=(parameter("a"), parameter("b", required=False))))
    after = manifest(
        "0.44.0",
        symbol(params=(parameter("a"), parameter("x", required=False), parameter("b", required=False))),
    )
    report = compare_public_api_manifests(before, after)
    assert any("inserted or reordered positional" in issue.message for issue in report.breaking)


def test_appending_optional_positional_parameter_is_compatible():
    before = manifest("0.43.0", symbol(params=(parameter("a"),)))
    after = manifest("0.44.0", symbol(params=(parameter("a"), parameter("b", required=False))))
    assert compare_public_api_manifests(before, after).compatible


def test_adding_optional_keyword_only_parameter_is_compatible():
    before = manifest("0.43.0", symbol(params=(parameter("a"),)))
    after = manifest(
        "0.44.0",
        symbol(params=(parameter("a"), parameter("trace", "keyword_only", required=False))),
    )
    assert compare_public_api_manifests(before, after).compatible


def test_new_public_symbol_is_info_not_breaking():
    before = manifest("0.43.0")
    after = manifest("0.44.0", symbol())
    report = compare_public_api_manifests(before, after)
    assert report.compatible
    assert report.issues[0].severity is CompatibilitySeverity.INFO
    assert report.issues[0].code == "symbol_added"


def test_candidate_to_deprecated_is_warning_not_immediate_removal():
    before = manifest("0.43.0", symbol())
    after = manifest(
        "0.44.0",
        symbol(stability=ApiStability.DEPRECATED, deprecated_since="0.44.0", replacement="pkg:Replacement"),
    )
    report = compare_public_api_manifests(before, after)
    assert report.compatible
    assert report.warnings[0].code == "symbol_deprecated"


def test_stable_symbol_cannot_be_downgraded():
    before = manifest("1.0.0", symbol(stability=ApiStability.STABLE))
    after = manifest("1.1.0", symbol(stability=ApiStability.CANDIDATE))
    assert any(issue.code == "stability_downgraded" for issue in compare_public_api_manifests(before, after).breaking)


def test_contract_version_change_is_a_single_manifest_break():
    before = manifest("0.43.0", symbol(), contract=1)
    after = manifest("0.44.0", symbol(), contract=2)
    report = compare_public_api_manifests(before, after)
    assert not report.compatible
    assert [item.code for item in report.issues] == ["contract_version_changed"]


def test_cli_write_generates_current_manifest(tmp_path):
    path = tmp_path / "current.json"
    assert compatibility_main(["write", str(path)]) == 0
    loaded = load_public_api_manifest(path)
    assert loaded.engine_version == "1.0.0"
    assert len(loaded.symbols) == len(ace.__all__)


def test_cli_check_baseline_passes_with_fail_on_breaking(capsys):
    assert compatibility_main(["check", str(BASELINE), "--fail-on-breaking"]) == 0
    output = capsys.readouterr().out
    assert "compatible" in output
    assert "no public API differences" in output


def test_cli_check_can_emit_json(capsys):
    assert compatibility_main(["check", str(BASELINE), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["compatible"] is True
    assert payload["issues"] == []


def test_cli_fail_on_breaking_returns_two(tmp_path):
    previous = manifest("0.43.0", symbol("Old"))
    path = tmp_path / "old.json"
    save_public_api_manifest(previous, path)
    assert compatibility_main(["check", str(path), "--fail-on-breaking"]) == 2


def test_compatibility_package_has_no_cognition_or_world_authority_imports():
    forbidden = {
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
    }
    offenders = []
    for path in COMPAT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == prefix or alias.name.startswith(prefix + ".") for prefix in forbidden):
                        offenders.append((path.name, alias.name))
            if module and any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                offenders.append((path.name, module))
    assert offenders == []


def test_compatibility_layer_does_not_contain_renderer_host_or_product_tokens():
    forbidden = ("vrm", "live2d", "openai", "anthropic", "gemini", "redis", "kafka", "sqs")
    offenders = []
    for path in COMPAT.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                offenders.append((path.name, token))
    assert offenders == []


def test_root_public_exports_have_no_duplicates_and_no_private_names():
    assert len(ace.__all__) == len(set(ace.__all__))
    assert all(not name.startswith("_") for name in ace.__all__)


def test_all_declared_root_public_exports_are_resolvable():
    missing = [name for name in ace.__all__ if not hasattr(ace, name)]
    assert missing == []


def test_package_versions_are_aligned_for_stabilization_release():
    from ai_character_engine_vrm import __version__ as vrm_version

    assert (ace.__version__, vrm_version) == ("1.0.0", "1.0.0")
