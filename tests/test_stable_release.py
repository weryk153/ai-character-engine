from __future__ import annotations

from ai_character_engine._version import VERSION
import ast
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

import ai_character_engine as ace
from ai_character_engine.compatibility import ApiStability, compare_public_api_manifests
from ai_character_engine.compatibility.__main__ import main as compatibility_main
from ai_character_engine.compatibility.manifest import build_public_api_manifest, load_public_api_manifest
from ai_character_engine.persistence import verify_persistence_fixture_directory
from ai_character_engine.release import check_release_readiness, default_release_matrix
from ai_character_engine.release_candidate import (
    RC_EVIDENCE_SCHEMA_VERSION,
    MatrixEvidence,
    MatrixEvidenceStatus,
    ReleaseCandidateGate,
    ReleaseCandidateGateStatus,
    ReleaseCandidateReport,
    evaluate_release_candidate,
    load_matrix_evidence,
    save_matrix_evidence,
    validate_matrix_evidence,
)
from ai_character_engine.release_candidate.__main__ import main as rc_main
from ai_character_engine.release_candidate.provenance import candidate_sha256

ROOT = Path(__file__).resolve().parents[1]
STABLE = ROOT / "docs/public_api_v1_stable.json"
SEALED_V047 = ROOT / "tests/fixtures/api/public_api_v0.47_sealed.json"
FIXTURES = ROOT / "tests/fixtures/persistence/v0.45"
RC_PACKAGE = ROOT / "src/ai_character_engine/release_candidate"


def evidence(platform: str, python_version: str, *, engine_version: str = VERSION, status: MatrixEvidenceStatus = MatrixEvidenceStatus.PASS, all_checks: bool = True, source: str = "test") -> MatrixEvidence:
    return MatrixEvidence(
        platform=platform,
        python_version=python_version,
        engine_version=engine_version,
        status=status,
        pytest_passed=all_checks,
        compileall_passed=all_checks,
        matrix_self_check_passed=all_checks,
        test_summary="matrix row passed" if all_checks else "matrix row failed",
        source=source,
        metadata={"runner": platform, "candidate_sha256": candidate_sha256(ROOT), "source_unchanged": True,
                  "checks": [{"name": n, "command": ["test-fixture"], "exit_code": 0, "output": "",
                              "output_sha256": hashlib.sha256(b"").hexdigest()} for n in ("pytest", "compileall", "matrix")]},
    )


def all_matrix_evidence() -> tuple[MatrixEvidence, ...]:
    return tuple(evidence(target.platform, target.python_version) for target in default_release_matrix().targets)


def test_rc_evidence_schema_version_is_explicit():
    assert RC_EVIDENCE_SCHEMA_VERSION == 1


@pytest.mark.parametrize("platform", ["linux", "LINUX", "macos", "windows"])
def test_matrix_evidence_normalises_platform(platform):
    item = evidence(platform, "3.11")
    assert item.platform == platform.lower()


@pytest.mark.parametrize("platform", ["android", "ios", "freebsd", ""])
def test_matrix_evidence_rejects_unsupported_platform(platform):
    with pytest.raises(ValueError, match="platform"):
        evidence(platform, "3.11")


@pytest.mark.parametrize("python_version", ["3", "3.11.1", "x.y", ""])
def test_matrix_evidence_rejects_invalid_python_version(python_version):
    with pytest.raises(ValueError, match="major.minor"):
        evidence("linux", python_version)


def test_matrix_evidence_rejects_empty_engine_version():
    with pytest.raises(ValueError, match="engine_version"):
        evidence("linux", "3.11", engine_version="")


def test_matrix_evidence_rejects_empty_source():
    with pytest.raises(ValueError, match="source"):
        MatrixEvidence("linux", "3.11", VERSION, MatrixEvidenceStatus.PASS, True, True, True, source="")


def test_matrix_evidence_rejects_unknown_schema():
    payload = evidence("linux", "3.11").to_dict()
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        MatrixEvidence.from_dict(payload)


def test_matrix_evidence_roundtrip_and_metadata_is_immutable():
    original = evidence("macos", "3.12")
    loaded = MatrixEvidence.from_dict(original.to_dict())
    assert loaded == original
    with pytest.raises(TypeError):
        loaded.metadata["x"] = 1  # type: ignore[index]


def test_matrix_evidence_passed_requires_all_three_checks():
    assert evidence("linux", "3.13").passed
    assert not evidence("linux", "3.13", all_checks=False).passed


def test_matrix_evidence_fail_status_never_passes():
    item = evidence("linux", "3.13", status=MatrixEvidenceStatus.FAIL)
    assert not item.passed


def test_matrix_evidence_file_roundtrip(tmp_path):
    path = tmp_path / "evidence.json"
    save_matrix_evidence(evidence("windows", "3.13"), path)
    assert load_matrix_evidence(path) == evidence("windows", "3.13")


def test_matrix_evidence_file_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_matrix_evidence(path)


def test_empty_matrix_evidence_blocks_all_nine_rows():
    gates = validate_matrix_evidence(default_release_matrix(), (), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert len(gates) == 9
    assert all(gate.status is ReleaseCandidateGateStatus.BLOCKED for gate in gates)


def test_all_matrix_evidence_passes_all_nine_rows():
    gates = validate_matrix_evidence(default_release_matrix(), all_matrix_evidence(), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert len(gates) == 9
    assert all(gate.status is ReleaseCandidateGateStatus.PASS for gate in gates)


def test_duplicate_matrix_evidence_fails_that_row():
    item = evidence("linux", "3.11")
    gates = validate_matrix_evidence(default_release_matrix(), (item, item), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert next(g for g in gates if g.name == "matrix.linux.py3.11").status is ReleaseCandidateGateStatus.FAIL


def test_unexpected_matrix_target_fails_explicitly():
    gates = validate_matrix_evidence(default_release_matrix(), (evidence("linux", "3.10"),), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert any(g.name == "matrix.unexpected.linux.py3.10" and g.status is ReleaseCandidateGateStatus.FAIL for g in gates)


def test_matrix_version_mismatch_is_failure():
    gates = validate_matrix_evidence(default_release_matrix(), (evidence("linux", "3.11", engine_version="0.47.0"),), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert next(g for g in gates if g.name == "matrix.linux.py3.11").status is ReleaseCandidateGateStatus.FAIL


def test_matrix_failed_status_is_failure():
    gates = validate_matrix_evidence(default_release_matrix(), (evidence("linux", "3.11", status=MatrixEvidenceStatus.FAIL),), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert next(g for g in gates if g.name == "matrix.linux.py3.11").status is ReleaseCandidateGateStatus.FAIL


def test_matrix_partial_check_failure_is_failure():
    gates = validate_matrix_evidence(default_release_matrix(), (evidence("linux", "3.11", all_checks=False),), engine_version=VERSION, candidate_sha256=candidate_sha256(ROOT))
    assert next(g for g in gates if g.name == "matrix.linux.py3.11").status is ReleaseCandidateGateStatus.FAIL


def test_release_candidate_report_ready_only_when_every_gate_passes():
    passing = ReleaseCandidateGate("x", ReleaseCandidateGateStatus.PASS, "ok")
    blocked = ReleaseCandidateGate("y", ReleaseCandidateGateStatus.BLOCKED, "wait")
    failing = ReleaseCandidateGate("z", ReleaseCandidateGateStatus.FAIL, "bad")
    assert ReleaseCandidateReport(VERSION, (passing,)).ready_for_v1
    assert not ReleaseCandidateReport(VERSION, (passing, blocked)).ready_for_v1
    assert not ReleaseCandidateReport(VERSION, (passing, failing)).ready_for_v1


def test_release_candidate_report_blocker_and_failure_views():
    gates = (
        ReleaseCandidateGate("pass", ReleaseCandidateGateStatus.PASS, "ok"),
        ReleaseCandidateGate("blocked", ReleaseCandidateGateStatus.BLOCKED, "wait"),
        ReleaseCandidateGate("fail", ReleaseCandidateGateStatus.FAIL, "bad"),
    )
    report = ReleaseCandidateReport(VERSION, gates)
    assert [item.name for item in report.blockers] == ["blocked", "fail"]
    assert [item.name for item in report.failures] == ["fail"]
    assert report.to_dict()["ready_for_v1"] is False


def test_stable_manifest_is_v1_and_all_symbols_are_stable():
    manifest = load_public_api_manifest(STABLE)
    assert manifest.engine_version == VERSION
    assert len(manifest.symbols) == 374
    assert all(item.stability is ApiStability.STABLE for item in manifest.symbols)


def test_stable_manifest_exactly_matches_current_root_api():
    stable = load_public_api_manifest(STABLE)
    current = build_public_api_manifest(stability=ApiStability.STABLE, engine_version=VERSION)
    report = compare_public_api_manifests(stable, current)
    assert report.compatible
    assert report.issues == ()


def test_last_sealed_api_to_stable_has_no_breaking_change():
    report = compare_public_api_manifests(load_public_api_manifest(SEALED_V047), load_public_api_manifest(STABLE))
    assert report.compatible
    assert not report.breaking
    assert not [issue for issue in report.issues if issue.code == "symbol_added"]


def test_persistence_replay_is_11_of_11():
    results = verify_persistence_fixture_directory(FIXTURES)
    assert len(results) == 11
    assert all(item.passed for item in results)


def test_current_rc_evaluation_accepts_license_and_blocks_missing_evidence():
    report = evaluate_release_candidate(ROOT)
    assert not report.ready_for_v1
    by_name = {gate.name: gate for gate in report.gates}
    assert by_name["release.license"].status is ReleaseCandidateGateStatus.PASS
    assert by_name["release.artifacts"].status is ReleaseCandidateGateStatus.BLOCKED
    assert by_name["api.stable_manifest"].status is ReleaseCandidateGateStatus.PASS
    assert by_name["persistence.cross_version_replay"].status is ReleaseCandidateGateStatus.PASS
    assert sum(g.status is ReleaseCandidateGateStatus.BLOCKED and g.name.startswith("matrix.") for g in report.gates) == 9


def test_one_local_matrix_row_turns_only_that_row_green():
    report = evaluate_release_candidate(ROOT, matrix_evidence=(evidence("linux", "3.13", source="local"),))
    by_name = {gate.name: gate for gate in report.gates}
    assert by_name["matrix.linux.py3.13"].status is ReleaseCandidateGateStatus.PASS
    assert by_name["matrix.macos.py3.13"].status is ReleaseCandidateGateStatus.BLOCKED


def test_all_matrix_rows_still_do_not_override_license_blocker(unlicensed_project):
    identity = candidate_sha256(unlicensed_project)
    rows = tuple(replace(row, metadata={**row.metadata, "candidate_sha256": identity})
                 for row in all_matrix_evidence())
    report = evaluate_release_candidate(unlicensed_project, matrix_evidence=rows)
    by_name = {gate.name: gate for gate in report.gates}
    assert all(by_name[f"matrix.{t.platform}.py{t.python_version}"].status is ReleaseCandidateGateStatus.PASS for t in default_release_matrix().targets)
    assert by_name["release.license"].status is ReleaseCandidateGateStatus.BLOCKED
    assert not report.ready_for_v1


def test_release_strict_v1_requires_stable_manifest_and_license(unlicensed_project):
    readiness = check_release_readiness(unlicensed_project, strict_v1=True)
    by_name = {check.name: check for check in readiness.checks}
    assert by_name["file.docs/public_api_v1_stable.json"].status.value == "pass"
    assert by_name["license"].status.value == "fail"


def test_compatibility_cli_can_check_stable_manifest():
    assert compatibility_main(["check", str(STABLE), "--stability", "stable", "--fail-on-breaking"]) == 0


def test_compatibility_cli_can_write_stable_manifest(tmp_path):
    output = tmp_path / "stable.json"
    assert compatibility_main(["write", str(output), "--stability", "stable"]) == 0
    manifest = load_public_api_manifest(output)
    assert manifest.engine_version == VERSION
    assert all(item.stability is ApiStability.STABLE for item in manifest.symbols)


def test_rc_cli_evidence_writes_versioned_record(tmp_path, monkeypatch):
    from ai_character_engine.release_candidate import __main__ as cli
    from ai_character_engine.release.matrix import current_release_target
    target = current_release_target()
    monkeypatch.setattr(cli, "collect_matrix", lambda *a, **k: evidence(target.platform, target.python_version))
    output = tmp_path / "row.json"
    assert rc_main(["evidence", "--output", str(output)]) == 0
    item = load_matrix_evidence(output)
    assert item.engine_version == VERSION
    assert item.passed


def test_rc_cli_check_reports_blockers_without_failing_by_default(tmp_path):
    output = tmp_path / "report.json"
    assert rc_main(["check", "--project-root", str(ROOT), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready_for_v1"] is False


def test_rc_cli_fail_on_blocker_returns_2():
    assert rc_main(["check", "--project-root", str(ROOT), "--fail-on-blocker"]) == 2


def test_rc_cli_evidence_dir_collects_records(tmp_path):
    save_matrix_evidence(evidence("linux", "3.13"), tmp_path / "one.json")
    output = tmp_path / "report.json"
    assert rc_main(["check", "--project-root", str(ROOT), "--evidence-dir", str(tmp_path), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["matrix_evidence"]) == 1


def test_release_candidate_module_does_not_import_runtime_authority():
    forbidden = {
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
    }
    imported: set[str] = set()
    for path in RC_PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden) for name in imported)


def test_release_candidate_tooling_is_not_exported_into_frozen_root_api():
    assert "evaluate_release_candidate" not in ace.__all__
    assert not hasattr(ace, "evaluate_release_candidate")


def test_versions_and_contracts_remain_synchronized():
    from ai_character_engine_vrm import __version__ as vrm_version

    assert (ace.__version__, vrm_version) == (VERSION, VERSION)
    assert (
        ace.PUBLIC_API_CONTRACT_VERSION,
        ace.EXTENSION_API_VERSION,
        ace.DISTRIBUTED_PROTOCOL_VERSION,
        ace.PERSISTENCE_SCHEMA_CONTRACT_VERSION,
    ) == (1, 1, 1, 1)


def test_pyproject_installs_release_candidate_console_script():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ai-character-engine-rc = "ai_character_engine.release_candidate.__main__:main"' in text


def test_ci_workflow_uploads_each_matrix_evidence_row():
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "ai_character_engine.release_candidate evidence" in text
    assert "actions/upload-artifact@v4" in text
    assert "${{ runner.os }}" in text
    assert "${{ matrix.python-version }}" in text


def test_release_workflow_downloads_matrix_evidence_and_writes_rc_report():
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "actions/download-artifact@v4" in text
    assert "--evidence-dir rc-evidence" in text
    assert "v1-release-candidate-report.json" in text


def test_release_workflow_only_enforces_blocker_gate_for_v1_tags():
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "startsWith(github.ref, 'refs/tags/v1.')" in text
    assert "--fail-on-blocker" in text


def test_release_documentation_requires_matching_evidence_and_license():
    text = (ROOT / "docs/release.md").read_text(encoding="utf-8")
    assert "--fail-on-blocker" in text
    assert "LICENSE" in text
    assert "9" in text


def test_stable_manifest_metadata_marks_v1_stable():
    manifest = load_public_api_manifest(STABLE)
    assert manifest.metadata["frozen_in"] == "0.48.0"
    assert manifest.metadata["status"] == "v1-stable"


@pytest.mark.parametrize("relative", [".", "packages/renderer-vrm"])
def test_owner_selected_apache_license_is_in_all_package_roots(relative):
    package = ROOT / relative
    license_bytes = (package / "LICENSE").read_bytes()
    assert license_bytes == (ROOT / "LICENSE").read_bytes()
    assert b"Apache License" in license_bytes
    assert b"Version 2.0, January 2004" in license_bytes
    metadata = tomllib.loads((package / "pyproject.toml").read_text())
    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]


def test_rc_report_json_is_machine_readable():
    report = evaluate_release_candidate(ROOT, matrix_evidence=(evidence("linux", "3.13"),))
    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert payload["engine_version"] == VERSION
    assert isinstance(payload["gates"], list)
    assert isinstance(payload["matrix_evidence"], list)


def test_the_version_gate_holds_for_this_release():
    by_name = {gate.name: gate for gate in evaluate_release_candidate(ROOT).gates}
    assert by_name["version.runtime"].status is ReleaseCandidateGateStatus.PASS
    assert by_name["api.stable_manifest"].status is ReleaseCandidateGateStatus.PASS
