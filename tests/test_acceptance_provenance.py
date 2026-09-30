from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from ai_character_engine.release import default_release_matrix
from ai_character_engine.release_candidate import MatrixEvidence, MatrixEvidenceStatus, evaluate_release_candidate, validate_matrix_evidence
from ai_character_engine.release_candidate import collector
from ai_character_engine.release_candidate.__main__ import main
from ai_character_engine.release_candidate.provenance import candidate_sha256, FINAL_CHECKS

ROOT = Path(__file__).resolve().parents[1]


def receipt(name, code=0):
    output = "fixture output"
    return {"name": name, "command": ["fixture"], "exit_code": code,
            "output": output, "output_sha256": hashlib.sha256(output.encode()).hexdigest()}


def row():
    return MatrixEvidence("linux", "3.13", "1.0.0", MatrixEvidenceStatus.PASS, True, True, True,
                          metadata={"candidate_sha256": candidate_sha256(ROOT), "source_unchanged": True,
                                    "checks": [receipt(n) for n in ("pytest", "compileall", "matrix")]})


def validate(item):
    return {g.name: g.status.value for g in validate_matrix_evidence(
        default_release_matrix(), [item], engine_version="1.0.0", candidate_sha256=candidate_sha256(ROOT))}["matrix.linux.py3.13"]


@pytest.mark.parametrize("field", ["pytest_passed", "compileall_passed", "matrix_self_check_passed"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_non_boolean_checks_cannot_be_coerced_to_pass(field, value):
    payload = row().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="booleans"):
        MatrixEvidence.from_dict(payload)


@pytest.mark.parametrize("change", ["stale", "missing", "failed", "tampered", "duplicate", "changed_during_run"])
def test_matrix_rejects_untraceable_or_stale_evidence(change):
    item = row()
    data = dict(item.metadata)
    if change == "stale":
        data["candidate_sha256"] = "0" * 64
    elif change == "missing":
        data.pop("candidate_sha256")
    elif change == "failed":
        data["checks"][0]["exit_code"] = 1
    elif change == "tampered":
        data["checks"][0]["output"] = "changed"
    elif change == "duplicate":
        data["checks"][1] = data["checks"][0]
    else:
        data["source_unchanged"] = False
    assert validate(replace(item, metadata=data)) == "fail"


def test_candidate_hash_is_path_independent_and_covers_docs_tests_workflows_license(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    (left / "pyproject.toml").write_text("baseline")
    for name in ("src/core.py", "tests/test_core.py", ".github/workflows/ci.yml", "docs/gate.md"):
        file = left / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("original")
    shutil.copytree(left, right)
    original = candidate_sha256(left)
    assert original == candidate_sha256(right)
    for name in ("src/core.py", "tests/test_core.py", ".github/workflows/ci.yml", "docs/gate.md", "LICENSE"):
        file = right / name
        before = file.read_bytes() if file.exists() else None
        file.write_text("changed")
        assert candidate_sha256(right) != original
        if before is None:
            file.unlink()
        else:
            file.write_bytes(before)
    cache = right / "src/__pycache__/core.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    assert candidate_sha256(right) == original


def final_record(name):
    return {"schema_version": 1, "gate": name, "engine_version": "1.0.0",
            "candidate_sha256": candidate_sha256(ROOT), "source_unchanged": True,
            "status": "pass", "checks": [receipt(name)]}


def final_status(records):
    return {g.name: g.status.value for g in evaluate_release_candidate(ROOT, acceptance_evidence=records).gates}


def test_missing_final_acceptance_blocks_even_with_validation_md_pass_claims():
    gates = final_status([])
    assert all(gates[f"acceptance.{name}"] == "blocked" for name in FINAL_CHECKS)


@pytest.mark.parametrize("name", ["soak", "performance", "lock", "architecture"])
def test_matching_final_receipt_passes_read_only_and_deterministically(name):
    before = candidate_sha256(ROOT)
    one = evaluate_release_candidate(ROOT, acceptance_evidence=[final_record(name)]).to_dict()
    two = evaluate_release_candidate(ROOT, acceptance_evidence=[final_record(name)]).to_dict()
    assert one == two
    assert next(g for g in one["gates"] if g["name"] == f"acceptance.{name}")["status"] == "pass"
    assert candidate_sha256(ROOT) == before


@pytest.mark.parametrize("field,value", [("candidate_sha256", "0" * 64), ("engine_version", "0.47.0"),
                                          ("schema_version", 2), ("source_unchanged", False), ("status", "fail")])
def test_final_receipts_fail_closed(field, value):
    record = final_record("soak")
    record[field] = value
    assert final_status([record])["acceptance.soak"] == "fail"


def test_final_failed_check_and_tampered_output_are_failures():
    record = final_record("performance")
    record["checks"][0]["exit_code"] = 1
    assert final_status([record])["acceptance.performance"] == "fail"
    record["checks"][0]["exit_code"] = 0
    record["checks"][0]["output"] = "tampered"
    assert final_status([record])["acceptance.performance"] == "fail"


def test_duplicate_final_records_and_missing_artifact_bindings_fail():
    record = final_record("soak")
    assert final_status([record, record])["acceptance.soak"] == "fail"
    for name in ("packaging", "fresh_install"):
        assert final_status([final_record(name)])[f"acceptance.{name}"] == "fail"


def test_matrix_collector_runs_all_checks_and_retains_real_failure(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("candidate")
    calls = []
    def run(name, command, root, **kwargs):
        calls.append((name, command))
        return receipt(name, 1 if name == "pytest" else 0)
    monkeypatch.setattr(collector, "run_check", run)
    result = collector.collect_matrix(tmp_path)
    assert [name for name, _ in calls] == ["pytest", "compileall", "matrix"]
    assert calls[0][1][1:] == ["-m", "pytest", "-q"]
    assert not result.passed and result.status is MatrixEvidenceStatus.FAIL
    assert result.metadata["checks"][0]["exit_code"] == 1


def test_matrix_collector_invalidates_source_mutation(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text("before")
    def run(name, *args, **kwargs):
        project.write_text("after")
        return receipt(name)
    monkeypatch.setattr(collector, "run_check", run)
    assert not collector.collect_matrix(tmp_path).passed


def test_cli_cannot_claim_another_platform(tmp_path):
    with pytest.raises(ValueError, match="actual interpreter"):
        main(["evidence", "--platform", "android", "--output", str(tmp_path / "row.json")])
    assert not (tmp_path / "row.json").exists()


def test_cli_failed_collection_returns_nonzero_and_retains_failure(tmp_path, monkeypatch):
    from ai_character_engine.release_candidate import __main__ as cli
    monkeypatch.setattr(cli, "collect_matrix", lambda *a, **kw: replace(row(), status=MatrixEvidenceStatus.FAIL))
    path = tmp_path / "row.json"
    assert main(["evidence", "--output", str(path)]) == 2
    assert json.loads(path.read_text())["status"] == "fail"


def test_standard_enum_manifest_is_portable_and_actual_member_lookup_still_works():
    from enum import Enum
    import ai_character_engine as ace
    from ai_character_engine.compatibility.manifest import _parameters
    count = 0
    for name in ace.__all__:
        value = getattr(ace, name)
        if isinstance(value, type) and issubclass(value, Enum) and value.__members__:
            count += 1
            for member in value:
                assert value(member.value) is member
                assert value(value=member.value) is member
            parameters = _parameters(value)
            assert [(p.name, p.kind, p.required) for p in parameters] == [("values", "var_positional", False)]
    assert count == 45


def test_custom_enum_metaclass_call_signature_is_not_hidden():
    from enum import Enum, EnumType
    from ai_character_engine.compatibility.manifest import _parameters
    class CustomType(EnumType):
        def __call__(cls, value, *, required_new_argument):
            return super().__call__(value)
    class Custom(Enum, metaclass=CustomType):
        ONE = 1
    assert any(p.name == "required_new_argument" and p.required for p in _parameters(Custom))


def test_explicit_enum_signature_is_not_normalized():
    from enum import Enum
    from inspect import Signature, Parameter
    from ai_character_engine.compatibility.manifest import _parameters
    class Custom(Enum):
        ONE = 1
    Custom.__signature__ = Signature([Parameter("custom", Parameter.KEYWORD_ONLY)])
    assert [(p.name, p.kind) for p in _parameters(Custom)] == [("custom", "keyword_only")]
