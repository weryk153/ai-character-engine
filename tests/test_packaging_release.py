from __future__ import annotations

from ai_character_engine._version import VERSION
import ast
from email.message import Message
import io
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile

import pytest

import ai_character_engine as ace
from ai_character_engine.compatibility import current_contract_versions
from ai_character_engine.release import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_PLATFORMS,
    SUPPORTED_PYTHON_VERSIONS,
    DistributionArtifact,
    DistributionArtifactKind,
    ReleaseCheck,
    ReleaseCheckStatus,
    ReleaseManifest,
    ReleaseMatrix,
    ReleaseReadinessReport,
    ReleaseTarget,
    artifact_hygiene_issues,
    build_release_manifest,
    check_release_readiness,
    default_release_matrix,
    inspect_distribution_artifact,
    load_release_manifest,
    save_release_manifest,
    sha256_file,
    validate_distribution_artifact,
)
from ai_character_engine.release.__main__ import main as release_main

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "src" / "ai_character_engine" / "release"
SEALED_V046 = ROOT / "tests/fixtures/api" / "public_api_v0.46_sealed.json"


def make_wheel(
    path: Path,
    *,
    name: str = "ai-character-engine",
    version: str = VERSION,
    tag: str = "py3-none-any",
    purelib: bool = True,
    extra_members: tuple[str, ...] = (),
) -> Path:
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\nRequires-Python: >=3.11\n\n"
    wheel = f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: {'true' if purelib else 'false'}\nTag: {tag}\n\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist}/__init__.py", "")
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(f"{info}/WHEEL", wheel)
        archive.writestr(f"{info}/RECORD", "")
        for member in extra_members:
            archive.writestr(member, b"x")
    return path


def make_sdist(
    path: Path,
    *,
    name: str = "ai-character-engine",
    version: str = VERSION,
    extra_members: tuple[str, ...] = (),
) -> Path:
    root = f"{name.replace('-', '_')}-{version}"
    pyproject = f'''[build-system]\nrequires=["setuptools>=68"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="{name}"\nversion="{version}"\nreadme="README.md"\nrequires-python=">=3.11"\n'''.encode()
    with tarfile.open(path, "w:gz") as archive:
        for member, data in (
            (f"{root}/pyproject.toml", pyproject),
            (f"{root}/README.md", b"# readme\n"),
            (f"{root}/src/pkg/__init__.py", b""),
        ):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for member in extra_members:
            data = b"x"
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_release_manifest_contract_version_is_explicit():
    assert RELEASE_MANIFEST_SCHEMA_VERSION == 1


def test_declared_release_matrix_is_linux_macos_windows_x_311_313():
    matrix = default_release_matrix()
    assert SUPPORTED_PLATFORMS == ("linux", "macos", "windows")
    assert SUPPORTED_PYTHON_VERSIONS == ("3.11", "3.12", "3.13")
    assert len(matrix.targets) == 9
    assert {(t.platform, t.python_version) for t in matrix.targets} == {
        (os_name, py) for os_name in SUPPORTED_PLATFORMS for py in SUPPORTED_PYTHON_VERSIONS
    }


@pytest.mark.parametrize("platform_name", ["linux", "macos", "windows", "LINUX"])
def test_release_target_normalises_supported_platform(platform_name):
    target = ReleaseTarget(platform_name, "3.11")
    assert target.platform == platform_name.lower()


@pytest.mark.parametrize("platform_name", ["android", "ios", "freebsd", ""])
def test_release_target_rejects_undeclared_platform(platform_name):
    with pytest.raises(ValueError, match="platform"):
        ReleaseTarget(platform_name, "3.11")


@pytest.mark.parametrize("version", ["3", "3.11.1", "x.y", ""])
def test_release_target_rejects_non_major_minor_python(version):
    with pytest.raises(ValueError, match="major.minor"):
        ReleaseTarget("linux", version)


def test_release_matrix_rejects_duplicates_and_empty():
    with pytest.raises(ValueError, match="at least"):
        ReleaseMatrix(())
    target = ReleaseTarget("linux", "3.11")
    with pytest.raises(ValueError, match="unique"):
        ReleaseMatrix((target, target))


def test_release_matrix_json_roundtrip():
    original = default_release_matrix()
    assert ReleaseMatrix.from_dict(original.to_dict()) == original


def test_sha256_file_is_deterministic(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_inspect_portable_wheel_extracts_metadata_tags_and_members(tmp_path):
    path = make_wheel(tmp_path / "ace.whl")
    artifact = inspect_distribution_artifact(path)
    assert artifact.kind is DistributionArtifactKind.WHEEL
    assert artifact.distribution == "ai-character-engine"
    assert artifact.version == VERSION
    assert artifact.requires_python == ">=3.11"
    assert artifact.wheel_tags == ("py3-none-any",)
    assert artifact.root_is_purelib is True
    assert artifact.portable_pure_python_wheel
    assert artifact.members


def test_inspect_nonportable_wheel_is_not_marked_portable(tmp_path):
    path = make_wheel(tmp_path / "native.whl", tag="cp313-cp313-macosx_14_0_arm64", purelib=False)
    artifact = inspect_distribution_artifact(path)
    assert not artifact.portable_pure_python_wheel
    assert "not portable pure Python" in validate_distribution_artifact(artifact)[0]


def test_inspect_sdist_reads_project_metadata(tmp_path):
    path = make_sdist(tmp_path / "ace.tar.gz")
    artifact = inspect_distribution_artifact(path)
    assert artifact.kind is DistributionArtifactKind.SDIST
    assert artifact.distribution == "ai-character-engine"
    assert artifact.version == VERSION
    assert artifact.requires_python == ">=3.11"
    assert validate_distribution_artifact(artifact) == ()


def test_inspector_rejects_unknown_artifact_extension(tmp_path):
    path = tmp_path / "x.zip"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        inspect_distribution_artifact(path)


def test_inspector_rejects_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_distribution_artifact(tmp_path / "missing.whl")


def test_wheel_requires_exactly_one_metadata_and_wheel_file(tmp_path):
    path = tmp_path / "bad.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x/__init__.py", "")
    with pytest.raises(ValueError, match="exactly one"):
        inspect_distribution_artifact(path)


def test_sdist_requires_top_level_pyproject(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("pkg/README.md")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="pyproject"):
        inspect_distribution_artifact(path)


def test_archive_member_path_traversal_is_rejected(tmp_path):
    path = make_wheel(tmp_path / "unsafe.whl", extra_members=("../escape.txt",))
    with pytest.raises(ValueError, match="unsafe archive"):
        inspect_distribution_artifact(path)


@pytest.mark.parametrize("member", ["pkg/x.pyc", "pkg/x.pyo", "pkg/__pycache__/x.py", "pkg/build/x.py", "pkg/dist/x.py"])
def test_artifact_hygiene_rejects_forbidden_release_payload(tmp_path, member):
    artifact = inspect_distribution_artifact(make_wheel(tmp_path / "dirty.whl", extra_members=(member,)))
    assert artifact_hygiene_issues(artifact)


def test_validation_detects_distribution_mismatch(tmp_path):
    artifact = inspect_distribution_artifact(make_wheel(tmp_path / "x.whl"))
    issues = validate_distribution_artifact(artifact, expected_distribution="other")
    assert any("distribution mismatch" in issue for issue in issues)


def test_validation_detects_version_mismatch(tmp_path):
    artifact = inspect_distribution_artifact(make_wheel(tmp_path / "x.whl"))
    issues = validate_distribution_artifact(artifact, expected_version="9.9.9")
    assert any("version mismatch" in issue for issue in issues)


def test_distribution_artifact_validation_rejects_bad_hash_and_size():
    with pytest.raises(ValueError, match="sha256"):
        DistributionArtifact("x", "x.whl", DistributionArtifactKind.WHEEL, "x", "1", "abc", 1)
    with pytest.raises(ValueError, match="size"):
        DistributionArtifact("x", "x.whl", DistributionArtifactKind.WHEEL, "x", "1", "0" * 64, 0)


def test_release_manifest_roundtrip_and_mapping_immutability(tmp_path):
    artifact = inspect_distribution_artifact(make_wheel(tmp_path / "x.whl"))
    manifest = build_release_manifest(
        engine_version=VERSION,
        python_requires=">=3.11",
        artifacts=(artifact,),
        contract_versions={"public_api": 1},
        metadata={"channel": "candidate"},
    )
    path = tmp_path / "manifest.json"
    save_release_manifest(manifest, path)
    loaded = load_release_manifest(path)
    assert loaded.to_dict() == manifest.to_dict()
    with pytest.raises(TypeError):
        loaded.contract_versions["x"] = 2  # type: ignore[index]


def test_release_manifest_rejects_unknown_schema():
    raw = {
        "schema_version": 99,
        "engine_version": VERSION,
        "python_requires": ">=3.11",
        "matrix": default_release_matrix().to_dict(),
        "artifacts": [],
        "contract_versions": {},
    }
    with pytest.raises(ValueError, match="schema"):
        ReleaseManifest.from_dict(raw)


def test_release_report_passes_with_warnings_but_not_failures():
    report = ReleaseReadinessReport((ReleaseCheck("x", ReleaseCheckStatus.WARNING, "warn"),))
    assert report.passed and len(report.warnings) == 1
    failed = ReleaseReadinessReport((ReleaseCheck("x", ReleaseCheckStatus.FAIL, "bad"),))
    assert not failed.passed and len(failed.failures) == 1


def test_current_project_release_readiness_accepts_owner_selected_license():
    report = check_release_readiness(ROOT)
    assert report.passed
    assert {warning.name for warning in report.warnings} == {"artifacts"}


def test_project_strict_v1_fails_without_explicit_license(unlicensed_project):
    report = check_release_readiness(unlicensed_project, strict_v1=True)
    assert not report.passed
    assert {failure.name for failure in report.failures} == {"license"}


def test_current_project_build_backend_and_src_discovery_are_explicit_setuptools():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_adapter_projects_share_core_version_and_setuptools_backend():
    for directory in ("renderer-vrm",):
        data = tomllib.loads((ROOT / "packages" / directory / "pyproject.toml").read_text())
        assert data["project"]["version"] == ace.__version__ == VERSION
        assert data["build-system"]["build-backend"] == "setuptools.build_meta"
        assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
        assert any(dep.startswith(f"ai-character-engine>={VERSION}") for dep in data["project"]["dependencies"])


def test_ci_workflow_declares_full_cross_platform_matrix():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text().lower()
    for token in ("ubuntu-latest", "macos-latest", "windows-latest", '"3.11"', '"3.12"', '"3.13"'):
        assert token in text
    assert "ai_character_engine.release_candidate evidence" in text
    assert ".[dev,service]" in text


def test_release_workflow_builds_both_distributions_without_publishing_credentials():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text().lower()
    assert "needs: validate" in text
    assert "actions/download-artifact" in text
    assert "name: final-acceptance" in text
    for package in ("core", "vrm"):
        assert f"/distributions/{package}/*.whl" in text
        assert f"/distributions/{package}/*.tar.gz" in text
    collector = (ROOT / "tools/final_acceptance.py").read_text()
    assert '"-m", "build"' in collector
    assert "packages/renderer-vrm" in collector
    assert "upload-artifact" in text
    assert "pypi" not in text and "twine upload" not in text


def test_release_core_has_no_authority_runtime_imports():
    forbidden = (
        "ai_character_engine.runtime",
        "ai_character_engine.memory",
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.world",
        "ai_character_engine.multi_character",
    )
    offenders = []
    for path in RELEASE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        offenders.append((path.name, alias.name))
            if module and module.startswith(forbidden):
                offenders.append((path.name, module))
    assert offenders == []


def test_release_core_has_no_registry_vendor_or_renderer_product_coupling():
    text = "\n".join(path.read_text().lower() for path in RELEASE.rglob("*.py"))
    for token in ("pypi", "aws", "gcp", "azure", "kubernetes", "docker", "live2d", "three_vrm", "cubism"):
        assert token not in text


def test_release_symbols_are_exported_from_root_package():
    expected = {
        "ReleaseManifest",
        "ReleaseMatrix",
        "ReleaseTarget",
        "DistributionArtifact",
        "check_release_readiness",
        "inspect_distribution_artifact",
        "default_release_matrix",
    }
    assert expected <= set(ace.__all__)
    assert all(hasattr(ace, name) for name in expected)


def test_later_sealed_api_only_adds_symbols():
    from ai_character_engine.compatibility import build_public_api_manifest, compare_public_api_manifests, load_public_api_manifest

    baseline = load_public_api_manifest(SEALED_V046)
    current = build_public_api_manifest()
    report = compare_public_api_manifests(baseline, current)
    assert baseline.engine_version == "0.46.0"
    assert current.engine_version == VERSION
    assert report.compatible and report.breaking == ()
    assert {issue.code for issue in report.issues} <= {"symbol_added"}


def test_release_cli_matrix(capsys):
    assert release_main(["matrix"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["targets"]) == 9


def test_release_cli_inspect(tmp_path, capsys):
    artifact = make_wheel(tmp_path / "x.whl")
    assert release_main(["inspect", str(artifact)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["wheel_tags"] == ["py3-none-any"]


def test_release_cli_manifest(tmp_path, capsys):
    artifact = make_wheel(tmp_path / "x.whl")
    output = tmp_path / "manifest.json"
    assert release_main(["manifest", str(artifact), "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine_version"] == VERSION
    assert output.is_file()


def test_release_cli_check_candidate_passes_and_writes_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert release_main(["check", "--project-root", str(ROOT), "--fail-on-error", "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert output.is_file()


def test_release_cli_strict_v1_fails_without_license(capsys, unlicensed_project):
    assert release_main(["check", "--project-root", str(unlicensed_project), "--strict-v1", "--fail-on-error"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False


def test_release_manifest_contract_versions_are_independent_from_engine_semver(tmp_path):
    artifact = inspect_distribution_artifact(make_wheel(tmp_path / "x.whl"))
    contracts = {surface.value: version for surface, version in current_contract_versions().items()}
    manifest = build_release_manifest(
        engine_version=ace.__version__,
        python_requires=">=3.11",
        artifacts=(artifact,),
        contract_versions=contracts,
    )
    assert manifest.engine_version == VERSION
    assert set(manifest.contract_versions.values()) == {1}


def test_the_version_being_released_is_named_once():
    # The one test that names the version; every other test reads VERSION.
    assert VERSION == "1.3.2"
