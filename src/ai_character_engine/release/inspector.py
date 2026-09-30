from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
from typing import Iterable
import zipfile

from .matrix import default_release_matrix
from .models import (
    DistributionArtifact,
    DistributionArtifactKind,
    ReleaseCheck,
    ReleaseCheckStatus,
    ReleaseManifest,
    ReleaseReadinessReport,
)

FORBIDDEN_ARCHIVE_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_ARCHIVE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}


def _normalise_distribution(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(members: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(member).replace("\\", "/") for member in members)
    for member in result:
        pure = PurePosixPath(member)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive member path: {member!r}")
    return result


def _metadata_from_bytes(raw: bytes):
    return BytesParser(policy=email_policy).parsebytes(raw)


def inspect_distribution_artifact(path: str | Path) -> DistributionArtifact:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    sha256 = sha256_file(file_path)
    if file_path.suffix == ".whl":
        with zipfile.ZipFile(file_path) as archive:
            members = _safe_members(archive.namelist())
            metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
            wheel_names = [name for name in members if name.endswith(".dist-info/WHEEL")]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ValueError("wheel must contain exactly one METADATA and WHEEL file")
            metadata = _metadata_from_bytes(archive.read(metadata_names[0]))
            wheel = _metadata_from_bytes(archive.read(wheel_names[0]))
            tags = tuple(wheel.get_all("Tag", []))
            purelib = str(wheel.get("Root-Is-Purelib", "")).lower() == "true"
        return DistributionArtifact(
            path=str(file_path),
            filename=file_path.name,
            kind=DistributionArtifactKind.WHEEL,
            distribution=str(metadata["Name"]),
            version=str(metadata["Version"]),
            sha256=sha256,
            size_bytes=file_path.stat().st_size,
            requires_python=metadata.get("Requires-Python"),
            wheel_tags=tags,
            root_is_purelib=purelib,
            members=members,
        )

    if file_path.name.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as archive:
            members = _safe_members(member.name for member in archive.getmembers())
            pyprojects = [name for name in members if name.count("/") == 1 and name.endswith("/pyproject.toml")]
            if len(pyprojects) != 1:
                raise ValueError("sdist must contain exactly one top-level pyproject.toml")
            extracted = archive.extractfile(pyprojects[0])
            if extracted is None:
                raise ValueError("cannot read sdist pyproject.toml")
            project = tomllib.loads(extracted.read().decode("utf-8"))["project"]
        return DistributionArtifact(
            path=str(file_path),
            filename=file_path.name,
            kind=DistributionArtifactKind.SDIST,
            distribution=str(project["name"]),
            version=str(project["version"]),
            sha256=sha256,
            size_bytes=file_path.stat().st_size,
            requires_python=project.get("requires-python"),
            members=members,
        )
    raise ValueError(f"unsupported distribution artifact: {file_path.name}")


def artifact_hygiene_issues(artifact: DistributionArtifact) -> tuple[str, ...]:
    issues: list[str] = []
    for member in artifact.members:
        lower = member.lower()
        parts = set(PurePosixPath(lower).parts)
        if lower.endswith(FORBIDDEN_ARCHIVE_SUFFIXES):
            issues.append(f"forbidden file: {member}")
        if parts & FORBIDDEN_ARCHIVE_PARTS:
            issues.append(f"forbidden build/cache path: {member}")
        if any(part.endswith(".egg-info") for part in parts):
            # sdists legitimately include the package's generated egg-info metadata.
            if artifact.kind is DistributionArtifactKind.WHEEL:
                issues.append(f"wheel unexpectedly contains egg-info: {member}")
    return tuple(dict.fromkeys(issues))


def validate_distribution_artifact(
    artifact: DistributionArtifact,
    *,
    expected_distribution: str | None = None,
    expected_version: str | None = None,
    require_portable_wheel: bool = True,
) -> tuple[str, ...]:
    issues = list(artifact_hygiene_issues(artifact))
    if expected_distribution and _normalise_distribution(artifact.distribution) != _normalise_distribution(expected_distribution):
        issues.append(
            f"distribution mismatch: expected {expected_distribution!r}, got {artifact.distribution!r}"
        )
    if expected_version and artifact.version != expected_version:
        issues.append(f"version mismatch: expected {expected_version!r}, got {artifact.version!r}")
    if artifact.kind is DistributionArtifactKind.WHEEL and require_portable_wheel:
        if not artifact.portable_pure_python_wheel:
            issues.append(f"wheel is not portable pure Python: tags={artifact.wheel_tags!r}")
    if artifact.kind is DistributionArtifactKind.SDIST:
        lower_members = tuple(member.lower() for member in artifact.members)
        if not any(member.endswith("/readme.md") for member in lower_members):
            issues.append("sdist is missing README.md")
        if not any("/src/" in member for member in lower_members):
            issues.append("sdist is missing src package files")
    return tuple(issues)


def build_release_manifest(
    *,
    engine_version: str,
    python_requires: str,
    artifacts: Iterable[DistributionArtifact],
    contract_versions: dict[str, int],
    metadata: dict | None = None,
) -> ReleaseManifest:
    return ReleaseManifest(
        engine_version=engine_version,
        python_requires=python_requires,
        matrix=default_release_matrix(),
        artifacts=tuple(artifacts),
        contract_versions=contract_versions,
        metadata=metadata or {},
    )


def save_release_manifest(manifest: ReleaseManifest, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_release_manifest(path: str | Path) -> ReleaseManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("release manifest must contain a JSON object")
    return ReleaseManifest.from_dict(raw)


def _status(name: str, ok: bool, pass_detail: str, fail_detail: str) -> ReleaseCheck:
    return ReleaseCheck(name, ReleaseCheckStatus.PASS if ok else ReleaseCheckStatus.FAIL, pass_detail if ok else fail_detail)


def check_release_readiness(
    project_root: str | Path,
    *,
    artifacts: Iterable[DistributionArtifact] = (),
    strict_v1: bool = False,
) -> ReleaseReadinessReport:
    root = Path(project_root)
    checks: list[ReleaseCheck] = []
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return ReleaseReadinessReport((ReleaseCheck("pyproject", ReleaseCheckStatus.FAIL, "pyproject.toml is missing"),))
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = str(project.get("version", ""))
    checks.append(_status("project.version", bool(version), f"project version is {version}", "project.version is missing"))
    checks.append(_status("project.requires-python", bool(project.get("requires-python")), f"requires-python={project.get('requires-python')}", "requires-python is missing"))
    backend = data.get("build-system", {}).get("build-backend")
    checks.append(_status("build.backend", backend == "setuptools.build_meta", "setuptools PEP 517 backend is configured", f"unexpected build backend: {backend!r}"))
    package_where = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("where")
    checks.append(_status("build.src-layout", package_where == ["src"], "src-layout package discovery is explicit", f"expected tool.setuptools.packages.find.where=['src'], got {package_where!r}"))

    version_file = root / "src" / "ai_character_engine" / "_version.py"
    version_text = version_file.read_text(encoding="utf-8") if version_file.is_file() else ""
    checks.append(_status("version.sync.core", f'VERSION = "{version}"' in version_text or f"VERSION = '{version}'" in version_text, "core runtime version matches pyproject", "core runtime version does not match pyproject"))

    package_projects: dict[str, dict] = {}
    packages_root = root / "packages"
    if packages_root.is_dir():
        for package_path in sorted(packages_root.glob("*/pyproject.toml")):
            package_dir = package_path.parent.name
            try:
                package_data = tomllib.loads(package_path.read_text(encoding="utf-8"))["project"]
            except Exception:
                checks.append(ReleaseCheck(f"version.sync.{package_dir}", ReleaseCheckStatus.FAIL, f"cannot read {package_path}"))
                continue
            package_name = str(package_data.get("name", package_dir))
            package_projects[_normalise_distribution(package_name)] = package_data
            ok = package_data.get("version") == version
            deps = tuple(str(dep) for dep in package_data.get("dependencies", ()))
            lower_bound_ok = any(dep.startswith(f"ai-character-engine>={version}") for dep in deps)
            checks.append(_status(f"version.sync.{package_dir}", ok and lower_bound_ok, f"{package_name} version/dependency match core {version}", f"{package_name} version or dependency floor does not match core {version}"))

    required_files = ["README.md", "CHANGELOG.md", "VALIDATION.md", "docs/public_api_v1_stable.json"]
    for required in required_files:
        checks.append(_status(f"file.{required}", (root / required).is_file(), f"{required} exists", f"{required} is missing"))

    license_exists = any(path.is_file() for path in (root / "LICENSE", root / "LICENSE.md", root / "LICENSE.txt"))
    license_status = ReleaseCheckStatus.PASS if license_exists else (ReleaseCheckStatus.FAIL if strict_v1 else ReleaseCheckStatus.WARNING)
    checks.append(ReleaseCheck("license", license_status, "license file exists" if license_exists else "no explicit LICENSE file; choose one before v1.0 open-source release"))

    ci = root / ".github" / "workflows" / "ci.yml"
    release = root / ".github" / "workflows" / "release.yml"
    checks.append(_status("workflow.ci", ci.is_file(), "cross-platform CI workflow exists", ".github/workflows/ci.yml is missing"))
    checks.append(_status("workflow.release", release.is_file(), "release workflow exists", ".github/workflows/release.yml is missing"))
    if ci.is_file():
        ci_text = ci.read_text(encoding="utf-8").lower()
        matrix_ok = all(token in ci_text for token in ("ubuntu-latest", "macos-latest", "windows-latest", "3.11", "3.12", "3.13"))
        checks.append(_status("workflow.matrix", matrix_ok, "CI declares Linux/macOS/Windows x Python 3.11/3.12/3.13", "CI matrix is missing a supported OS/Python target"))

    inspected = tuple(artifacts)
    if inspected:
        expected_names = {
            _normalise_distribution(str(project.get("name", "ai-character-engine"))): {
                DistributionArtifactKind.WHEEL, DistributionArtifactKind.SDIST
            },
            **{
                name: {DistributionArtifactKind.WHEEL, DistributionArtifactKind.SDIST}
                for name in package_projects
            },
        }
        by_name: dict[str, set[DistributionArtifactKind]] = {}
        artifact_issues: list[str] = []
        for artifact in inspected:
            normalized = _normalise_distribution(artifact.distribution)
            by_name.setdefault(normalized, set()).add(artifact.kind)
            artifact_issues.extend(validate_distribution_artifact(artifact, expected_version=version))
        complete = all(by_name.get(name) == kinds for name, kinds in expected_names.items())
        checks.append(_status("artifacts.complete", complete, "core + optional package wheel/sdist artifacts are present", f"artifact set is incomplete: {by_name!r}"))
        checks.append(_status("artifacts.valid", not artifact_issues, "all distribution artifacts passed portability/hygiene validation", "; ".join(artifact_issues)))
    else:
        checks.append(ReleaseCheck("artifacts", ReleaseCheckStatus.WARNING, "no built artifacts supplied to readiness check"))

    return ReleaseReadinessReport(tuple(checks))
