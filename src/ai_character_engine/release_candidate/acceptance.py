from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Iterable

from ai_character_engine._version import VERSION
from ai_character_engine.compatibility import ApiStability, compare_public_api_manifests
from ai_character_engine.compatibility.manifest import build_public_api_manifest, load_public_api_manifest
from ai_character_engine.persistence import verify_persistence_fixture_directory
from ai_character_engine.release import (
    DistributionArtifact,
    ReleaseCheckStatus,
    check_release_readiness,
    default_release_matrix,
)

from .evidence import validate_matrix_evidence
from .provenance import candidate_sha256, FINAL_CHECKS, receipt_error, canonical_artifacts
from .models import (
    MatrixEvidence,
    ReleaseCandidateGate,
    ReleaseCandidateGateStatus,
    ReleaseCandidateReport,
)


def _gate(name: str, ok: bool, pass_detail: str, fail_detail: str) -> ReleaseCandidateGate:
    return ReleaseCandidateGate(
        name,
        ReleaseCandidateGateStatus.PASS if ok else ReleaseCandidateGateStatus.FAIL,
        pass_detail if ok else fail_detail,
    )


def _project_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def evaluate_release_candidate(
    project_root: str | Path,
    *,
    matrix_evidence: Iterable[MatrixEvidence] = (),
    artifacts: Iterable[DistributionArtifact] = (),
    stable_api_path: str | Path | None = None,
    acceptance_evidence: Iterable[dict] = (),
) -> ReleaseCandidateReport:
    root = Path(project_root)
    version = _project_version(root)
    identity = candidate_sha256(root)
    matrix_evidence = tuple(matrix_evidence)
    artifacts = tuple(artifacts)
    acceptance_evidence = tuple(acceptance_evidence)
    gates: list[ReleaseCandidateGate] = []

    readiness = check_release_readiness(root, artifacts=tuple(artifacts), strict_v1=True)
    for check in readiness.checks:
        if check.status is ReleaseCheckStatus.PASS:
            status = ReleaseCandidateGateStatus.PASS
        elif check.name in {"license", "artifacts"}:
            status = ReleaseCandidateGateStatus.BLOCKED
        else:
            status = ReleaseCandidateGateStatus.FAIL
        gates.append(ReleaseCandidateGate(f"release.{check.name}", status, check.detail))

    stable_path = Path(stable_api_path) if stable_api_path is not None else root / "docs/public_api_v1_stable.json"
    if not stable_path.is_file():
        gates.append(
            ReleaseCandidateGate(
                "api.stable_manifest",
                ReleaseCandidateGateStatus.BLOCKED,
                "docs/public_api_v1_stable.json has not been frozen",
            )
        )
    else:
        try:
            stable = load_public_api_manifest(stable_path)
            current = build_public_api_manifest(stability=ApiStability.STABLE, engine_version=version)
            report = compare_public_api_manifests(stable, current)
            all_stable = bool(stable.symbols) and all(item.stability is ApiStability.STABLE for item in stable.symbols)
            gates.append(
                _gate(
                    "api.stable_manifest",
                    stable.engine_version == version and all_stable and report.compatible and not report.issues
                    and stable.metadata.get("status") == "v1-stable"
                    and stable.metadata.get("frozen_in") == "0.48.0",
                    f"stable API freeze matches runtime ({len(stable.symbols)} symbols)",
                    "stable API freeze does not exactly match the current root public API",
                )
            )
        except Exception as exc:
            gates.append(ReleaseCandidateGate("api.stable_manifest", ReleaseCandidateGateStatus.FAIL, f"{type(exc).__name__}: {exc}"))

    prior_path = root / "tests/fixtures/api/public_api_v0.47_sealed.json"
    if not prior_path.is_file() or not stable_path.is_file():
        gates.append(
            ReleaseCandidateGate(
                "api.v047_upgrade",
                ReleaseCandidateGateStatus.FAIL,
                "sealed v0.47 and stable v1 candidate manifests are required",
            )
        )
    else:
        prior = load_public_api_manifest(prior_path)
        stable = load_public_api_manifest(stable_path)
        report = compare_public_api_manifests(prior, stable)
        gates.append(
            _gate(
                "api.v047_upgrade",
                report.compatible,
                f"v0.47 -> v{version} stable API compatible ({len(report.issues)} additive/warning issues)",
                f"v0.47 -> v{version} contains {len(report.breaking)} breaking API issues",
            )
        )

    sealed_v048 = root / "tests/fixtures/api/public_api_v0.48_sealed.json"
    if sealed_v048.is_file() and stable_path.is_file():
        upgrade = compare_public_api_manifests(load_public_api_manifest(sealed_v048), load_public_api_manifest(stable_path))
        gates.append(_gate("api.v048_upgrade", upgrade.compatible and not upgrade.issues,
                           "sealed v0.48 -> v1 stable API: 0 breaking / 0 additions",
                           "v1 stable API differs from sealed v0.48"))
    else:
        gates.append(_gate("api.v048_upgrade", False, "", "sealed v0.48 and stable v1 manifests are required"))

    fixture_dir = root / "tests/fixtures/persistence/v0.45"
    try:
        replay_results = verify_persistence_fixture_directory(fixture_dir)
        failed = tuple(item for item in replay_results if not item.passed)
        gates.append(
            _gate(
                "persistence.cross_version_replay",
                len(replay_results) == 11 and not failed,
                "11/11 sealed v0.45 persistence fixtures replay deterministically",
                f"persistence replay failures: {len(failed)} / {len(replay_results)}",
            )
        )
    except Exception as exc:
        gates.append(ReleaseCandidateGate("persistence.cross_version_replay", ReleaseCandidateGateStatus.FAIL, f"{type(exc).__name__}: {exc}"))

    gates.extend(validate_matrix_evidence(default_release_matrix(), matrix_evidence, engine_version=version, candidate_sha256=identity))

    for name in FINAL_CHECKS:
        records = [r for r in acceptance_evidence if r.get("gate") == name]
        status = ReleaseCandidateGateStatus.BLOCKED
        detail = "executable final acceptance evidence has not been supplied"
        if records:
            status = ReleaseCandidateGateStatus.FAIL
            detail = "duplicate or incompatible final acceptance evidence"
            if len(records) == 1:
                record = records[0]
                error = receipt_error(record.get("checks"), (name,))
                valid = (type(record.get("schema_version")) is int and record["schema_version"] == 1
                         and record.get("candidate_sha256") == identity
                         and record.get("engine_version") == version
                         and record.get("source_unchanged") is True
                         and record.get("status") == "pass" and not error)
                if name in {"packaging", "fresh_install"}:
                    valid = valid and bool(artifacts) and record.get("artifacts") == canonical_artifacts(artifacts)
                if valid:
                    status = ReleaseCandidateGateStatus.PASS
                    detail = f"{name} executable evidence matches candidate {identity}"
                elif error:
                    detail = error
        gates.append(ReleaseCandidateGate(f"acceptance.{name}", status, detail))
    for record in acceptance_evidence:
        if record.get("gate") not in FINAL_CHECKS:
            gates.append(ReleaseCandidateGate("acceptance.unexpected", ReleaseCandidateGateStatus.FAIL,
                                             "unsupported acceptance evidence gate"))

    docs_ok = all(
        (root / name).is_file()
        for name in (
            "README.md",
            "CHANGELOG.md",
            "VALIDATION.md",
            "docs/release.md",
            "docs/public_api_v1_stable.json",
        )
    )
    gates.append(_gate("docs.release_candidate", docs_ok, "release documentation is present", "release documentation is incomplete"))

    gates.append(
        _gate(
            "version.runtime",
            version == VERSION and VERSION.split(".", 1)[0] == "1",
            f"runtime/project version is {VERSION}",
            f"version mismatch: project={version}, runtime={VERSION}",
        )
    )
    return ReleaseCandidateReport(version, tuple(gates), matrix_evidence, candidate_sha256=identity, acceptance_evidence=acceptance_evidence)
