from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ai_character_engine.release import ReleaseMatrix

from .provenance import MATRIX_CHECKS, receipt_error

from .models import (
    MatrixEvidence,
    ReleaseCandidateGate,
    ReleaseCandidateGateStatus,
)


def save_matrix_evidence(evidence: MatrixEvidence, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_matrix_evidence(path: str | Path) -> MatrixEvidence:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("matrix evidence must contain a JSON object")
    return MatrixEvidence.from_dict(raw)


def validate_matrix_evidence(
    matrix: ReleaseMatrix,
    evidence: Iterable[MatrixEvidence],
    *,
    engine_version: str,
    candidate_sha256: str | None = None,
) -> tuple[ReleaseCandidateGate, ...]:
    expected = {(item.platform, item.python_version) for item in matrix.targets}
    indexed: dict[tuple[str, str], MatrixEvidence] = {}
    duplicates: set[tuple[str, str]] = set()
    unexpected: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item.platform, item.python_version)
        if key not in expected:
            unexpected.add(key)
            continue
        if key in indexed:
            duplicates.add(key)
        indexed[key] = item

    gates: list[ReleaseCandidateGate] = []
    for platform, python_version in sorted(expected):
        key = (platform, python_version)
        name = f"matrix.{platform}.py{python_version}"
        if key in duplicates:
            gates.append(ReleaseCandidateGate(name, ReleaseCandidateGateStatus.FAIL, "duplicate evidence records"))
            continue
        item = indexed.get(key)
        if item is None:
            gates.append(ReleaseCandidateGate(name, ReleaseCandidateGateStatus.BLOCKED, "CI evidence has not been supplied"))
            continue
        if item.engine_version != engine_version:
            gates.append(
                ReleaseCandidateGate(
                    name,
                    ReleaseCandidateGateStatus.FAIL,
                    f"evidence version {item.engine_version} does not match candidate {engine_version}",
                )
            )
            continue
        identity = item.metadata.get("candidate_sha256")
        if candidate_sha256 is None or identity != candidate_sha256:
            gates.append(ReleaseCandidateGate(name, ReleaseCandidateGateStatus.FAIL,
                                             "evidence candidate bytes do not match the selected candidate"))
            continue
        error = receipt_error(item.metadata.get("checks"), MATRIX_CHECKS)
        if error or item.metadata.get("source_unchanged") is not True:
            gates.append(ReleaseCandidateGate(name, ReleaseCandidateGateStatus.FAIL,
                                             error or "candidate changed during checks"))
            continue
        if not item.passed:
            gates.append(
                ReleaseCandidateGate(
                    name,
                    ReleaseCandidateGateStatus.FAIL,
                    item.test_summary or "matrix row did not pass pytest/compile/matrix self-check",
                )
            )
            continue
        gates.append(
            ReleaseCandidateGate(
                name,
                ReleaseCandidateGateStatus.PASS,
                item.test_summary or "pytest, compileall and release-matrix self-check passed",
            )
        )

    for platform, python_version in sorted(unexpected):
        gates.append(
            ReleaseCandidateGate(
                f"matrix.unexpected.{platform}.py{python_version}",
                ReleaseCandidateGateStatus.FAIL,
                "evidence target is outside the supported release matrix",
            )
        )
    return tuple(gates)
