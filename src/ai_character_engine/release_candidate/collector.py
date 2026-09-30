"""Execute fixed matrix checks against this interpreter and source candidate.

Only evidence files are written. No runtime state or policy is changed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from ai_character_engine._version import VERSION
from ai_character_engine.release.matrix import current_release_target
from .models import MatrixEvidence, MatrixEvidenceStatus
from .provenance import candidate_sha256


def run_check(name: str, command: list[str], root: Path, *, env=None) -> dict:
    result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    return {"name": name, "command": command, "exit_code": result.returncode,
            "output": result.stdout, "output_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def collect_matrix(root: Path, *, source: str = "local") -> MatrixEvidence:
    root = root.resolve()
    identity = candidate_sha256(root)
    target = current_release_target()
    source_dirs = [root / "src", *sorted((root / "packages").glob("*/src"))]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in source_dirs))
    commands = (
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q", *[str(p.relative_to(root)) for p in source_dirs]]),
        ("matrix", [sys.executable, "-m", "ai_character_engine.release", "matrix"]),
    )
    checks = [run_check(name, command, root, env=env) for name, command in commands]
    unchanged = identity == candidate_sha256(root)
    ok = [c["exit_code"] == 0 and unchanged for c in checks]
    summary = checks[0]["output"].strip().splitlines()
    return MatrixEvidence(target.platform, target.python_version, VERSION,
                          MatrixEvidenceStatus.PASS if all(ok) else MatrixEvidenceStatus.FAIL,
                          *ok, test_summary=summary[-1] if summary else "pytest produced no output",
                          source=source, metadata={"candidate_sha256": identity, "source_unchanged": unchanged,
                                                   "checks": checks, "python": sys.version})
