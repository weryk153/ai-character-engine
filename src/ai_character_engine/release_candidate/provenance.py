"""Read-only candidate identity and executable-check receipt validation.

This is release tooling, not part of the frozen root runtime API. Receipts are
trusted runner attestations, not cryptographic proof of an untrusted runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


IGNORED = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist", "node_modules", ".git", ".data", "benchmarks", "rc-evidence"}
MATRIX_CHECKS = ("pytest", "compileall", "matrix")
FINAL_CHECKS = ("soak", "performance", "packaging", "fresh_install", "lock", "architecture")


def candidate_sha256(root: str | Path) -> str:
    """Hash sorted source paths and bytes; exclude only generated caches/builds.

    Every source file (including scripts, documentation and a future LICENSE)
    participates. Evidence must be stored outside
    these inputs. Checkout must preserve LF bytes on all platforms.
    """
    root = Path(root).resolve()
    files = [p for p in root.rglob("*") if p.is_file()
             and not any(part in IGNORED or part.endswith(".egg-info") for part in p.relative_to(root).parts)
             and p.suffix != ".pyc" and p.name != ".DS_Store"]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"candidate source symlink is not supported: {path}")
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def receipt_error(checks, required) -> str | None:
    if not isinstance(checks, list) or len(checks) != len(required):
        return "missing executable check receipts"
    if {c.get("name") for c in checks if isinstance(c, dict)} != set(required):
        return "missing, duplicate or unexpected check receipts"
    for check in checks:
        if type(check.get("exit_code")) is not int or check["exit_code"] != 0:
            return f"check {check.get('name')} failed"
        if not isinstance(check.get("command"), list) or not check["command"]:
            return "check command is missing"
        output = check.get("output")
        if not isinstance(output, str) or hashlib.sha256(output.encode()).hexdigest() != check.get("output_sha256"):
            return "check output digest mismatch"
    return None


def canonical_artifacts(artifacts) -> dict[str, str]:
    return {item.filename: item.sha256 for item in artifacts}


def load_final_evidence(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("final acceptance evidence must be a JSON object")
    return value
