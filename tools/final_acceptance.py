"""Collect final acceptance receipts without modifying engine/runtime policy.

Run with the candidate installed and put --output outside the source tree.
The validator only reads these records; execution is an explicit operator step.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

from ai_character_engine._version import VERSION
from ai_character_engine.release import inspect_distribution_artifact, check_release_readiness
from ai_character_engine.release_candidate.collector import run_check
from ai_character_engine.release_candidate.provenance import candidate_sha256, canonical_artifacts, FINAL_CHECKS


def execute(command, root, env=None):
    print(json.dumps({"command": [str(x) for x in command], "cwd": str(root)}), flush=True)
    subprocess.run([str(x) for x in command], cwd=root, env=env, check=True)


def artifacts(output):
    paths = sorted((output / "distributions").rglob("*.whl")) + sorted((output / "distributions").rglob("*.tar.gz"))
    return tuple(inspect_distribution_artifact(p) for p in paths)


def check(name, root, output):
    if name == "soak":
        execute([sys.executable, "tools/acceptance/long_run_soak_failure_injection.py"], root)
    elif name == "performance":
        execute([sys.executable, "tools/acceptance/performance_profiling_budgets.py"], root)
    elif name == "architecture":
        execute([sys.executable, "-m", "pytest", "-q", "tests/test_architecture.py"], root)
    elif name == "lock":
        execute(["uv", "lock", "--check", "--offline"], root)
    elif name == "packaging":
        for package, dest in ((root, "core"), (root / "packages/renderer-vrm", "vrm")):
            execute([sys.executable, "-m", "build", package, "--outdir", output / "distributions" / dest], root)
        built = artifacts(output)
        assert len(built) == 4, "expected exactly four artifacts"
        for artifact in built:
            unexpected = [name for name in artifact.members
                          if "textbook" in name.lower() or name.lower().endswith((".docx", ".pdf"))
                          or any(part in {"tests", "examples"} for part in Path(name).parts)]
            assert not unexpected, f"non-distribution content in {artifact.filename}: {unexpected}"
            if artifact.filename.endswith(".whl"):
                with zipfile.ZipFile(artifact.path) as archive:
                    metadata_names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                    assert len(metadata_names) == 1, "expected one wheel metadata record"
                    metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
                    prefix = metadata_names[0].rsplit("/", 1)[0]
                    license_bytes = archive.read(prefix + "/licenses/LICENSE")
            else:
                with tarfile.open(artifact.path) as archive:
                    metadata_names = [n for n in archive.getnames() if n.endswith("/PKG-INFO") and n.count("/") == 1]
                    assert len(metadata_names) == 1, "expected one sdist metadata record"
                    metadata = BytesParser().parsebytes(archive.extractfile(metadata_names[0]).read())
                    prefix = metadata_names[0].rsplit("/", 1)[0]
                    license_bytes = archive.extractfile(prefix + "/LICENSE").read()
            assert metadata["License-Expression"] == "Apache-2.0", artifact.filename
            assert "LICENSE" in metadata.get_all("License-File", []), artifact.filename
            assert license_bytes == (root / "LICENSE").read_bytes(), artifact.filename
            print(json.dumps({"artifact": artifact.filename, "license": "Apache-2.0", "license_bytes_match": True}))
        report = check_release_readiness(root, artifacts=built)
        print(json.dumps(report.to_dict(), sort_keys=True))
        assert report.passed, "artifact inspection failed"
    elif name == "fresh_install":
        built = artifacts(output)
        assert len(built) == 4, "all four artifacts are required"
        # Wheel and sdist are independently installed into different clean venvs.
        for suffix in (".whl", ".tar.gz"):
            with tempfile.TemporaryDirectory(prefix="ace-fresh-") as temp:
                outside = Path(temp).resolve()
                assert not outside.is_relative_to(root)
                venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(outside / "venv")
                python = outside / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                clean = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
                selected = [a.path for a in built if a.filename.endswith(suffix)]
                execute([python, "-m", "pip", "install", "--no-deps", *selected], outside, clean)
                code = ("import pathlib,sys; import ai_character_engine as a; "
                        "import ai_character_engine_vrm as v; "
                        "modules=(a,v); "
                        "assert all(m.__version__ == " + repr(VERSION) + " for m in modules); "
                        "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()) for m in modules); "
                        "print([(m.__name__,m.__version__,m.__file__) for m in modules])")
                execute([python, "-I", "-c", code], outside, clean)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", choices=FINAL_CHECKS)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        parser.error("evidence output must be outside the candidate source tree")
    output.mkdir(parents=True, exist_ok=True)
    if args.check:
        check(args.check, root, output)
        return 0
    identity = candidate_sha256(root)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(root / p) for p in (
        "src", "packages/renderer-vrm/src")))
    evidence_dir = output / "acceptance"
    evidence_dir.mkdir(exist_ok=True)
    failed = False
    for name in FINAL_CHECKS:
        command = [sys.executable, str(Path(__file__).resolve()), "--project-root", str(root),
                   "--output", str(output), "--check", name]
        receipt = run_check(name, command, root, env=env)
        unchanged = candidate_sha256(root) == identity
        passed = receipt["exit_code"] == 0 and unchanged
        record = {"schema_version": 1, "gate": name, "engine_version": VERSION,
                  "candidate_sha256": identity, "source_unchanged": unchanged,
                  "status": "pass" if passed else "fail", "checks": [receipt]}
        if name in {"packaging", "fresh_install"}:
            record["artifacts"] = canonical_artifacts(artifacts(output))
        (evidence_dir / (name + ".json")).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{name}: {record['status']}", flush=True)
        if not passed:
            print(receipt["output"], flush=True)
        failed |= not passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
