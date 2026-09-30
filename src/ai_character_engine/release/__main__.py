from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ai_character_engine.compatibility import current_contract_versions
from ai_character_engine._version import VERSION
from .inspector import (
    build_release_manifest,
    check_release_readiness,
    inspect_distribution_artifact,
    save_release_manifest,
)
from .matrix import default_release_matrix


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Character Engine packaging/release validation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("matrix", help="print the declared cross-platform release matrix")

    inspect_parser = sub.add_parser("inspect", help="inspect built wheel/sdist artifacts")
    inspect_parser.add_argument("artifacts", nargs="+", type=Path)

    manifest_parser = sub.add_parser("manifest", help="write a release manifest from built artifacts")
    manifest_parser.add_argument("artifacts", nargs="+", type=Path)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--python-requires", default=">=3.11")

    check_parser = sub.add_parser("check", help="check repository and artifact release readiness")
    check_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    check_parser.add_argument("--artifact", action="append", default=[], type=Path)
    check_parser.add_argument("--strict-v1", action="store_true")
    check_parser.add_argument("--fail-on-error", action="store_true")
    check_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "matrix":
        _print_json(default_release_matrix().to_dict())
        return 0
    if args.command == "inspect":
        _print_json([inspect_distribution_artifact(path).to_dict() for path in args.artifacts])
        return 0
    if args.command == "manifest":
        artifacts = tuple(inspect_distribution_artifact(path) for path in args.artifacts)
        contracts = {surface.value: version for surface, version in current_contract_versions().items()}
        manifest = build_release_manifest(
            engine_version=VERSION,
            python_requires=args.python_requires,
            artifacts=artifacts,
            contract_versions=contracts,
        )
        save_release_manifest(manifest, args.output)
        _print_json(manifest.to_dict())
        return 0
    if args.command == "check":
        artifacts = tuple(inspect_distribution_artifact(path) for path in args.artifact)
        report = check_release_readiness(args.project_root, artifacts=artifacts, strict_v1=args.strict_v1)
        payload = report.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_json(payload)
        if args.fail_on_error and not report.passed:
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
