from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_character_engine._version import VERSION
from ai_character_engine.release import inspect_distribution_artifact

from .acceptance import evaluate_release_candidate
from .evidence import load_matrix_evidence, save_matrix_evidence
from .models import MatrixEvidence, MatrixEvidenceStatus
from .collector import collect_matrix
from .provenance import load_final_evidence
from ai_character_engine.release.matrix import current_release_target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Character Engine v1 release-candidate acceptance tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    evidence = sub.add_parser("evidence", help="write one cross-platform matrix evidence record")
    evidence.add_argument("--project-root", type=Path, default=Path.cwd())
    evidence.add_argument("--platform")
    evidence.add_argument("--python-version")
    evidence.add_argument("--source", default="ci")
    evidence.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("check", help="evaluate the complete v1.0 release-candidate gate")
    check.add_argument("--project-root", type=Path, default=Path.cwd())
    check.add_argument("--evidence", type=Path, action="append", default=[])
    check.add_argument("--evidence-dir", type=Path)
    check.add_argument("--acceptance-evidence-dir", type=Path)
    check.add_argument("--artifact", type=Path, action="append", default=[])
    check.add_argument("--stable-api", type=Path)
    check.add_argument("--output", type=Path)
    check.add_argument("--fail-on-blocker", action="store_true")
    return parser


def _load_evidence(args) -> list[MatrixEvidence]:
    paths = list(args.evidence)
    if args.evidence_dir is not None and args.evidence_dir.is_dir():
        paths.extend(sorted(args.evidence_dir.glob("*.json")))
    return [load_matrix_evidence(path) for path in paths]


def _emit(report) -> None:
    print(f"v1 ready={report.ready_for_v1}")
    for gate in report.gates:
        print(f"[{gate.status.value}] {gate.name}: {gate.detail}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evidence":
        if args.output.resolve().is_relative_to(args.project_root.resolve()):
            raise ValueError("evidence output must be outside the candidate source tree")
        target = current_release_target()
        if ((args.platform and args.platform.lower() != target.platform)
                or (args.python_version and args.python_version != target.python_version)):
            raise ValueError("requested target does not match the actual interpreter/platform")
        evidence = collect_matrix(args.project_root, source=args.source)
        save_matrix_evidence(evidence, args.output)
        print(args.output)
        return 0 if evidence.passed else 2

    if args.command == "check":
        evidence = _load_evidence(args)
        artifacts = [inspect_distribution_artifact(path) for path in args.artifact]
        report = evaluate_release_candidate(
            args.project_root,
            matrix_evidence=evidence,
            artifacts=artifacts,
            stable_api_path=args.stable_api,
            acceptance_evidence=([load_final_evidence(p) for p in sorted(args.acceptance_evidence_dir.glob("*.json"))]
                                 if args.acceptance_evidence_dir else ()),
        )
        _emit(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2 if args.fail_on_blocker and not report.ready_for_v1 else 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
