from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checker import compare_public_api_manifests
from .manifest import build_public_api_manifest, load_public_api_manifest, save_public_api_manifest
from .models import ApiStability


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Character Engine public API compatibility tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="write the current public API manifest")
    write.add_argument("path", type=Path)
    write.add_argument("--stability", choices=[item.value for item in ApiStability], default=ApiStability.CANDIDATE.value)

    check = sub.add_parser("check", help="compare a baseline manifest with the current runtime")
    check.add_argument("path", type=Path)
    check.add_argument("--stability", choices=[item.value for item in ApiStability], default=ApiStability.CANDIDATE.value)
    check.add_argument("--json", action="store_true", dest="json_output")
    check.add_argument("--fail-on-breaking", action="store_true")

    compare = sub.add_parser("compare", help="compare two saved public API manifests")
    compare.add_argument("previous", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--json", action="store_true", dest="json_output")
    compare.add_argument("--fail-on-breaking", action="store_true")
    return parser


def _emit(report, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"{report.previous_version} -> {report.current_version}: {'compatible' if report.compatible else 'BREAKING'}")
    if not report.issues:
        print("no public API differences")
    for issue in report.issues:
        print(f"[{issue.severity.value}] {issue.code} {issue.symbol}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        save_public_api_manifest(build_public_api_manifest(stability=ApiStability(args.stability)), args.path)
        print(args.path)
        return 0
    if args.command == "check":
        report = compare_public_api_manifests(
            load_public_api_manifest(args.path),
            build_public_api_manifest(stability=ApiStability(args.stability)),
        )
        _emit(report, json_output=args.json_output)
        return 2 if args.fail_on_breaking and not report.compatible else 0
    if args.command == "compare":
        report = compare_public_api_manifests(
            load_public_api_manifest(args.previous),
            load_public_api_manifest(args.current),
        )
        _emit(report, json_output=args.json_output)
        return 2 if args.fail_on_breaking and not report.compatible else 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
