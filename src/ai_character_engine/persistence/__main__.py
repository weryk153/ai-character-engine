from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fixtures import verify_persistence_fixture_directory
from .io import PersistenceFileFormat, migrate_persistence_file
from .models import PersistenceSurface
from .registry import CURRENT_PERSISTENCE_SCHEMA_VERSIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistence migration and replay verification tools")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate", help="migrate a JSON/JSONL artifact to the current schema")
    migrate.add_argument("input")
    migrate.add_argument("output")
    migrate.add_argument("--format", choices=[item.value for item in PersistenceFileFormat], required=True)
    migrate.add_argument(
        "--surface",
        choices=["auto", *(item.value for item in PersistenceSurface)],
        required=True,
        help="use auto only for records with a stable record_type",
    )

    verify = sub.add_parser("verify-fixtures", help="verify deterministic cross-version replay fixtures")
    verify.add_argument("path")
    verify.add_argument("--fail-on-violations", action="store_true")

    sub.add_parser("schema-versions", help="print current persistence schema versions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "schema-versions":
        print(
            json.dumps(
                {surface.value: version for surface, version in CURRENT_PERSISTENCE_SCHEMA_VERSIONS.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "migrate":
        surface = None if args.surface == "auto" else PersistenceSurface(args.surface)
        count = migrate_persistence_file(
            args.input,
            args.output,
            file_format=PersistenceFileFormat(args.format),
            surface=surface,
        )
        print(json.dumps({"migrated_records": count, "output": str(Path(args.output))}, indent=2))
        return 0
    results = verify_persistence_fixture_directory(args.path)
    payload = {
        "passed": all(result.passed for result in results),
        "fixtures": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_violations and not payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
