from __future__ import annotations

from pathlib import Path

from ai_character_engine.release import check_release_readiness, default_release_matrix

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    matrix = default_release_matrix()
    report = check_release_readiness(ROOT)
    print(f"declared targets={len(matrix.targets)}")
    for target in matrix.targets:
        print(f"- {target.platform} / Python {target.python_version}")
    print(f"candidate readiness passed={report.passed}")
    for warning in report.warnings:
        print(f"warning: {warning.name}: {warning.detail}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
