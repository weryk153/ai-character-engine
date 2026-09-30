from __future__ import annotations

from pathlib import Path

from ai_character_engine.compatibility import (
    ContractSurface,
    ContractVersionRequirement,
    build_public_api_manifest,
    check_contract_requirements,
    compare_public_api_manifests,
    load_public_api_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests/fixtures/api" / "public_api_v1_candidate.json"


def main() -> None:
    baseline = load_public_api_manifest(BASELINE)
    current = build_public_api_manifest()
    report = compare_public_api_manifests(baseline, current)

    contracts = check_contract_requirements(
        ContractVersionRequirement(ContractSurface.PUBLIC_API, 1),
        ContractVersionRequirement(ContractSurface.EXTENSION_API, 1),
        ContractVersionRequirement(ContractSurface.DISTRIBUTED_PROTOCOL, 1),
    )

    print(f"engine={current.engine_version}")
    print(f"public_symbols={len(current.symbols)}")
    print(f"api_compatible={report.compatible}")
    print(f"contract_matrix_compatible={contracts.compatible}")
    if report.issues:
        for issue in report.issues:
            print(f"{issue.severity.value}: {issue.code}: {issue.symbol}: {issue.message}")

    if not report.compatible or not contracts.compatible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
