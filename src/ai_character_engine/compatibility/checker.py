from __future__ import annotations

from collections.abc import Iterable

from .manifest import manifest_symbol_map
from .models import (
    ApiStability,
    CompatibilityIssue,
    CompatibilityReport,
    CompatibilitySeverity,
    MigrationRule,
    ParameterContract,
    PublicApiManifest,
    PublicApiSymbol,
)


def _migration_map(rules: Iterable[MigrationRule]) -> dict[str, MigrationRule]:
    result: dict[str, MigrationRule] = {}
    for rule in rules:
        if rule.old_path in result:
            raise ValueError(f"duplicate migration rule for {rule.old_path}")
        result[rule.old_path] = rule
    return result


def _parameter_breaks(previous: tuple[ParameterContract, ...], current: tuple[ParameterContract, ...]) -> list[str]:
    """Return source-compatibility reasons without overpromising type compatibility.

    The v0.43 contract deliberately tracks call shape only: names, kinds and
    whether a parameter is required. Annotation changes remain documentation-level
    until the v1 API freeze.
    """

    reasons: list[str] = []
    previous_by_name = {item.name: item for item in previous}
    current_by_name = {item.name: item for item in current}

    for old in previous:
        new = current_by_name.get(old.name)
        if new is None:
            reasons.append(f"removed parameter {old.name!r}")
            continue
        if old.kind != new.kind:
            reasons.append(f"changed parameter kind for {old.name!r}: {old.kind} -> {new.kind}")
        if not old.required and new.required:
            reasons.append(f"optional parameter {old.name!r} became required")

    for new in current:
        if new.name not in previous_by_name and new.required:
            reasons.append(f"added required parameter {new.name!r}")

    old_positional = [item.name for item in previous if item.kind in {"positional_only", "positional_or_keyword"}]
    new_positional = [item.name for item in current if item.kind in {"positional_only", "positional_or_keyword"}]
    if new_positional[: len(old_positional)] != old_positional:
        reasons.append("inserted or reordered positional parameters before the existing positional suffix")
    return reasons


def _version_tuple(value: str) -> tuple[int, ...]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    try:
        return tuple(int(part) for part in core.split("."))
    except ValueError as exc:
        raise ValueError(f"unsupported version string {value!r}") from exc


def compare_public_api_manifests(
    previous: PublicApiManifest,
    current: PublicApiManifest,
    *,
    migrations: Iterable[MigrationRule] = (),
) -> CompatibilityReport:
    if current.contract_version != previous.contract_version:
        issue = CompatibilityIssue(
            CompatibilitySeverity.BREAKING,
            "contract_version_changed",
            "<manifest>",
            f"public API contract changed from {previous.contract_version} to {current.contract_version}",
        )
        return CompatibilityReport(previous.engine_version, current.engine_version, (issue,))

    rules = _migration_map(migrations)
    before = manifest_symbol_map(previous)
    after = manifest_symbol_map(current)
    issues: list[CompatibilityIssue] = []

    for key, old in sorted(before.items()):
        new = after.get(key)
        rule = rules.get(key)
        if new is None:
            removal_allowed = (
                old.stability is ApiStability.DEPRECATED
                and rule is not None
                and rule.remove_in is not None
                and old.deprecated_since == rule.deprecated_in
                and _version_tuple(current.engine_version) >= _version_tuple(rule.remove_in)
            )
            if removal_allowed:
                severity = CompatibilitySeverity.WARNING
                message = (
                    f"deprecated symbol removed after declared window; replacement={rule.replacement!r}; "
                    f"remove_in={rule.remove_in}"
                )
            else:
                severity = CompatibilitySeverity.BREAKING
                message = "public symbol was removed without a completed deprecation/migration window"
                if rule is not None:
                    message += f"; migration points to {rule.replacement!r}"
            issues.append(CompatibilityIssue(severity, "symbol_removed", key, message))
            continue

        if old.kind is not new.kind:
            issues.append(
                CompatibilityIssue(
                    CompatibilitySeverity.BREAKING,
                    "symbol_kind_changed",
                    key,
                    f"symbol kind changed from {old.kind.value} to {new.kind.value}",
                )
            )

        if old.import_path != new.import_path:
            severity = CompatibilitySeverity.WARNING if rule is not None else CompatibilitySeverity.BREAKING
            issues.append(
                CompatibilityIssue(
                    severity,
                    "implementation_path_changed",
                    key,
                    f"implementation path changed from {old.import_path} to {new.import_path}",
                )
            )

        reasons = _parameter_breaks(old.parameters, new.parameters)
        if reasons:
            issues.append(
                CompatibilityIssue(
                    CompatibilitySeverity.BREAKING,
                    "call_shape_changed",
                    key,
                    "; ".join(reasons),
                )
            )

        if old.stability is ApiStability.STABLE and new.stability is not ApiStability.STABLE:
            issues.append(
                CompatibilityIssue(
                    CompatibilitySeverity.BREAKING,
                    "stability_downgraded",
                    key,
                    f"stable symbol changed to {new.stability.value}",
                )
            )
        elif old.stability is ApiStability.CANDIDATE and new.stability is ApiStability.DEPRECATED:
            issues.append(
                CompatibilityIssue(
                    CompatibilitySeverity.WARNING,
                    "symbol_deprecated",
                    key,
                    f"candidate symbol deprecated since {new.deprecated_since}",
                )
            )

    for key in sorted(after.keys() - before.keys()):
        issues.append(
            CompatibilityIssue(
                CompatibilitySeverity.INFO,
                "symbol_added",
                key,
                "new public symbol added",
            )
        )

    return CompatibilityReport(previous.engine_version, current.engine_version, tuple(issues))
