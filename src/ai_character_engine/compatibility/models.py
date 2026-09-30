from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


PUBLIC_API_CONTRACT_VERSION = 1


class ApiStability(str, Enum):
    """Lifecycle status for a declared public API surface."""

    CANDIDATE = "candidate"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class ApiSymbolKind(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    CONSTANT = "constant"
    OTHER = "other"


class CompatibilitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BREAKING = "breaking"


@dataclass(frozen=True, slots=True)
class ParameterContract:
    name: str
    kind: str
    required: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name is required")
        if not self.kind:
            raise ValueError("parameter kind is required")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "required": self.required}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterContract":
        return cls(name=str(value["name"]), kind=str(value["kind"]), required=bool(value["required"]))


@dataclass(frozen=True, slots=True)
class PublicApiSymbol:
    module: str
    name: str
    import_path: str
    kind: ApiSymbolKind
    parameters: tuple[ParameterContract, ...] = ()
    stability: ApiStability = ApiStability.CANDIDATE
    deprecated_since: str | None = None
    replacement: str | None = None

    def __post_init__(self) -> None:
        if not self.module or not self.name or not self.import_path:
            raise ValueError("module, name and import_path are required")
        if self.name.startswith("_"):
            raise ValueError("private symbols cannot enter the public API manifest")
        if self.stability is ApiStability.DEPRECATED and not self.deprecated_since:
            raise ValueError("deprecated symbols require deprecated_since")
        if self.stability is not ApiStability.DEPRECATED and self.deprecated_since is not None:
            raise ValueError("deprecated_since is only valid for deprecated symbols")

    @property
    def key(self) -> str:
        return f"{self.module}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "import_path": self.import_path,
            "kind": self.kind.value,
            "parameters": [item.to_dict() for item in self.parameters],
            "stability": self.stability.value,
            "deprecated_since": self.deprecated_since,
            "replacement": self.replacement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicApiSymbol":
        return cls(
            module=str(value["module"]),
            name=str(value["name"]),
            import_path=str(value["import_path"]),
            kind=ApiSymbolKind(str(value["kind"])),
            parameters=tuple(ParameterContract.from_dict(item) for item in value.get("parameters", ())),
            stability=ApiStability(str(value.get("stability", ApiStability.CANDIDATE.value))),
            deprecated_since=value.get("deprecated_since"),
            replacement=value.get("replacement"),
        )


@dataclass(frozen=True, slots=True)
class PublicApiManifest:
    engine_version: str
    contract_version: int
    symbols: tuple[PublicApiSymbol, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_version:
            raise ValueError("engine_version is required")
        if self.contract_version <= 0:
            raise ValueError("contract_version must be > 0")
        keys = [item.key for item in self.symbols]
        if len(keys) != len(set(keys)):
            raise ValueError("public API manifest contains duplicate symbols")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "contract_version": self.contract_version,
            "symbols": [item.to_dict() for item in self.symbols],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicApiManifest":
        return cls(
            engine_version=str(value["engine_version"]),
            contract_version=int(value["contract_version"]),
            symbols=tuple(PublicApiSymbol.from_dict(item) for item in value.get("symbols", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MigrationRule:
    old_path: str
    replacement: str | None
    deprecated_in: str
    remove_in: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.old_path or ":" not in self.old_path:
            raise ValueError("old_path must use '<module>:<symbol>' form")
        if self.replacement is not None and ":" not in self.replacement:
            raise ValueError("replacement must use '<module>:<symbol>' form")
        if not self.deprecated_in:
            raise ValueError("deprecated_in is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_path": self.old_path,
            "replacement": self.replacement,
            "deprecated_in": self.deprecated_in,
            "remove_in": self.remove_in,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    severity: CompatibilitySeverity
    code: str
    symbol: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "symbol": self.symbol,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    previous_version: str
    current_version: str
    issues: tuple[CompatibilityIssue, ...]

    @property
    def breaking(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(item for item in self.issues if item.severity is CompatibilitySeverity.BREAKING)

    @property
    def warnings(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(item for item in self.issues if item.severity is CompatibilitySeverity.WARNING)

    @property
    def compatible(self) -> bool:
        return not self.breaking

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_version": self.previous_version,
            "current_version": self.current_version,
            "compatible": self.compatible,
            "issues": [item.to_dict() for item in self.issues],
        }
