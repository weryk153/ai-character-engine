from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from ai_character_engine._version import VERSION
from ai_character_engine.distributed import DISTRIBUTED_PROTOCOL_VERSION
from ai_character_engine.extensions import EXTENSION_API_VERSION
from ai_character_engine.persistence import PERSISTENCE_SCHEMA_CONTRACT_VERSION

from .models import PUBLIC_API_CONTRACT_VERSION


class ContractSurface(str, Enum):
    PUBLIC_API = "public_api"
    EXTENSION_API = "extension_api"
    DISTRIBUTED_PROTOCOL = "distributed_protocol"
    PERSISTENCE_SCHEMA = "persistence_schema"


@dataclass(frozen=True, slots=True)
class ContractVersionRequirement:
    surface: ContractSurface
    version: int

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("contract version must be > 0")


@dataclass(frozen=True, slots=True)
class ContractCompatibilityResult:
    engine_version: str
    supported: Mapping[ContractSurface, int]
    mismatches: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.mismatches


def current_contract_versions() -> dict[ContractSurface, int]:
    return {
        ContractSurface.PUBLIC_API: PUBLIC_API_CONTRACT_VERSION,
        ContractSurface.EXTENSION_API: EXTENSION_API_VERSION,
        ContractSurface.DISTRIBUTED_PROTOCOL: DISTRIBUTED_PROTOCOL_VERSION,
        ContractSurface.PERSISTENCE_SCHEMA: PERSISTENCE_SCHEMA_CONTRACT_VERSION,
    }


def check_contract_requirements(
    *requirements: ContractVersionRequirement,
) -> ContractCompatibilityResult:
    supported = current_contract_versions()
    mismatches = tuple(
        f"{requirement.surface.value}: required {requirement.version}, supported {supported[requirement.surface]}"
        for requirement in requirements
        if supported[requirement.surface] != requirement.version
    )
    return ContractCompatibilityResult(VERSION, supported, mismatches)
