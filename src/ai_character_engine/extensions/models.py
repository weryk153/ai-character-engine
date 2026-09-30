from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ai_character_engine._version import VERSION
from ai_character_engine.tools.models import ToolDefinition

EXTENSION_API_VERSION = 1
ENTRY_POINT_GROUP = "ai_character_engine.plugins"

_PLUGIN_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_EXTENSION_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")


class ExtensionPoint(str, Enum):
    LLM_CLIENT = "llm_client"
    EMBEDDING_PROVIDER = "embedding_provider"
    VISION_PROVIDER = "vision_provider"
    OBSERVABILITY_SINK = "observability_sink"
    WORLD_PERCEPTION_POLICY = "world_perception_policy"
    COGNITIVE_SPECIALIST = "cognitive_specialist"
    TOOL = "tool"


class PluginCapability(str, Enum):
    MODEL_IO = "model_io"
    OBSERVABILITY = "observability"
    WORLD_PERCEPTION = "world_perception"
    COGNITIVE_SPECIALIST = "cognitive_specialist"
    TOOL_EXECUTION = "tool_execution"


POINT_CAPABILITY: dict[ExtensionPoint, PluginCapability] = {
    ExtensionPoint.LLM_CLIENT: PluginCapability.MODEL_IO,
    ExtensionPoint.EMBEDDING_PROVIDER: PluginCapability.MODEL_IO,
    ExtensionPoint.VISION_PROVIDER: PluginCapability.MODEL_IO,
    ExtensionPoint.OBSERVABILITY_SINK: PluginCapability.OBSERVABILITY,
    ExtensionPoint.WORLD_PERCEPTION_POLICY: PluginCapability.WORLD_PERCEPTION,
    ExtensionPoint.COGNITIVE_SPECIALIST: PluginCapability.COGNITIVE_SPECIALIST,
    ExtensionPoint.TOOL: PluginCapability.TOOL_EXECUTION,
}


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    extension_points: tuple[ExtensionPoint, ...]
    capabilities: tuple[PluginCapability, ...]
    api_version: int = EXTENSION_API_VERSION
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plugin_id = self.plugin_id.strip()
        name = self.name.strip()
        version = self.version.strip()
        if not _PLUGIN_ID_RE.fullmatch(plugin_id):
            raise ValueError("plugin_id must use lowercase letters, digits, '.', '_' or '-'")
        if not name:
            raise ValueError("plugin name must not be empty")
        if not version:
            raise ValueError("plugin version must not be empty")
        if self.api_version < 1:
            raise ValueError("api_version must be >= 1")
        if len(set(self.extension_points)) != len(self.extension_points):
            raise ValueError("extension_points must be unique")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must be unique")
        declared = set(self.capabilities)
        missing = {
            POINT_CAPABILITY[point]
            for point in self.extension_points
            if POINT_CAPABILITY[point] not in declared
        }
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"manifest is missing required capabilities: {names}")
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))


@dataclass(frozen=True, slots=True)
class ExtensionFactoryContext:
    plugin_id: str
    engine_version: str = VERSION
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id must not be empty")
        if not self.engine_version.strip():
            raise ValueError("engine_version must not be empty")
        object.__setattr__(self, "config", MappingProxyType(copy.deepcopy(dict(self.config))))


@dataclass(frozen=True, slots=True)
class ToolExtension:
    definition: ToolDefinition
    handler: Callable[..., Any]

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")


@dataclass(frozen=True, slots=True)
class RegisteredExtension:
    plugin_id: str
    local_name: str
    point: ExtensionPoint
    factory: Callable[[ExtensionFactoryContext], Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PLUGIN_ID_RE.fullmatch(self.plugin_id):
            raise ValueError("invalid plugin_id")
        if not _EXTENSION_NAME_RE.fullmatch(self.local_name):
            raise ValueError("extension name must use lowercase letters, digits, '.', '_' or '-'")
        if not callable(self.factory):
            raise TypeError("extension factory must be callable")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    @property
    def extension_id(self) -> str:
        return f"{self.plugin_id}:{self.local_name}"


@dataclass(frozen=True, slots=True)
class PluginActivationRecord:
    plugin_id: str
    plugin_version: str
    api_version: int
    extension_ids: tuple[str, ...]
    capabilities: tuple[PluginCapability, ...]

    def __post_init__(self) -> None:
        if len(set(self.extension_ids)) != len(self.extension_ids):
            raise ValueError("activation extension ids must be unique")


@dataclass(frozen=True, slots=True)
class DiscoveredPlugin:
    name: str
    value: str
    group: str = ENTRY_POINT_GROUP
    distribution: str | None = None
