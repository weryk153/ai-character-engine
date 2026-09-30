from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_character_engine._version import VERSION

from .errors import PluginCompatibilityError, PluginRegistrationError
from .models import EXTENSION_API_VERSION, PluginActivationRecord, PluginManifest
from .policy import PluginActivationPolicy
from .protocols import ExtensionPlugin
from .registry import ExtensionRegistrar, ExtensionRegistry


class PluginManager:
    """Explicit plugin activation and lifecycle manager.

    It coordinates extension registration only. It deliberately never receives
    CharacterRuntime, MemoryManager, GoalManager, WorldRuntime or commit-authority
    objects, so activation cannot become a shortcut around those boundaries.
    """

    def __init__(self, registry: ExtensionRegistry | None = None, *, engine_version: str = VERSION) -> None:
        self.registry = registry or ExtensionRegistry()
        self.engine_version = engine_version
        self._active: dict[str, PluginActivationRecord] = {}

    def active_plugins(self) -> tuple[PluginActivationRecord, ...]:
        return tuple(sorted(self._active.values(), key=lambda item: item.plugin_id))

    def activation(self, plugin_id: str) -> PluginActivationRecord | None:
        return self._active.get(plugin_id)

    def activate(
        self,
        plugin: ExtensionPlugin,
        *,
        policy: PluginActivationPolicy | None = None,
    ) -> PluginActivationRecord:
        manifest = _manifest_of(plugin)
        if manifest.api_version != EXTENSION_API_VERSION:
            raise PluginCompatibilityError(
                f"plugin {manifest.plugin_id} targets extension API {manifest.api_version}; "
                f"engine supports {EXTENSION_API_VERSION}"
            )
        if manifest.plugin_id in self._active:
            raise PluginRegistrationError(f"plugin already activated: {manifest.plugin_id}")
        if policy is not None:
            policy.validate(manifest)

        registrar = ExtensionRegistrar(manifest)
        plugin.register(registrar)
        extensions = registrar._seal()
        registered_points = {item.point for item in extensions}
        undeclared_unused = set(manifest.extension_points) - registered_points
        # Declaring a point without contributing one is legal for optional/config-driven plugins.
        _ = undeclared_unused

        # Atomic commit: registry mutation occurs only after register() returns successfully.
        self.registry._commit(manifest.plugin_id, extensions)
        record = PluginActivationRecord(
            plugin_id=manifest.plugin_id,
            plugin_version=manifest.version,
            api_version=manifest.api_version,
            extension_ids=tuple(item.extension_id for item in extensions),
            capabilities=manifest.capabilities,
        )
        self._active[manifest.plugin_id] = record
        return record

    def deactivate(self, plugin_id: str) -> PluginActivationRecord | None:
        record = self._active.pop(plugin_id, None)
        if record is None:
            return None
        self.registry.deactivate(plugin_id)
        return record


def _manifest_of(plugin: ExtensionPlugin) -> PluginManifest:
    manifest = getattr(plugin, "manifest", None)
    if not isinstance(manifest, PluginManifest):
        raise PluginRegistrationError("plugin.manifest must be a PluginManifest")
    if not callable(getattr(plugin, "register", None)):
        raise PluginRegistrationError("plugin must define register(registrar)")
    return manifest
