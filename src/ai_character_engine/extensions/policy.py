from __future__ import annotations

from dataclasses import dataclass

from .errors import PluginPolicyError
from .models import ExtensionPoint, PluginCapability, PluginManifest


@dataclass(frozen=True, slots=True)
class PluginActivationPolicy:
    """Optional host-side allowlist.

    Activating a Python plugin already executes trusted third-party code. This
    policy is therefore an engine contract gate, not a process sandbox.
    """

    allowed_capabilities: frozenset[PluginCapability] | None = None
    allowed_extension_points: frozenset[ExtensionPoint] | None = None
    allowed_plugin_ids: frozenset[str] | None = None

    def validate(self, manifest: PluginManifest) -> None:
        if self.allowed_plugin_ids is not None and manifest.plugin_id not in self.allowed_plugin_ids:
            raise PluginPolicyError(f"plugin is not allowlisted: {manifest.plugin_id}")
        if self.allowed_capabilities is not None:
            denied = set(manifest.capabilities) - set(self.allowed_capabilities)
            if denied:
                names = ", ".join(sorted(item.value for item in denied))
                raise PluginPolicyError(f"plugin capabilities are denied: {names}")
        if self.allowed_extension_points is not None:
            denied_points = set(manifest.extension_points) - set(self.allowed_extension_points)
            if denied_points:
                names = ", ".join(sorted(item.value for item in denied_points))
                raise PluginPolicyError(f"plugin extension points are denied: {names}")
