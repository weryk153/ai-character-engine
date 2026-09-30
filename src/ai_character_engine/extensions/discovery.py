from __future__ import annotations

from importlib import metadata
from typing import Any

from .errors import PluginRegistrationError
from .models import DiscoveredPlugin, ENTRY_POINT_GROUP


def discover_plugins(*, group: str = ENTRY_POINT_GROUP) -> tuple[DiscoveredPlugin, ...]:
    """Discover metadata only; this does not import or activate plugin code."""
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, ())
    discovered: list[DiscoveredPlugin] = []
    for entry_point in selected:
        distribution = getattr(getattr(entry_point, "dist", None), "name", None)
        discovered.append(
            DiscoveredPlugin(
                name=entry_point.name,
                value=entry_point.value,
                group=entry_point.group,
                distribution=distribution,
            )
        )
    return tuple(sorted(discovered, key=lambda item: (item.name, item.value)))


def load_plugin(name: str, *, group: str = ENTRY_POINT_GROUP) -> Any:
    """Explicitly import one discovered entry point.

    Loading Python plugin code is a host trust decision. The SDK does not claim
    to sandbox arbitrary Python packages.
    """
    matches = []
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, ())
    for entry_point in selected:
        if entry_point.name == name:
            matches.append(entry_point)
    if not matches:
        raise PluginRegistrationError(f"plugin entry point not found: {name}")
    if len(matches) > 1:
        raise PluginRegistrationError(f"multiple plugin entry points named: {name}")
    loaded = matches[0].load()
    candidate = loaded
    if isinstance(loaded, type):
        candidate = loaded()
    elif not hasattr(loaded, "manifest") and callable(loaded):
        candidate = loaded()
    if not hasattr(candidate, "manifest") or not callable(getattr(candidate, "register", None)):
        raise PluginRegistrationError(
            f"entry point {name} did not produce an ExtensionPlugin"
        )
    return candidate
