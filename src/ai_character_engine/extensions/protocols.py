from __future__ import annotations

from typing import Protocol

from .models import PluginManifest


class ExtensionPlugin(Protocol):
    @property
    def manifest(self) -> PluginManifest: ...

    def register(self, registrar: "ExtensionRegistrar") -> None: ...


# Avoid importing the concrete registrar at runtime and creating a circular import.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .registry import ExtensionRegistrar
