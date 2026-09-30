from __future__ import annotations

from ai_character_engine.extensions import (
    ExtensionPoint,
    PluginCapability,
    PluginManager,
    PluginManifest,
    ToolExtension,
)
from ai_character_engine.observability import InMemoryObservabilitySink
from ai_character_engine.tools import ToolDefinition, ToolRegistry


class OfflinePlugin:
    manifest = PluginManifest(
        plugin_id="example.offline",
        name="Offline Extension Example",
        version="1.0.0",
        extension_points=(ExtensionPoint.OBSERVABILITY_SINK, ExtensionPoint.TOOL),
        capabilities=(PluginCapability.OBSERVABILITY, PluginCapability.TOOL_EXECUTION),
    )

    def register(self, registrar) -> None:
        registrar.register_observability_sink(
            "telemetry",
            lambda ctx: InMemoryObservabilitySink(),
        )
        registrar.register_tool(
            "echo",
            lambda ctx: ToolExtension(
                ToolDefinition(
                    name="offline_echo",
                    description="Echo a text value.",
                    parameters={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                ),
                lambda text: f"echo:{text}",
            ),
        )


manager = PluginManager()
record = manager.activate(OfflinePlugin())
print(f"activated: {record.plugin_id}")
print(f"extension: {manager.registry.list(ExtensionPoint.OBSERVABILITY_SINK)[0].extension_id}")

sink = manager.registry.create_observability_sink("example.offline:telemetry")
assert isinstance(sink, InMemoryObservabilitySink)

# Activating the plugin does not install executable tools into the engine.
tools = ToolRegistry()
print(f"tools before explicit install: {len(tools)}")
manager.registry.install_tool(tools, "example.offline:echo")
print(f"tools after explicit install: {len(tools)}")
assert tools.get("offline_echo").handler(text="hello") == "echo:hello"
