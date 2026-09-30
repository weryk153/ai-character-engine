from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_character_engine.collaboration.models import SpecialistSpec
from ai_character_engine.llm.base import LLMClient
from ai_character_engine.memory.embedding import AsyncEmbeddingProvider, EmbeddingProvider
from ai_character_engine.observability.sink import ObservabilitySink
from ai_character_engine.tools.registry import ToolRegistry
from ai_character_engine.vision.base import VisionProvider
from ai_character_engine.world.perception import WorldPerceptionPolicy

from .errors import ExtensionNotFoundError, ExtensionValidationError, PluginRegistrationError
from .models import (
    ExtensionFactoryContext,
    ExtensionPoint,
    POINT_CAPABILITY,
    PluginManifest,
    RegisteredExtension,
    ToolExtension,
)

Factory = Callable[[ExtensionFactoryContext], Any]


@dataclass(frozen=True, slots=True)
class _StagedRegistration:
    extension: RegisteredExtension


class ExtensionRegistrar:
    """Per-plugin staging registrar.

    Registrations are invisible to the shared registry until PluginManager
    validates and atomically commits the whole batch.
    """

    def __init__(self, manifest: PluginManifest) -> None:
        self.manifest = manifest
        self._staged: list[_StagedRegistration] = []
        self._sealed = False

    def _register(
        self,
        point: ExtensionPoint,
        name: str,
        factory: Factory,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._sealed:
            raise PluginRegistrationError("registrar is sealed")
        if point not in self.manifest.extension_points:
            raise PluginRegistrationError(
                f"plugin did not declare extension point: {point.value}"
            )
        required = POINT_CAPABILITY[point]
        if required not in self.manifest.capabilities:
            raise PluginRegistrationError(
                f"plugin did not declare capability: {required.value}"
            )
        extension = RegisteredExtension(
            plugin_id=self.manifest.plugin_id,
            local_name=name.strip(),
            point=point,
            factory=factory,
            metadata=metadata or {},
        )
        if any(item.extension.extension_id == extension.extension_id for item in self._staged):
            raise PluginRegistrationError(
                f"duplicate extension id in plugin: {extension.extension_id}"
            )
        self._staged.append(_StagedRegistration(extension))

    def register_llm_client(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.LLM_CLIENT, name, factory, metadata=metadata)

    def register_embedding_provider(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.EMBEDDING_PROVIDER, name, factory, metadata=metadata)

    def register_vision_provider(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.VISION_PROVIDER, name, factory, metadata=metadata)

    def register_observability_sink(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.OBSERVABILITY_SINK, name, factory, metadata=metadata)

    def register_world_perception_policy(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.WORLD_PERCEPTION_POLICY, name, factory, metadata=metadata)

    def register_cognitive_specialist(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.COGNITIVE_SPECIALIST, name, factory, metadata=metadata)

    def register_tool(self, name: str, factory: Factory, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._register(ExtensionPoint.TOOL, name, factory, metadata=metadata)

    def _seal(self) -> tuple[RegisteredExtension, ...]:
        self._sealed = True
        return tuple(item.extension for item in self._staged)


class ExtensionRegistry:
    def __init__(self) -> None:
        self._extensions: dict[str, RegisteredExtension] = {}
        self._by_plugin: dict[str, tuple[str, ...]] = {}

    def _commit(self, plugin_id: str, extensions: tuple[RegisteredExtension, ...]) -> None:
        if plugin_id in self._by_plugin:
            raise PluginRegistrationError(f"plugin already activated: {plugin_id}")
        duplicates = [ext.extension_id for ext in extensions if ext.extension_id in self._extensions]
        if duplicates:
            raise PluginRegistrationError(
                f"extension ids already registered: {', '.join(sorted(duplicates))}"
            )
        ids = tuple(ext.extension_id for ext in extensions)
        # All validation above happens before either mapping is mutated.
        self._extensions.update({ext.extension_id: ext for ext in extensions})
        self._by_plugin[plugin_id] = ids

    def deactivate(self, plugin_id: str) -> tuple[str, ...]:
        ids = self._by_plugin.pop(plugin_id, ())
        for extension_id in ids:
            self._extensions.pop(extension_id, None)
        return ids

    def plugin_extension_ids(self, plugin_id: str) -> tuple[str, ...]:
        return self._by_plugin.get(plugin_id, ())

    def list(self, point: ExtensionPoint | None = None) -> tuple[RegisteredExtension, ...]:
        items = self._extensions.values()
        if point is not None:
            items = (item for item in items if item.point is point)
        return tuple(sorted(items, key=lambda item: item.extension_id))

    def get(self, extension_id: str) -> RegisteredExtension:
        try:
            return self._extensions[extension_id]
        except KeyError as exc:
            raise ExtensionNotFoundError(f"unknown extension: {extension_id}") from exc

    def create(self, extension_id: str, *, config: Mapping[str, Any] | None = None) -> Any:
        extension = self.get(extension_id)
        context = ExtensionFactoryContext(
            plugin_id=extension.plugin_id,
            config=config or {},
        )
        try:
            value = extension.factory(context)
        except Exception:
            # Factory failure is intentionally local: registry/activation state is unchanged.
            raise
        self._validate_value(extension.point, value)
        return value

    def create_llm_client(self, extension_id: str, *, config: Mapping[str, Any] | None = None) -> LLMClient:
        return self._create_typed(extension_id, ExtensionPoint.LLM_CLIENT, config=config)

    def create_embedding_provider(
        self, extension_id: str, *, config: Mapping[str, Any] | None = None
    ) -> EmbeddingProvider | AsyncEmbeddingProvider:
        return self._create_typed(extension_id, ExtensionPoint.EMBEDDING_PROVIDER, config=config)

    def create_vision_provider(self, extension_id: str, *, config: Mapping[str, Any] | None = None) -> VisionProvider:
        return self._create_typed(extension_id, ExtensionPoint.VISION_PROVIDER, config=config)

    def create_observability_sink(self, extension_id: str, *, config: Mapping[str, Any] | None = None) -> ObservabilitySink:
        return self._create_typed(extension_id, ExtensionPoint.OBSERVABILITY_SINK, config=config)

    def create_world_perception_policy(
        self, extension_id: str, *, config: Mapping[str, Any] | None = None
    ) -> WorldPerceptionPolicy:
        return self._create_typed(extension_id, ExtensionPoint.WORLD_PERCEPTION_POLICY, config=config)

    def create_cognitive_specialist(
        self, extension_id: str, *, config: Mapping[str, Any] | None = None
    ) -> SpecialistSpec:
        return self._create_typed(extension_id, ExtensionPoint.COGNITIVE_SPECIALIST, config=config)

    def create_tool(self, extension_id: str, *, config: Mapping[str, Any] | None = None) -> ToolExtension:
        return self._create_typed(extension_id, ExtensionPoint.TOOL, config=config)

    def install_tool(
        self,
        tool_registry: ToolRegistry,
        extension_id: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolExtension:
        """Explicitly grants one contributed tool access to ToolRegistry.

        Merely activating a plugin never installs or executes a tool.
        """
        tool = self.create_tool(extension_id, config=config)
        tool_registry.register(tool.definition, tool.handler)
        return tool

    def _create_typed(
        self,
        extension_id: str,
        point: ExtensionPoint,
        *,
        config: Mapping[str, Any] | None,
    ) -> Any:
        extension = self.get(extension_id)
        if extension.point is not point:
            raise ExtensionValidationError(
                f"extension {extension_id} is {extension.point.value}, not {point.value}"
            )
        return self.create(extension_id, config=config)

    @staticmethod
    def _validate_value(point: ExtensionPoint, value: Any) -> None:
        if point is ExtensionPoint.LLM_CLIENT:
            _require_callable(value, "generate", "LLM client")
        elif point is ExtensionPoint.EMBEDDING_PROVIDER:
            if not isinstance(getattr(value, "model_id", None), str):
                raise ExtensionValidationError("embedding provider requires string model_id")
            dimensions = getattr(value, "dimensions", None)
            if not isinstance(dimensions, int) or dimensions < 1:
                raise ExtensionValidationError("embedding provider requires positive dimensions")
            if not any(callable(getattr(value, name, None)) for name in ("aembed_many", "embed_many", "embed")):
                raise ExtensionValidationError("embedding provider requires an embedding method")
        elif point is ExtensionPoint.VISION_PROVIDER:
            _require_callable(value, "analyze", "vision provider")
        elif point is ExtensionPoint.OBSERVABILITY_SINK:
            for name in ("record_span", "record_turn", "record_eval"):
                _require_callable(value, name, "observability sink")
        elif point is ExtensionPoint.WORLD_PERCEPTION_POLICY:
            if not isinstance(getattr(value, "name", None), str) or not value.name.strip():
                raise ExtensionValidationError("world perception policy requires a non-empty name")
            _require_callable(value, "project", "world perception policy")
        elif point is ExtensionPoint.COGNITIVE_SPECIALIST:
            if not isinstance(value, SpecialistSpec):
                raise ExtensionValidationError("cognitive specialist extension must return SpecialistSpec")
        elif point is ExtensionPoint.TOOL:
            if not isinstance(value, ToolExtension):
                raise ExtensionValidationError("tool extension must return ToolExtension")
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ExtensionValidationError(f"unsupported extension point: {point}")


def _require_callable(value: Any, attr: str, label: str) -> None:
    if not callable(getattr(value, attr, None)):
        raise ExtensionValidationError(f"{label} requires callable {attr}()")
