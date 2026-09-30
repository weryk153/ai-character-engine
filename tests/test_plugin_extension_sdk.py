from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_character_engine.collaboration import SpecialistKind, SpecialistSpec
from ai_character_engine.cognition import CognitiveRole
from ai_character_engine.extensions import (
    ENTRY_POINT_GROUP,
    EXTENSION_API_VERSION,
    DiscoveredPlugin,
    ExtensionFactoryContext,
    ExtensionNotFoundError,
    ExtensionPoint,
    ExtensionRegistry,
    ExtensionValidationError,
    PluginActivationPolicy,
    PluginCapability,
    PluginCompatibilityError,
    PluginManager,
    PluginManifest,
    PluginPolicyError,
    PluginRegistrationError,
    ToolExtension,
    discover_plugins,
    load_plugin,
)
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.observability import InMemoryObservabilitySink
from ai_character_engine.tools import ToolDefinition, ToolRegistry
from ai_character_engine.vision import VisionAnalysis
from ai_character_engine.world import WorldPerceptionProjection


class FakeLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="ok", model="plugin/fake")


class FakeEmbedding:
    model_id = "plugin/embed"
    dimensions = 2

    async def aembed_many(self, texts):
        return tuple((1.0, 0.0) for _ in texts)


class FakeVision:
    async def analyze(self, image, *, prompt=None):
        return VisionAnalysis(text="scene", provider="plugin", model="vision")


class FakeWorldPolicy:
    name = "plugin-policy"

    def project(self, *, character_id, event):
        return WorldPerceptionProjection(content=f"{character_id} saw event", fact_keys=())


class BasicPlugin:
    manifest = PluginManifest(
        plugin_id="example.basic",
        name="Example Basic",
        version="1.2.3",
        extension_points=(
            ExtensionPoint.LLM_CLIENT,
            ExtensionPoint.EMBEDDING_PROVIDER,
            ExtensionPoint.VISION_PROVIDER,
            ExtensionPoint.OBSERVABILITY_SINK,
            ExtensionPoint.WORLD_PERCEPTION_POLICY,
            ExtensionPoint.COGNITIVE_SPECIALIST,
        ),
        capabilities=(
            PluginCapability.MODEL_IO,
            PluginCapability.OBSERVABILITY,
            PluginCapability.WORLD_PERCEPTION,
            PluginCapability.COGNITIVE_SPECIALIST,
        ),
        metadata={"homepage": "https://example.invalid"},
    )

    def register(self, registrar):
        registrar.register_llm_client("chat", lambda ctx: FakeLLM())
        registrar.register_embedding_provider("embed", lambda ctx: FakeEmbedding())
        registrar.register_vision_provider("vision", lambda ctx: FakeVision())
        registrar.register_observability_sink("telemetry", lambda ctx: InMemoryObservabilitySink())
        registrar.register_world_perception_policy("perception", lambda ctx: FakeWorldPolicy())
        registrar.register_cognitive_specialist(
            "lore",
            lambda ctx: SpecialistSpec(
                specialist_id="lore",
                kind=SpecialistKind.MEMORY,
                role=CognitiveRole.MEMORY,
                purpose="Review supplied lore evidence.",
                system_prompt="Use only supplied lore evidence.",
            ),
        )


class ToolPlugin:
    manifest = PluginManifest(
        plugin_id="example.tools",
        name="Example Tools",
        version="1.0.0",
        extension_points=(ExtensionPoint.TOOL,),
        capabilities=(PluginCapability.TOOL_EXECUTION,),
    )

    def register(self, registrar):
        registrar.register_tool(
            "echo",
            lambda ctx: ToolExtension(
                ToolDefinition(
                    name="plugin_echo",
                    description="Echo text.",
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


def test_extension_api_version_is_explicit_and_independent_from_engine_semver():
    assert EXTENSION_API_VERSION == 1


def test_entry_point_group_is_stable_and_namespaced():
    assert ENTRY_POINT_GROUP == "ai_character_engine.plugins"


def test_extension_points_are_bounded_typed_seams_not_generic_hooks():
    assert [item.value for item in ExtensionPoint] == [
        "llm_client",
        "embedding_provider",
        "vision_provider",
        "observability_sink",
        "world_perception_policy",
        "cognitive_specialist",
        "tool",
    ]
    assert "pre_turn" not in {item.value for item in ExtensionPoint}
    assert "post_turn" not in {item.value for item in ExtensionPoint}
    assert "state_mutator" not in {item.value for item in ExtensionPoint}


def test_manifest_requires_lowercase_stable_plugin_id():
    with pytest.raises(ValueError, match="plugin_id"):
        PluginManifest("Bad Plugin", "Bad", "1", (), ())


def test_manifest_rejects_duplicate_points_and_capabilities():
    with pytest.raises(ValueError, match="extension_points"):
        PluginManifest(
            "x.plugin", "X", "1",
            (ExtensionPoint.OBSERVABILITY_SINK, ExtensionPoint.OBSERVABILITY_SINK),
            (PluginCapability.OBSERVABILITY,),
        )
    with pytest.raises(ValueError, match="capabilities"):
        PluginManifest(
            "x.plugin", "X", "1", (),
            (PluginCapability.OBSERVABILITY, PluginCapability.OBSERVABILITY),
        )


def test_manifest_requires_capability_for_each_declared_extension_point():
    with pytest.raises(ValueError, match="missing required capabilities"):
        PluginManifest(
            "x.plugin", "X", "1",
            (ExtensionPoint.TOOL,),
            (),
        )


def test_manifest_metadata_is_defensively_copied_and_read_only():
    meta = {"nested": {"x": 1}}
    manifest = PluginManifest("x.plugin", "X", "1", (), (), metadata=meta)
    meta["nested"]["x"] = 2
    assert manifest.metadata["nested"]["x"] == 1
    with pytest.raises(TypeError):
        manifest.metadata["new"] = 1


def test_factory_context_defensively_copies_plugin_config():
    config = {"nested": {"x": 1}}
    ctx = ExtensionFactoryContext("x.plugin", config=config)
    config["nested"]["x"] = 2
    assert ctx.config["nested"]["x"] == 1
    with pytest.raises(TypeError):
        ctx.config["x"] = 2


def test_basic_plugin_activation_is_atomic_and_namespaced():
    manager = PluginManager()
    record = manager.activate(BasicPlugin())
    assert record.plugin_id == "example.basic"
    assert record.extension_ids == (
        "example.basic:chat",
        "example.basic:embed",
        "example.basic:vision",
        "example.basic:telemetry",
        "example.basic:perception",
        "example.basic:lore",
    )
    assert manager.registry.plugin_extension_ids("example.basic") == record.extension_ids


def test_same_local_extension_name_can_exist_in_different_plugin_namespaces():
    class A:
        manifest = PluginManifest(
            "plugin.a", "A", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            registrar.register_observability_sink("default", lambda ctx: InMemoryObservabilitySink())

    class B:
        manifest = PluginManifest(
            "plugin.b", "B", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            registrar.register_observability_sink("default", lambda ctx: InMemoryObservabilitySink())

    manager = PluginManager()
    manager.activate(A())
    manager.activate(B())
    assert manager.registry.get("plugin.a:default").plugin_id == "plugin.a"
    assert manager.registry.get("plugin.b:default").plugin_id == "plugin.b"


def test_plugin_cannot_register_an_undeclared_extension_point():
    class Bad:
        manifest = PluginManifest(
            "bad.plugin", "Bad", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            registrar.register_llm_client("oops", lambda ctx: FakeLLM())

    manager = PluginManager()
    with pytest.raises(PluginRegistrationError, match="did not declare"):
        manager.activate(Bad())
    assert manager.active_plugins() == ()
    assert manager.registry.list() == ()


def test_duplicate_extension_ids_inside_one_plugin_fail_without_partial_activation():
    class Bad:
        manifest = PluginManifest(
            "bad.duplicate", "Bad", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            registrar.register_observability_sink("same", lambda ctx: InMemoryObservabilitySink())
            registrar.register_observability_sink("same", lambda ctx: InMemoryObservabilitySink())

    manager = PluginManager()
    with pytest.raises(PluginRegistrationError, match="duplicate extension"):
        manager.activate(Bad())
    assert manager.registry.list() == ()


def test_register_exception_does_not_partially_commit_plugin():
    class Explodes:
        manifest = PluginManifest(
            "bad.explodes", "Bad", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            registrar.register_observability_sink("one", lambda ctx: InMemoryObservabilitySink())
            raise RuntimeError("registration failed")

    manager = PluginManager()
    with pytest.raises(RuntimeError, match="registration failed"):
        manager.activate(Explodes())
    assert manager.registry.list() == ()
    assert manager.activation("bad.explodes") is None


def test_api_version_mismatch_is_rejected_before_registration():
    touched = False

    class Old:
        manifest = PluginManifest(
            "old.plugin", "Old", "1", (), (), api_version=EXTENSION_API_VERSION + 1,
        )
        def register(self, registrar):
            nonlocal touched
            touched = True

    with pytest.raises(PluginCompatibilityError, match="targets extension API"):
        PluginManager().activate(Old())
    assert touched is False


def test_duplicate_plugin_activation_is_rejected():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    with pytest.raises(PluginRegistrationError, match="already activated"):
        manager.activate(BasicPlugin())


def test_deactivate_removes_registry_contributions_but_does_not_claim_to_unload_python_code():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    record = manager.deactivate("example.basic")
    assert record is not None
    assert manager.activation("example.basic") is None
    assert manager.registry.list() == ()
    assert manager.deactivate("example.basic") is None


def test_registry_unknown_extension_is_explicit_error():
    with pytest.raises(ExtensionNotFoundError):
        ExtensionRegistry().get("missing:thing")


def test_registry_can_filter_extensions_by_point():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    assert [x.extension_id for x in manager.registry.list(ExtensionPoint.LLM_CLIENT)] == [
        "example.basic:chat"
    ]


@pytest.mark.asyncio
async def test_llm_client_factory_contract_and_config_are_provider_neutral():
    seen = {}

    class P:
        manifest = PluginManifest(
            "config.llm", "Config", "1",
            (ExtensionPoint.LLM_CLIENT,),
            (PluginCapability.MODEL_IO,),
        )
        def register(self, registrar):
            def factory(ctx):
                seen.update(ctx.config)
                return FakeLLM()
            registrar.register_llm_client("client", factory)

    manager = PluginManager()
    manager.activate(P())
    client = manager.registry.create_llm_client("config.llm:client", config={"model": "local/demo"})
    assert seen == {"model": "local/demo"}
    assert (await client.generate([])).model == "plugin/fake"


def test_embedding_provider_factory_contract():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    provider = manager.registry.create_embedding_provider("example.basic:embed")
    assert provider.model_id == "plugin/embed"
    assert provider.dimensions == 2


@pytest.mark.asyncio
async def test_vision_provider_factory_contract():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    provider = manager.registry.create_vision_provider("example.basic:vision")
    result = await provider.analyze(None)
    assert result.text == "scene"


def test_observability_sink_factory_contract():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    sink = manager.registry.create_observability_sink("example.basic:telemetry")
    assert isinstance(sink, InMemoryObservabilitySink)


def test_world_perception_policy_factory_contract():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    policy = manager.registry.create_world_perception_policy("example.basic:perception")
    assert policy.name == "plugin-policy"


def test_cognitive_specialist_factory_returns_existing_bounded_specialist_spec():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    spec = manager.registry.create_cognitive_specialist("example.basic:lore")
    assert isinstance(spec, SpecialistSpec)
    assert spec.specialist_id == "lore"
    assert spec.role is CognitiveRole.MEMORY


def test_factory_returning_wrong_contract_is_rejected_at_creation_not_registration():
    class BadFactory:
        manifest = PluginManifest(
            "bad.factory", "Bad", "1",
            (ExtensionPoint.LLM_CLIENT,),
            (PluginCapability.MODEL_IO,),
        )
        def register(self, registrar):
            registrar.register_llm_client("client", lambda ctx: object())

    manager = PluginManager()
    manager.activate(BadFactory())
    with pytest.raises(ExtensionValidationError, match="generate"):
        manager.registry.create_llm_client("bad.factory:client")
    assert manager.activation("bad.factory") is not None


def test_typed_creation_rejects_wrong_extension_point():
    manager = PluginManager()
    manager.activate(BasicPlugin())
    with pytest.raises(ExtensionValidationError, match="not llm_client"):
        manager.registry.create_llm_client("example.basic:telemetry")


def test_tool_plugin_activation_does_not_install_tool_automatically():
    tools = ToolRegistry()
    manager = PluginManager()
    manager.activate(ToolPlugin())
    assert len(tools) == 0
    assert "plugin_echo" not in tools


def test_tool_installation_requires_explicit_host_call():
    tools = ToolRegistry()
    manager = PluginManager()
    manager.activate(ToolPlugin())
    contribution = manager.registry.install_tool(tools, "example.tools:echo")
    assert contribution.definition.name == "plugin_echo"
    assert "plugin_echo" in tools
    assert tools.get("plugin_echo").handler(text="hello") == "echo:hello"


def test_tool_extension_requires_callable_handler():
    definition = ToolDefinition(
        name="x", description="x",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    with pytest.raises(TypeError, match="handler"):
        ToolExtension(definition, None)


def test_policy_can_deny_tool_execution_capability():
    policy = PluginActivationPolicy(
        allowed_capabilities=frozenset({PluginCapability.OBSERVABILITY})
    )
    with pytest.raises(PluginPolicyError, match="tool_execution"):
        PluginManager().activate(ToolPlugin(), policy=policy)


def test_policy_can_deny_extension_point_even_when_capability_is_allowed():
    policy = PluginActivationPolicy(
        allowed_capabilities=frozenset({PluginCapability.TOOL_EXECUTION}),
        allowed_extension_points=frozenset(),
    )
    with pytest.raises(PluginPolicyError, match="tool"):
        PluginManager().activate(ToolPlugin(), policy=policy)


def test_policy_can_allowlist_plugin_ids():
    policy = PluginActivationPolicy(allowed_plugin_ids=frozenset({"other.plugin"}))
    with pytest.raises(PluginPolicyError, match="allowlisted"):
        PluginManager().activate(BasicPlugin(), policy=policy)


def test_empty_plugin_is_legal_for_config_driven_optional_contributions():
    class Empty:
        manifest = PluginManifest("empty.plugin", "Empty", "1", (), ())
        def register(self, registrar):
            pass

    record = PluginManager().activate(Empty())
    assert record.extension_ids == ()


def test_activation_record_does_not_store_plugin_config_or_runtime_authority_objects():
    record = PluginManager().activate(BasicPlugin())
    assert not hasattr(record, "config")
    assert not hasattr(record, "runtime")
    assert not hasattr(record, "memory")
    assert not hasattr(record, "world")


def test_factory_context_exposes_no_character_or_cognition_authority():
    ctx = ExtensionFactoryContext("x.plugin", config={"token": "secret"})
    for name in ("runtime", "character", "memory", "beliefs", "goals", "commit", "world"):
        assert not hasattr(ctx, name)


def test_factory_exception_does_not_change_registry_or_activation_state():
    class P:
        manifest = PluginManifest(
            "factory.failure", "Failure", "1",
            (ExtensionPoint.OBSERVABILITY_SINK,),
            (PluginCapability.OBSERVABILITY,),
        )
        def register(self, registrar):
            def factory(ctx):
                raise RuntimeError("factory boom")
            registrar.register_observability_sink("sink", factory)

    manager = PluginManager()
    record = manager.activate(P())
    before = manager.registry.list()
    with pytest.raises(RuntimeError, match="factory boom"):
        manager.registry.create_observability_sink("factory.failure:sink")
    assert manager.registry.list() == before
    assert manager.activation("factory.failure") == record


def test_extension_package_does_not_import_character_authority_managers_or_renderers():
    root = Path(__file__).resolve().parents[1] / "src" / "ai_character_engine" / "extensions"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for forbidden in (
        "ai_character_engine.long_term_cognition",
        "ai_character_engine.goals",
        "ai_character_engine.commit",
        "ai_character_engine.runtime",
        "ai_character_engine.multi_character",
        "ai_character_engine.avatar",
        "ai_character_engine.voice",
        "ai_character_engine_vrm",
    ):
        assert forbidden not in source


def test_discovery_is_metadata_only_and_does_not_load_entry_point(monkeypatch):
    loaded = False

    class EP:
        name = "demo"
        value = "demo.module:plugin"
        group = ENTRY_POINT_GROUP
        dist = SimpleNamespace(name="demo-dist")
        def load(self):
            nonlocal loaded
            loaded = True
            return BasicPlugin()

    class EPS(list):
        def select(self, *, group):
            return [item for item in self if item.group == group]

    monkeypatch.setattr("ai_character_engine.extensions.discovery.metadata.entry_points", lambda: EPS([EP()]))
    found = discover_plugins()
    assert found == (DiscoveredPlugin("demo", "demo.module:plugin", distribution="demo-dist"),)
    assert loaded is False


def test_load_plugin_is_explicit_and_loads_exact_named_entry_point(monkeypatch):
    calls = []

    class EP:
        group = ENTRY_POINT_GROUP
        value = "demo.module:plugin"
        dist = None
        def __init__(self, name):
            self.name = name
        def load(self):
            calls.append(self.name)
            return BasicPlugin

    class EPS(list):
        def select(self, *, group):
            return [item for item in self if item.group == group]

    monkeypatch.setattr(
        "ai_character_engine.extensions.discovery.metadata.entry_points",
        lambda: EPS([EP("demo"), EP("other")]),
    )
    plugin = load_plugin("demo")
    assert isinstance(plugin, BasicPlugin)
    assert calls == ["demo"]


def test_load_plugin_rejects_missing_and_ambiguous_entry_points(monkeypatch):
    class EP:
        group = ENTRY_POINT_GROUP
        value = "x:y"
        dist = None
        def __init__(self, name): self.name = name
        def load(self): return BasicPlugin()
    class EPS(list):
        def select(self, *, group): return list(self)

    monkeypatch.setattr(
        "ai_character_engine.extensions.discovery.metadata.entry_points",
        lambda: EPS([EP("same"), EP("same")]),
    )
    with pytest.raises(PluginRegistrationError, match="not found"):
        load_plugin("missing")
    with pytest.raises(PluginRegistrationError, match="multiple"):
        load_plugin("same")


def test_plugin_manager_does_not_auto_discover_or_auto_load_installed_plugins(monkeypatch):
    called = False
    def explode():
        nonlocal called
        called = True
        raise AssertionError("must not discover during manager construction")
    monkeypatch.setattr("ai_character_engine.extensions.discovery.metadata.entry_points", explode)
    manager = PluginManager()
    assert manager.active_plugins() == ()
    assert called is False


def test_public_exports_and_versions():
    import ai_character_engine as ace

    assert ace.__version__ == "1.0.0"
    assert ace.PluginManager is PluginManager
    assert ace.ExtensionPoint.TOOL.value == "tool"


def test_package_versions_match():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    core = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    vrm = tomllib.loads((root / "packages/renderer-vrm/pyproject.toml").read_text())["project"]["version"]
    assert (core, vrm) == ("1.0.0", "1.0.0")


def test_docs_describe_extension_trust_boundary():
    root = Path(__file__).resolve().parents[1]
    guide = (root / "docs" / "extensions.md").read_text(encoding="utf-8")
    assert "not a security sandbox" in guide
    assert "activation is not authority" in guide


def test_offline_example_runs_without_external_plugin_or_provider():
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run(
        [sys.executable, str(root / "tests/scenarios/plugin_extension_sdk.py")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    assert "activated: example.offline" in proc.stdout
    assert "extension: example.offline:telemetry" in proc.stdout
    assert "tools before explicit install: 0" in proc.stdout
    assert "tools after explicit install: 1" in proc.stdout
