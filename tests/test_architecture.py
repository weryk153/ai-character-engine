from pathlib import Path
import ast
import tomllib
import ai_character_engine
from ai_character_engine.avatar import AvatarRuntime, AvatarBehaviorRuntime

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "ai_character_engine"
FORBIDDEN_IMPORT_PREFIXES = ("ai_character_engine_vrm", "ai_character_engine_live2d", "live2d", "three_vrm", "cubism")
FORBIDDEN_CORE_TOKENS = ("vrm", "live2d", "three-vrm", "three_vrm", "cubism")

def test_core_import_graph_has_no_renderer_or_product_specific_dependency():
    offenders=[]
    for path in CORE.rglob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node, ast.Import): names=[alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom): names=[node.module or ""]
            for name in names:
                if name.lower().startswith(FORBIDDEN_IMPORT_PREFIXES): offenders.append((path.relative_to(ROOT), name))
    assert offenders == []

def test_core_source_vocabulary_is_renderer_and_product_neutral():
    offenders=[]
    for path in CORE.rglob("*.py"):
        text=path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_CORE_TOKENS:
            if token in text: offenders.append((str(path.relative_to(ROOT)), token))
    assert offenders == []

def test_core_distribution_dependencies_do_not_include_renderer_sdks():
    data=tomllib.loads((ROOT/"pyproject.toml").read_text())
    deps=" ".join(data["project"].get("dependencies", [])).lower()
    assert all(token not in deps for token in ("vrm", "live2d", "cubism", "three"))

def test_renderer_packages_depend_on_core_not_core_on_renderer_packages():
    vrm=tomllib.loads((ROOT/"packages/renderer-vrm/pyproject.toml").read_text())
    assert any(dep.startswith("ai-character-engine>=") for dep in vrm["project"]["dependencies"])

def test_neutral_avatar_runtime_has_no_renderer_adapter_constructor_argument():
    import inspect
    assert "vrm" not in inspect.signature(AvatarRuntime).parameters
    assert "vrm" not in inspect.signature(AvatarBehaviorRuntime).parameters

def test_neutral_avatar_cues_do_not_expose_renderer_payload():
    from ai_character_engine.avatar import AvatarCueBundle, AvatarBehaviorCueBundle, AvatarBehaviorPhase
    cue=AvatarCueBundle("turn",0,0,0.0,0.0).to_dict()
    behavior=AvatarBehaviorCueBundle(0, AvatarBehaviorPhase.IDLE, 0.0).to_dict()
    assert "vrm" not in cue
    assert "vrm" not in behavior

def test_core_package_version_is_0292():
    assert ai_character_engine.__version__ == "1.0.0"
