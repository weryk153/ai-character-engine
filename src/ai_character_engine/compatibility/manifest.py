from __future__ import annotations

import importlib
import inspect
import json
from enum import Enum, EnumType
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_character_engine._version import VERSION

from .models import (
    ApiStability,
    ApiSymbolKind,
    PUBLIC_API_CONTRACT_VERSION,
    ParameterContract,
    PublicApiManifest,
    PublicApiSymbol,
)


DEFAULT_PUBLIC_MODULES: tuple[str, ...] = ("ai_character_engine",)


def _kind(value: Any) -> ApiSymbolKind:
    if inspect.isclass(value):
        return ApiSymbolKind.CLASS
    if inspect.isfunction(value) or inspect.ismethod(value):
        return ApiSymbolKind.FUNCTION
    if isinstance(value, (str, int, float, bool, bytes, tuple, frozenset, type(None))):
        return ApiSymbolKind.CONSTANT
    return ApiSymbolKind.OTHER


def _parameters(value: Any) -> tuple[ParameterContract, ...]:
    if not (inspect.isclass(value) or callable(value)):
        return ()
    # Populated standard enums have the same member-lookup behavior on supported
    # interpreters, but inspect exposes EnumType's functional-construction form
    # on 3.11 and (*values) on 3.12+. Keep the sealed 3.13 representation without
    # disguising explicitly customized signatures or metaclass call contracts.
    enum_class = inspect.isclass(value) and issubclass(value, Enum)
    explicit_signature = enum_class and (
        any("__signature__" in base.__dict__ for base in value.__mro__ if base is not Enum)
        or any("__signature__" in base.__dict__ for base in type(value).__mro__
               if base not in (EnumType, type, object))
    )
    if (enum_class and value.__members__ and type(value).__call__ is EnumType.__call__
            and not explicit_signature):
        return (ParameterContract(name="values", kind="var_positional", required=False),)
    try:
        if enum_class and type(value).__call__ is not EnumType.__call__ and not explicit_signature:
            # Newer Enum supplies a generic __signature__ classmethod that can
            # otherwise hide a subclass's actual custom __call__ parameters.
            signature = inspect.signature(type(value).__call__)
            signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        else:
            signature = inspect.signature(value)
    except (TypeError, ValueError):
        return ()
    result: list[ParameterContract] = []
    for parameter in signature.parameters.values():
        required = (
            parameter.default is inspect.Parameter.empty
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        result.append(
            ParameterContract(
                name=parameter.name,
                kind=parameter.kind.name.lower(),
                required=required,
            )
        )
    return tuple(result)


def _import_path(value: Any, module_name: str, public_name: str) -> str:
    # Track the documented import surface, not the implementation module.
    # Internal refactors are allowed as long as the public import keeps working.
    del value
    return f"{module_name}.{public_name}"


def build_public_api_manifest(
    *,
    modules: Sequence[str] = DEFAULT_PUBLIC_MODULES,
    stability: ApiStability = ApiStability.CANDIDATE,
    engine_version: str = VERSION,
    metadata: Mapping[str, Any] | None = None,
) -> PublicApiManifest:
    symbols: list[PublicApiSymbol] = []
    for module_name in modules:
        module = importlib.import_module(module_name)
        names = getattr(module, "__all__", None)
        if names is None:
            raise ValueError(f"public module {module_name!r} must define __all__")
        for name in sorted(set(str(item) for item in names)):
            if name.startswith("_"):
                raise ValueError(f"public module {module_name!r} exports private name {name!r}")
            if not hasattr(module, name):
                raise ValueError(f"public module {module_name!r} declares missing export {name!r}")
            value = getattr(module, name)
            symbols.append(
                PublicApiSymbol(
                    module=module_name,
                    name=name,
                    import_path=_import_path(value, module_name, name),
                    kind=_kind(value),
                    parameters=_parameters(value),
                    stability=stability,
                )
            )
    symbols.sort(key=lambda item: item.key)
    return PublicApiManifest(
        engine_version=engine_version,
        contract_version=PUBLIC_API_CONTRACT_VERSION,
        symbols=tuple(symbols),
        metadata=dict(metadata or {}),
    )


def save_public_api_manifest(manifest: PublicApiManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    target.write_text(payload + "\n", encoding="utf-8")


def load_public_api_manifest(path: str | Path) -> PublicApiManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("public API manifest must contain a JSON object")
    return PublicApiManifest.from_dict(value)


def manifest_symbol_map(manifest: PublicApiManifest) -> dict[str, PublicApiSymbol]:
    return {item.key: item for item in manifest.symbols}


def public_export_names(module_name: str = "ai_character_engine") -> tuple[str, ...]:
    module = importlib.import_module(module_name)
    names: Iterable[str] = getattr(module, "__all__", ())
    return tuple(sorted(set(str(item) for item in names)))
