from __future__ import annotations

from typing import Any

from ai_character_engine.tools.errors import ToolValidationError

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ToolValidationError("tool schema must have type=object")

    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    additional_properties = schema.get("additionalProperties", True)

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolValidationError(f"missing required arguments: {', '.join(missing)}")

    if additional_properties is False:
        extra = [name for name in arguments if name not in properties]
        if extra:
            raise ToolValidationError(f"unexpected arguments: {', '.join(extra)}")

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if property_schema is None:
            continue

        expected_name = property_schema.get("type")
        expected_type = _TYPE_MAP.get(expected_name)
        if expected_type is not None and not isinstance(value, expected_type):
            raise ToolValidationError(
                f"argument '{name}' must be {expected_name}, got {type(value).__name__}"
            )

        enum = property_schema.get("enum")
        if enum is not None and value not in enum:
            raise ToolValidationError(f"argument '{name}' must be one of {enum}")


def normalize_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Narrow compatibility normalization for model-produced tool wrappers.

    Some OpenAI-compatible local models (observed with Qwen through LM Studio)
    occasionally emit ``{"parameters": {...}}`` even though the function-call
    protocol already places arguments at the top level. Unwrap only when:

    - ``parameters`` is the *only* supplied key;
    - the declared schema itself does not define a ``parameters`` property; and
    - the wrapped value is an object (or a JSON object string).

    Everything else remains strict and is rejected by ``validate_arguments``.
    """

    if not isinstance(arguments, dict):
        return arguments
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if set(arguments) != {"parameters"} or "parameters" in properties:
        return arguments
    wrapped = arguments.get("parameters")
    if isinstance(wrapped, str):
        import json
        try:
            wrapped = json.loads(wrapped)
        except json.JSONDecodeError:
            return arguments
    if isinstance(wrapped, dict):
        return dict(wrapped)
    return arguments
