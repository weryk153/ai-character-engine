from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be a JSON Schema object")

    def to_json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


@dataclass(slots=True, frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    is_error: bool = False
