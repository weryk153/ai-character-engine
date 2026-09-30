from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ai_character_engine.tools.errors import (
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolTimeoutError,
)
from ai_character_engine.tools.models import ToolCall, ToolResult
from ai_character_engine.tools.registry import ToolRegistry
from ai_character_engine.tools.validation import normalize_arguments, validate_arguments

logger = logging.getLogger(__name__)

ToolAuthorizer = Callable[[ToolCall], bool | Awaitable[bool]]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 15.0,
        authorizer: ToolAuthorizer | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.authorizer = authorizer

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            registered = self.registry.get(call.name)
            normalized_arguments = normalize_arguments(
                call.arguments, registered.definition.parameters
            )
            validate_arguments(normalized_arguments, registered.definition.parameters)
            normalized_call = ToolCall(
                call_id=call.call_id, name=call.name, arguments=normalized_arguments
            )
            await self._authorize(
                normalized_call, requires_approval=registered.definition.requires_approval
            )
            output = await asyncio.wait_for(
                self._invoke(registered.handler, normalized_arguments),
                timeout=self.timeout_seconds,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=self._serialize_output(output),
            )
        except asyncio.TimeoutError:
            error = ToolTimeoutError(
                f"tool '{call.name}' timed out after {self.timeout_seconds}s"
            )
            logger.warning("tool_timeout tool=%s call_id=%s", call.name, call.call_id)
            return ToolResult(call.call_id, call.name, str(error), is_error=True)
        except ToolError as exc:
            logger.warning(
                "tool_error tool=%s call_id=%s error=%s", call.name, call.call_id, exc
            )
            return ToolResult(call.call_id, call.name, str(exc), is_error=True)
        except Exception as exc:
            error = ToolExecutionError(f"tool '{call.name}' failed: {exc}")
            logger.exception("tool_execution_failed tool=%s call_id=%s", call.name, call.call_id)
            return ToolResult(call.call_id, call.name, str(error), is_error=True)

    async def _authorize(self, call: ToolCall, *, requires_approval: bool) -> None:
        if not requires_approval:
            return
        if self.authorizer is None:
            raise ToolPermissionError(f"tool '{call.name}' requires approval")
        allowed = self.authorizer(call)
        if inspect.isawaitable(allowed):
            allowed = await allowed
        if not allowed:
            raise ToolPermissionError(f"tool '{call.name}' was not approved")

    async def _invoke(self, handler: Any, arguments: dict[str, Any]) -> Any:
        result = handler(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _serialize_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False)
        except TypeError:
            return str(output)
