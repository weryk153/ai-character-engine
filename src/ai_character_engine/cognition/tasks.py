from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.tasks.models import TaskContext, TaskOutput
from ai_character_engine.tools.models import ToolDefinition

from .models import CognitiveRole, CognitiveRouteRequirements
from .runtime import CognitiveModelRuntime

PromptBuilder = Callable[[TaskContext], list[Message] | Awaitable[list[Message]]]
ResultBuilder = Callable[[TaskContext, LLMResponse], TaskOutput | Any | Awaitable[TaskOutput | Any]]


@dataclass(slots=True)
class CognitiveTaskHandler:
    """MultiTaskRuntime handler that asks for a role, never a concrete model id."""

    models: CognitiveModelRuntime
    role: CognitiveRole
    prompt_builder: PromptBuilder
    requirements: CognitiveRouteRequirements | None = None
    tools: tuple[ToolDefinition, ...] = field(default_factory=tuple)
    result_builder: ResultBuilder | None = None

    async def __call__(self, context: TaskContext) -> TaskOutput:
        messages = self.prompt_builder(context)
        if inspect.isawaitable(messages):
            messages = await messages
        response = await self.models.generate(
            self.role,
            list(messages),
            tools=list(self.tools) or None,
            requirements=self.requirements,
        )
        if self.result_builder is None:
            return TaskOutput(
                value=response.text,
                metadata={
                    "model": response.model,
                    "cognitive": dict(response.metadata.get("cognitive", {})),
                },
            )
        built = self.result_builder(context, response)
        if inspect.isawaitable(built):
            built = await built
        if isinstance(built, TaskOutput):
            return built
        return TaskOutput(
            value=built,
            metadata={
                "model": response.model,
                "cognitive": dict(response.metadata.get("cognitive", {})),
            },
        )
