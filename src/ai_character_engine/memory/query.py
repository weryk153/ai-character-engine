from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class QueryRewriter(Protocol):
    async def rewrite(
        self,
        query: str,
        *,
        character_id: str | None = None,
        context: Sequence[str] = (),
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class IdentityQueryRewriter:
    async def rewrite(
        self,
        query: str,
        *,
        character_id: str | None = None,
        context: Sequence[str] = (),
    ) -> str:
        return query


@dataclass(frozen=True, slots=True)
class ContextAppendingQueryRewriter:
    """Deterministic baseline rewriter for demos/tests.

    It does not claim semantic understanding. It simply appends bounded context
    when the current query is short or referential, providing a transparent
    baseline before introducing an LLM-based rewrite provider.
    """

    max_context_items: int = 3
    max_context_chars: int = 500

    async def rewrite(
        self,
        query: str,
        *,
        character_id: str | None = None,
        context: Sequence[str] = (),
    ) -> str:
        cleaned = " ".join(query.split())
        selected = [" ".join(item.split()) for item in context[-self.max_context_items :] if item.strip()]
        suffix = " ".join(selected)
        if not suffix:
            return cleaned
        suffix = suffix[: self.max_context_chars]
        return f"{cleaned} {suffix}".strip()


@dataclass(frozen=True, slots=True)
class CallableQueryRewriter:
    rewrite_fn: Callable[..., str | Awaitable[str]]

    async def rewrite(
        self,
        query: str,
        *,
        character_id: str | None = None,
        context: Sequence[str] = (),
    ) -> str:
        result = self.rewrite_fn(
            query,
            character_id=character_id,
            context=tuple(context),
        )
        if inspect.isawaitable(result):
            result = await result
        rewritten = str(result).strip()
        return rewritten or query
