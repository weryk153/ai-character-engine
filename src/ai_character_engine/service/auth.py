from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass(slots=True, frozen=True)
class AuthContext:
    subject: str | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


class AuthHook(Protocol):
    async def authenticate_http(self, request: Any) -> AuthContext: ...

    async def authenticate_websocket(self, websocket: Any) -> AuthContext: ...


class NoopAuthHook:
    """Default hook for local/development use.

    Production applications should inject a real hook instead of teaching the
    character engine about JWT/OAuth/session-cookie details.
    """

    async def authenticate_http(self, request: Any) -> AuthContext:
        return AuthContext()

    async def authenticate_websocket(self, websocket: Any) -> AuthContext:
        return AuthContext()
