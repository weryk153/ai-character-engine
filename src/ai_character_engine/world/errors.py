from __future__ import annotations


class WorldError(RuntimeError):
    """Base error for the provider-neutral world/environment layer."""


class WorldRevisionConflictError(WorldError):
    """Raised when a caller tries to mutate a stale world revision."""


class WorldPerceptionDeniedError(WorldError):
    """Raised when a character is not eligible to perceive a world event."""


class WorldPerceptionProjectionError(WorldError):
    """Raised when a perception policy tries to expose facts outside an event."""
