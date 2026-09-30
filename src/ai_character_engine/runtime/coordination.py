"""Single-event-loop session turn ownership, shared by all runtime entry points."""
from contextlib import contextmanager
from dataclasses import dataclass


class RuntimeBusyError(RuntimeError):
    """A session already has a reserved or executing turn."""


class PartialTurnError(RuntimeError):
    """Local state was restored, but tools/storage may have external effects."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"Turn requires host review before retry (phase={phase}).")


@dataclass(eq=False)
class TurnLease:
    executing: bool = False
    requires_review: bool = False


class TurnCoordinator:
    """Non-waiting guard; use on one asyncio event loop, not across threads."""

    def __init__(self):
        self._lease: TurnLease | None = None

    @property
    def busy(self) -> bool:
        return self._lease is not None

    @contextmanager
    def reserve(self):
        if self.busy:
            raise RuntimeBusyError("A character turn is already running.")
        lease = self._lease = TurnLease()
        try:
            yield lease
        finally:
            self._lease = None

    @contextmanager
    def execute(self, lease: TurnLease):
        if lease is not self._lease or lease.executing:
            raise RuntimeBusyError("Invalid or already executing turn lease.")
        lease.executing = True
        try:
            yield
        finally:
            lease.executing = False


@dataclass
class TurnEffects:
    phase: str = "inference"
    requires_review: bool = False

    def begin(self, phase: str):
        self.phase = phase
        self.requires_review = True
