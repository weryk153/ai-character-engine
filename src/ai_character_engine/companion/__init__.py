"""One object for a host: a character that replies, remembers and changes."""

from .access import ModelAccess, PoliteClient
from .companion import (
    CharacterCompanion,
    CompanionClosed,
    CompanionSnapshot,
    TurnInterrupted,
)
from .settings import CompanionSettings

__all__ = [
    "CharacterCompanion",
    "CompanionClosed",
    "CompanionSettings",
    "CompanionSnapshot",
    "ModelAccess",
    "PoliteClient",
    "TurnInterrupted",
]
