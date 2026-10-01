"""One object for a host: a character that replies, remembers and changes."""

from ai_character_engine.context.builder import SELF_MEMORY_LINE

from .access import ModelAccess, PoliteClient
from .companion import (
    ACKNOWLEDGEMENTS,
    ASSISTANT_SPEAK,
    CharacterCompanion,
    CompanionClosed,
    CompanionSnapshot,
    TurnInterrupted,
)
from .settings import CompanionSettings

__all__ = [
    "ACKNOWLEDGEMENTS",
    "ASSISTANT_SPEAK",
    "CharacterCompanion",
    "CompanionClosed",
    "CompanionSettings",
    "CompanionSnapshot",
    "ModelAccess",
    "PoliteClient",
    "SELF_MEMORY_LINE",
    "TurnInterrupted",
]
