"""One object for a host: a character that replies, remembers and changes."""

from ai_character_engine.context.builder import SELF_MEMORY_LINE
from ai_character_engine.memory.self_kinds import CONVERSATION_SELF_MEMORY_KINDS, SELF_MEMORY_KINDS
from ai_character_engine.state.mood import CHARACTER_MOODS

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
    "CHARACTER_MOODS",
    "CONVERSATION_SELF_MEMORY_KINDS",
    "CharacterCompanion",
    "CompanionClosed",
    "CompanionSettings",
    "CompanionSnapshot",
    "ModelAccess",
    "PoliteClient",
    "SELF_MEMORY_KINDS",
    "SELF_MEMORY_LINE",
    "TurnInterrupted",
]
