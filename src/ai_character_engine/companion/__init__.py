"""One object for a host: a character that replies, remembers and changes."""

from ai_character_engine.context.builder import SELF_MEMORY_LINE
from ai_character_engine.memory.self_kinds import CONVERSATION_SELF_MEMORY_KINDS, SELF_MEMORY_KINDS
from ai_character_engine.state.mood import CHARACTER_MOODS

from .access import ModelAccess, PoliteClient
from .across_runs import AcrossRunsMemory
from .avatar_actions import AvatarChoices, LineActions, ReplyActions
from .companion import (
    ACKNOWLEDGEMENTS,
    ASSISTANT_SPEAK,
    CharacterCompanion,
    CompanionClosed,
    CompanionSnapshot,
    DiaryEntry,
    MemoryConflict,
    TurnInterrupted,
    UserState,
)
from .save_state import StateBusy, StateFormatError, load_state_file, save_state_file
from .settings import CompanionSettings

__all__ = [
    "ACKNOWLEDGEMENTS",
    "AcrossRunsMemory",
    "AvatarChoices",
    "ASSISTANT_SPEAK",
    "CHARACTER_MOODS",
    "CONVERSATION_SELF_MEMORY_KINDS",
    "CharacterCompanion",
    "CompanionClosed",
    "CompanionSettings",
    "CompanionSnapshot",
    "DiaryEntry",
    "LineActions",
    "MemoryConflict",
    "ModelAccess",
    "PoliteClient",
    "ReplyActions",
    "SELF_MEMORY_KINDS",
    "SELF_MEMORY_LINE",
    "StateBusy",
    "StateFormatError",
    "TurnInterrupted",
    "UserState",
    "load_state_file",
    "save_state_file",
]
