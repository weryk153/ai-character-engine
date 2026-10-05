"""What kind of thing she said about herself, and which conversation it came from.

Some of what she says about herself is hers in every conversation: who she
is, her tastes and habits. Some holds only where it was said: what she is
doing in this conversation (teaching the user, a game), what she means to do
next in it, and what she thinks of the user there. A small model names the
kind freely ("habits", "physical_trait", "feeling"); the names it uses are
brought onto one list here.
"""

from __future__ import annotations

# Hers in every conversation.
DURABLE_SELF_MEMORY_KINDS: tuple[str, ...] = (
    "identity",
    "trait",
    "taste",
    "habit",
    "history",
    "relationship",
    "opinion",
)
# Hers only in the conversation they were said in.
CONVERSATION_SELF_MEMORY_KINDS: tuple[str, ...] = ("working_on", "plan", "view_of_user")
# In this order wherever they are listed: the self-memory prompt, documentation.
SELF_MEMORY_KINDS: tuple[str, ...] = DURABLE_SELF_MEMORY_KINDS + CONVERSATION_SELF_MEMORY_KINDS

# Metadata key of a record (memory, self memory, goal, reflection) and
# provenance key of a proposal: the conversation it came from. A record without
# it was written before 1.2.0 or outside a CharacterCompanion.
CONVERSATION_KEY = "conversation_id"

# Other names a model gives these kinds.
_SAME_KIND = {"physical_trait": "trait", "feeling": "view_of_user"}


def self_memory_kind(value: object) -> str:
    """The kind on the list that ``value`` names, or ``value`` itself (cleaned)
    when it names none; "fact" when it is empty. A kind off the list is kept:
    it is hers in every conversation, as every kind was before 1.2.0."""
    kind = "_".join(str(value or "").strip().casefold().replace("-", " ").split())
    if not kind:
        return "fact"
    if kind in SELF_MEMORY_KINDS:
        return kind
    if kind.endswith("s") and kind[:-1] in SELF_MEMORY_KINDS:
        return kind[:-1]
    single = kind[:-1] if kind.endswith("s") else kind
    return _SAME_KIND.get(kind) or _SAME_KIND.get(single) or kind


def stays_in_conversation(kind: object) -> bool:
    """Whether what she said, of this kind, holds only where it was said."""
    return self_memory_kind(kind) in CONVERSATION_SELF_MEMORY_KINDS
