"""What kind of thing she said about herself, and which conversation it came from.

Some of what she says about herself is hers in every conversation: who she
is, her tastes and habits. Some holds only where it was said: what she is
doing in this conversation (teaching the user, a game), what she means to do
next in it, and what she thinks of the user there. A small model names the
kind freely ("habits", "physical_trait", "feeling"); the names it uses are
brought onto one list here.
"""

from __future__ import annotations

import re

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


# Durable kinds a model files its judgements of the user under. relationship
# and taste are left out: "I like you" is hers to keep.
_KINDS_MISTAKEN_FOR_A_VIEW = frozenset({"opinion", "trait", "habit", "history"})

# How a summary names the user, by language. Whole words for Latin script;
# the CJK and Korean words need no boundary.
_USER_WORDS = re.compile(
    r"(?:\b(?:the\s+user|user|you|your|yours)\b"
    r"|用戶|用户|使用者|對方|对方|ユーザー|あなた|君|사용자|당신)",
    re.IGNORECASE,
)


def about_the_user(summary: object) -> bool:
    """Whether a summary of what she said is about the user: it names them."""
    return bool(_USER_WORDS.search(str(summary or "")))


def kind_of_what_she_said(kind: object, summary: object) -> str:
    """The kind a self-memory item is kept under.

    A small model files what she thinks of the user under opinion, trait or
    habit — told to use view_of_user it still does, on the comparison pages
    10 times out of 10. Those are hers in every conversation, so her
    reproaches from one lesson followed her into the next. The summary tells:
    one that names the user, under one of those kinds, is a view of the user.
    """
    normalized = self_memory_kind(kind)
    if normalized in _KINDS_MISTAKEN_FOR_A_VIEW and about_the_user(summary):
        return "view_of_user"
    return normalized
