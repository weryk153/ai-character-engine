from __future__ import annotations

from dataclasses import dataclass

from ai_character_engine.state.mood import DEFAULT_MOOD_FLOOR, DEFAULT_MOOD_HALF_LIFE_SECONDS


@dataclass(frozen=True, slots=True)
class CompanionSettings:
    """How often each background worker runs, and how patient the companion is.

    ``*_every`` is in turns; 0 turns a worker off. The defaults were measured on
    a local 9B model on a 16 GB laptop, where one call takes about 7 s for
    emotion, 5 s for memory, 10-20 s for goals, 12 s for reflection and 15 s
    for a summary, and one turn of conversation lasts about 18 s. Running all
    five every turn needs 27 s per turn and the queue only grows. A host with a
    faster or separate background model can lower the numbers.
    """

    emotion_every: int = 1
    # Her reply read back for slips (talking about prompts or being an AI,
    # markup said aloud, contradicting her persona, repeating her last reply,
    # the wrong language); a slip found is pointed out to her on her next
    # reply only. Every turn: the note is for the next reply.
    reply_check_every: int = 1
    # Her own mood, from both sides of the conversation; about 7 s a call on
    # the hardware above, so every second turn. Between readings the emotion of
    # the user still moves it. A host with a separate background model can
    # read it every turn.
    mood_every: int = 2
    memory_every: int = 2
    # A memory of the user just written is held against the earlier ones on
    # its topic: one more call, only when there are any. A fact that moved on
    # (a new job, a move) replaces the old one; two that cannot both be true
    # are kept, and she asks the user which is right, once.
    memory_conflicts: bool = True
    goal_every: int = 4
    reflection_every: int = 6
    summary_every: int = 0
    # What she said about herself: her tastes, habits, plans. Read from her
    # own lines, kept for every conversation; see self_memories_kept. One more
    # call like memory's, so every second turn like memory: each run reads
    # every line of hers since the one before, nothing is skipped.
    self_memory_every: int = 2
    # One model call. The engine's own worker defaults (12 s for emotion, 20 s
    # for a summary) time out on the hardware above.
    call_timeout_seconds: float = 60.0
    # A background result this many turns late is still used. Memory is used
    # however late: what the user stated stays true, and no later job reads
    # the same lines again.
    max_turns_late: int = 3
    # Goals do not expire by themselves; older ones leave the context.
    goal_max_age_days: int = 7
    # How many goals and thoughts she keeps in mind: the most pressing goals
    # and the newest thoughts. Every goal went in before, bounded by tokens
    # alone; a small model's goals are many and uneven, and each one showed
    # in the reply as one more question.
    goals_shown: int = 3
    thoughts_shown: int = 2
    # Background work resumes after this long without an end-of-reply signal.
    foreground_patience_seconds: float = 120.0
    # How many conversations keep their recent messages in memory at once.
    conversations_kept: int = 8
    max_history_messages: int = 40
    # Forward text as it is generated even when tools are registered.
    stream_text_with_tools: bool = True
    # How many finished background tasks, commit decisions and events are
    # kept to be asked about. The rest is let go: a host runs for days.
    records_kept: int = 256
    # How many memories of the conversation she is given at most. What the
    # newest message is about comes first, then what matters most.
    memories_recalled: int = 40
    # The language memories, goals and thoughts are written in, by any name a
    # model understands ("繁體中文" for Traditional Chinese, "Japanese"). Empty:
    # the language the user writes in, which a small model does not always work
    # out.
    language: str = ""
    # How many things she said about herself she keeps, and how many of the
    # newest are in her mind on a turn. Over the first, the oldest are
    # forgotten.
    self_memories_kept: int = 40
    self_memories_shown: int = 12
    # Her mood fades as time passes, talked to or not: its intensity halves
    # every mood_half_life_seconds (a quarter is left after ten minutes away),
    # and below mood_floor she is neutral again.
    mood_half_life_seconds: float = DEFAULT_MOOD_HALF_LIFE_SECONDS
    mood_floor: float = DEFAULT_MOOD_FLOOR
    # What she is doing in a conversation stays in it: what she said she is
    # working on or plans to do there and what she thinks of the user there,
    # her short-term goals and her thoughts are in mind only in the
    # conversation they came from. Her tastes, habits, history and long-term
    # goals are hers in every conversation. False: everything in every
    # conversation, as before 1.2.0.
    plans_stay_in_conversation: bool = True
    # A short-term goal left untouched this long leaves her mind, also in its
    # own conversation; 0: only goal_max_age_days.
    short_term_goal_max_age_hours: float = 24.0

    def __post_init__(self) -> None:
        for name in (
            "emotion_every",
            "reply_check_every",
            "mood_every",
            "memory_every",
            "goal_every",
            "reflection_every",
            "summary_every",
            "self_memory_every",
            "max_turns_late",
            "goal_max_age_days",
            "goals_shown",
            "thoughts_shown",
            "max_history_messages",
            "memories_recalled",
            "self_memories_shown",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.self_memories_kept < 1:
            raise ValueError("self_memories_kept must be >= 1")
        if self.records_kept < 1:
            raise ValueError("records_kept must be >= 1")
        if self.conversations_kept < 1:
            raise ValueError("conversations_kept must be >= 1")
        if self.call_timeout_seconds <= 0:
            raise ValueError("call_timeout_seconds must be > 0")
        if self.foreground_patience_seconds <= 0:
            raise ValueError("foreground_patience_seconds must be > 0")
        if self.mood_half_life_seconds <= 0:
            raise ValueError("mood_half_life_seconds must be > 0")
        if not 0 <= self.mood_floor <= 1:
            raise ValueError("mood_floor must be between 0 and 1")
        if self.short_term_goal_max_age_hours < 0:
            raise ValueError("short_term_goal_max_age_hours must be >= 0")
