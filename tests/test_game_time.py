"""Her clock is the game's.

A game runs its own time: days pass in an evening, and loading a save turns
the clock back. What she writes down is dated by her clock, ages by it and
is found by it, whatever the system clock says.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import CharacterCompanion, CompanionSettings
from tests.test_companion import BLACK_TEA, DAWN, DiaryWorker, Foreground, dawns_work, her_tea
from tests.test_conversation_plans import THOUGHT, Her, Worker, goals_citing_the_event

# A morning long before the system clock's: nothing she writes down may be
# dated by the system clock.
GAME_START = datetime(2001, 4, 1, 9, 0, tzinfo=UTC).timestamp()
TEACH = "teach me Japanese"


class GameClock:
    def __init__(self, start: float = GAME_START) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def hours(self, count: float) -> None:
        self.now += count * 3600


def make(tmp_path, clock, *, llm=None, workers=None, **settings):
    every = {
        "emotion_every": 0,
        "mood_every": 0,
        "memory_every": 0,
        "self_memory_every": 0,
        "goal_every": 0,
        "reflection_every": 0,
        "summary_every": 0,
    }
    every.update(settings)
    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=llm or Foreground((BLACK_TEA, "How are you?")),
        background_llm=workers or {},
        storage_dir=tmp_path / "engine",
        settings=CompanionSettings(**every),
        clock=clock,
    )


def everything_at_once(tmp_path, clock, llm=None):
    """A companion whose every worker writes something on the first turn."""
    return make(
        tmp_path,
        clock,
        llm=llm,
        workers={
            "memory": Worker(dawns_work),
            "self_memory": Worker(her_tea),
            "goal": Worker(goals_citing_the_event),
            "reflection": Worker(THOUGHT),
        },
        memory_every=1,
        self_memory_every=1,
        goal_every=1,
        reflection_every=1,
    )


def test_what_she_writes_down_is_dated_by_her_clock(tmp_path):
    clock = GameClock()

    async def scenario():
        current = everything_at_once(tmp_path, clock)
        await current.reply(f"{DAWN}. {TEACH}", conversation_id="a")
        await current.settle()
        store = current.runtime.memory_manager.store
        dates = {
            "memory": {r.created_at for r in store.list_for_character("mei:a")},
            "self memory": {r.created_at for r in store.list_for_character("mei#self")},
            "goals": {
                date
                for g in current.runtime.goal_manager.store.list_goals("mei")
                for date in (g.created_at, g.updated_at)
            },
            "thoughts": {
                r.created_at for r in current.runtime.long_term_cognition.reflections(character_id="mei")
            },
        }
        await current.close()
        return dates

    morning = datetime.fromtimestamp(GAME_START, UTC)
    assert asyncio.run(scenario()) == {
        "memory": {morning},
        "self memory": {morning},
        "goals": {morning},
        "thoughts": {morning},
    }


def test_a_short_term_goal_leaves_her_mind_a_day_later_by_her_clock(tmp_path):
    clock = GameClock()

    async def scenario():
        current = everything_at_once(tmp_path, clock)
        await current.reply(TEACH, conversation_id="a")
        await current.settle()
        that_day = set(current.snapshot().goals)
        clock.hours(25)
        next_day = set(current.snapshot().goals)
        await current.close()
        return that_day, next_day

    that_day, next_day = asyncio.run(scenario())
    assert that_day == {"Give the user the next Japanese exercise", "Become a good teacher"}
    assert next_day == {"Become a good teacher"}


def test_her_clock_turned_back_breaks_nothing(tmp_path):
    clock = GameClock()

    async def scenario():
        her = Her(BLACK_TEA + "How are you?", "Of course I do.")
        current = everything_at_once(tmp_path, clock, her)
        await current.reply(f"{DAWN}. {TEACH}", conversation_id="a")
        await current.settle()
        clock.hours(-24 * 7)
        reply = await current.reply("do you remember me?", conversation_id="a")
        await current.settle()
        seen = (reply.text, set(current.snapshot().goals), current.memories("a"))
        await current.close()
        return seen

    text, goals, memories = asyncio.run(scenario())
    assert text == "Of course I do."
    assert goals == {"Give the user the next Japanese exercise", "Become a good teacher"}
    assert memories == ["Dawn works at a print shop."]


def test_her_diary_is_about_her_day_by_her_clock(tmp_path):
    clock = GameClock()

    async def scenario():
        diary = DiaryWorker()
        current = make(
            tmp_path,
            clock,
            workers={"memory": Worker(dawns_work), "self_memory": Worker(her_tea), "diary": diary},
            memory_every=1,
            self_memory_every=1,
        )
        await current.reply(DAWN, conversation_id="a")
        await current.settle()
        clock.hours(1)
        entry = await current.write_diary()
        await current.close()
        return diary.prompts, entry

    prompts, entry = asyncio.run(scenario())
    assert "- Dawn works at a print shop." in prompts[0]
    assert "- Mei likes strong black tea." in prompts[0]
    assert entry is not None and entry.date == "2001-04-01"
