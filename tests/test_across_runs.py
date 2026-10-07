"""What she remembers from before the world began again.

Only the game writes it; it is kept in meta_dir, apart from her state, so
that a new game and a loaded save both leave it as it is.
"""
from __future__ import annotations

import asyncio
import io
import zipfile

import pytest

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import AcrossRunsMemory, CharacterCompanion, CompanionSettings
from tests.test_companion import DAWN, dawns_work, her_tea
from tests.test_conversation_plans import THOUGHT, Her, Worker, goals_citing_the_event
from tests.test_game_time import GameClock

FRAMING = "From before this world began again, you remember:"


def make(storage, meta, *, llm=None, workers=None, clock=None, **settings):
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
        character=CharacterProfile(
            id="mei",
            name="Mei",
            description="A researcher.",
            background="She grew up above her parents' bakery.",
        ),
        llm=llm or Her("Hello again."),
        background_llm=workers or {},
        storage_dir=storage,
        settings=CompanionSettings(**every),
        clock=clock or GameClock(),
        meta_dir=meta,
    )


def system_prompt(her):
    return her.calls[-1][0].content


def test_what_the_game_has_her_remember_is_in_her_prompt_after_who_she_is(tmp_path):
    clock = GameClock()

    async def scenario():
        her = Her("Hello again.")
        current = make(tmp_path / "s", tmp_path / "meta", llm=her, clock=clock)
        await current.reply("hi")
        before = system_prompt(her)
        first = current.remember_across_runs("The player saved you from the fire.", run=1)
        current.remember_across_runs("You promised to wait at the bridge.", tags=["promise"], run=1)
        await current.reply("hi again")
        after = system_prompt(her)
        kept = current.across_runs()
        await current.close()
        return before, after, kept, first

    before, after, kept, first = asyncio.run(scenario())
    assert FRAMING not in before
    section = f"{FRAMING}\n- The player saved you from the fire.\n- You promised to wait at the bridge."
    assert section in after
    assert after.index("Background:") < after.index(section)
    assert kept == [
        AcrossRunsMemory(first, "The player saved you from the fire.", (), clock(), 1),
        AcrossRunsMemory(kept[1].id, "You promised to wait at the bridge.", ("promise",), clock(), 1),
    ]


def test_the_game_says_it_its_own_way_and_only_the_newest_are_in_mind(tmp_path):
    async def scenario(**settings):
        her = Her("Hello again.")
        current = make(tmp_path / str(len(settings)), tmp_path / f"meta{len(settings)}", llm=her, **settings)
        for number in range(4):
            current.remember_across_runs(f"Echo {number}")
        await current.reply("hi")
        await current.close()
        return system_prompt(her)

    prompt = asyncio.run(
        scenario(across_runs_framing="You can't shake the feeling:", across_runs_in_context=2)
    )
    assert "You can't shake the feeling:\n- Echo 2\n- Echo 3" in prompt
    assert "Echo 1" not in prompt
    assert f"{FRAMING}\n- Echo 0\n- Echo 1\n- Echo 2\n- Echo 3" in asyncio.run(scenario())
    assert "Echo" not in asyncio.run(scenario(across_runs_in_context=0, across_runs_framing="x"))


def test_a_new_game_and_a_loaded_save_leave_them_as_they_are(tmp_path):
    meta = tmp_path / "meta"

    async def scenario():
        first_run = make(tmp_path / "run1", meta)
        await first_run.reply("hi")
        first_run.remember_across_runs("The fire.", run=1)
        save = await first_run.export_state()
        first_run.remember_across_runs("The bridge.", run=1)
        await first_run.close()

        new_game = make(tmp_path / "run2", meta)
        in_new_game = [memory.text for memory in new_game.across_runs()]
        await new_game.close()

        loaded = make(tmp_path / "run3", meta)
        loaded.import_state(save)
        after_loading = [memory.text for memory in loaded.across_runs()]
        await loaded.close()
        return in_new_game, after_loading, zipfile.ZipFile(io.BytesIO(save)).namelist()

    in_new_game, after_loading, saved = asyncio.run(scenario())
    assert in_new_game == ["The fire.", "The bridge."]
    assert after_loading == ["The fire.", "The bridge."]
    assert not any("across" in name for name in saved)


def test_a_memory_forgotten_leaves_her_prompt(tmp_path):
    async def scenario():
        her = Her("Hello again.")
        current = make(tmp_path / "s", tmp_path / "meta", llm=her)
        kept = current.remember_across_runs("The fire.")
        gone = current.remember_across_runs("The bridge.")
        forgot = current.forget_across_runs(gone), current.forget_across_runs(gone)
        await current.reply("hi")
        await current.close()
        return forgot, system_prompt(her), [m.id for m in current.across_runs()], kept

    forgot, prompt, left, kept = asyncio.run(scenario())
    assert forgot == (True, False)
    assert "- The fire." in prompt and "The bridge." not in prompt
    assert left == [kept]


def test_her_background_work_never_writes_them(tmp_path):
    meta = tmp_path / "meta"

    async def scenario():
        current = make(
            tmp_path / "s",
            meta,
            llm=Her("I like strong black tea. How are you?"),
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
        current.remember_across_runs("The fire.")
        await current.reply(f"{DAWN}. teach me Japanese", conversation_id="a")
        await current.settle()
        remembered = current.memories("a")
        await current.close()
        return remembered, (meta / "across_runs.jsonl").read_text(encoding="utf-8").splitlines()

    remembered, lines = asyncio.run(scenario())
    assert remembered == ["Dawn works at a print shop."]
    assert len(lines) == 1 and "The fire." in lines[0]


def test_without_meta_dir_there_is_no_such_memory(tmp_path):
    current = make(tmp_path / "s", None)
    assert current.across_runs() == []
    assert current.forget_across_runs("anything") is False
    with pytest.raises(RuntimeError):
        current.remember_across_runs("The fire.")
    with_meta = make(tmp_path / "s2", tmp_path / "meta")
    with pytest.raises(ValueError):
        with_meta.remember_across_runs("   ")
    with pytest.raises(ValueError):
        CompanionSettings(across_runs_in_context=-1)
