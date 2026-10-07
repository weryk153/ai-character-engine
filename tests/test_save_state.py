"""Saves: a game keeps her state in its own save and loads it back.

A new game starts with a new companion and a fresh (or no) storage_dir;
loading a save gives a new companion everything she kept when it was made.
"""
from __future__ import annotations

import asyncio
import io
import json
import zipfile

import pytest

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import (
    CharacterCompanion,
    CompanionSettings,
    StateBusy,
    StateFormatError,
    load_state_file,
    save_state_file,
)
from tests.test_companion import BLACK_TEA, DAWN, DiaryWorker, Foreground, dawns_work, her_tea
from tests.test_companion import Worker as GatedWorker
from tests.test_conversation_plans import THOUGHT, Her, Worker, goals_citing_the_event
from tests.test_game_time import GameClock

TEACH = "teach me Japanese"
MOOD = {"mood": "happy", "intensity": 0.8, "confidence": 0.9, "evidence": []}


def make(storage, clock, *, llm=None, workers=None, character="mei", **settings):
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
        character=CharacterProfile(id=character, name="Mei", description="A researcher."),
        llm=llm or Her(BLACK_TEA + "How are you?"),
        background_llm=workers or {},
        storage_dir=storage,
        settings=CompanionSettings(**every),
        clock=clock,
    )


def lived(storage, clock, llm=None):
    """A companion that keeps something of every kind once she has talked."""
    return make(
        storage,
        clock,
        llm=llm,
        workers={
            "memory": Worker(dawns_work),
            "self_memory": Worker(her_tea),
            "goal": Worker(goals_citing_the_event),
            "reflection": Worker(THOUGHT),
            "mood": Worker(MOOD),
            "diary": DiaryWorker(),
        },
        memory_every=1,
        self_memory_every=1,
        goal_every=1,
        reflection_every=1,
        mood_every=1,
    )


async def a_day(current, clock):
    """Two conversations, the diary of the day, and "a" at hand again."""
    await current.reply("hello from the other room", conversation_id="b")
    await current.settle()
    await current.reply(f"{DAWN}. {TEACH}", conversation_id="a")
    await current.settle()
    clock.hours(1)
    await current.write_diary()


def everything(current):
    return {
        "snapshot in a": current.snapshot(),
        "memories of a": current.memories("a"),
        "memories of b": current.memories("b"),
        "about herself": current.self_memories(),
        "diary": current.diary(),
        "holds a": current.has_conversation("a"),
        "holds b": current.has_conversation("b"),
    }


@pytest.mark.parametrize("on_disk", [True, False])
def test_a_save_loaded_into_a_new_companion_is_her_as_she_was(tmp_path, on_disk):
    clock = GameClock()

    async def scenario():
        her = Her(BLACK_TEA + "How are you?", "Still here.", "Go on.")
        first = lived(tmp_path / "first", clock, her)
        await a_day(first, clock)
        data = await first.export_state()
        before = everything(first)

        her_again = Her("Go on.")
        second = lived(tmp_path / "second" if on_disk else None, clock, her_again)
        second.import_state(data)
        after = everything(second)

        # The next turn is the same turn for both.
        await first.reply("go on", conversation_id="a")
        await second.reply("go on", conversation_id="a")
        prompts = [[(m.role, m.content) for m in llm.calls[-1]] for llm in (her, her_again)]
        await first.close()
        await second.close()
        return before, after, prompts

    before, after, (first_prompt, second_prompt) = asyncio.run(scenario())
    assert before["memories of a"] == ["Dawn works at a print shop."]
    assert before["about herself"] == ["Mei likes strong black tea."]
    assert before["snapshot in a"].goals and before["snapshot in a"].thoughts
    assert before["diary"]
    assert after == before
    assert second_prompt == first_prompt


def test_a_save_waits_for_the_reply_under_way(tmp_path):
    clock = GameClock()

    async def scenario():
        gate = asyncio.Event()
        current = make(tmp_path, clock, llm=Foreground(("Hello. ", "Welcome back."), gate=gate))
        replying = asyncio.create_task(current.reply("hi", conversation_id="a"))
        await asyncio.sleep(0.01)
        saving = asyncio.create_task(current.export_state())
        await asyncio.sleep(0.01)
        waited = not saving.done()
        gate.set()
        await replying
        data = await saving
        later = make(tmp_path / "later", clock)
        later.import_state(data)
        history = [(m.role, m.content) for m in later.runtime.history]
        await current.close()
        await later.close()
        return waited, history

    waited, history = asyncio.run(scenario())
    assert waited
    assert history == [("user", "hi"), ("assistant", "Hello. Welcome back.")]


def test_a_save_waits_for_her_background_work_and_gives_up_without_saving(tmp_path):
    clock = GameClock()

    async def scenario():
        gate = asyncio.Event()
        current = make(
            tmp_path,
            clock,
            workers={"memory": GatedWorker(dawns_work, gate=gate)},
            memory_every=1,
        )
        await current.reply(DAWN, conversation_id="a")
        with pytest.raises(StateBusy):
            await current.export_state(timeout=0.05)
        gate.set()
        data = await current.export_state()
        later = make(tmp_path / "later", clock)
        later.import_state(data)
        remembered = later.memories("a")
        await current.close()
        await later.close()
        return remembered

    assert asyncio.run(scenario()) == ["Dawn works at a print shop."]


def _rewritten(data: bytes, change) -> bytes:
    """The save with its files changed by ``change(name, content)``."""
    source = zipfile.ZipFile(io.BytesIO(data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for name in source.namelist():
            target.writestr(name, change(name, source.read(name)))
    return buffer.getvalue()


def test_a_save_that_cannot_be_hers_is_refused_and_changes_nothing(tmp_path):
    clock = GameClock()

    async def scenario():
        first = lived(tmp_path / "first", clock)
        await a_day(first, clock)
        data = await first.export_state()
        await first.close()

        newer = _rewritten(
            data,
            lambda name, content: json.dumps({**json.loads(content), "format_version": 2}).encode()
            if name == "manifest.json"
            else content,
        )
        damaged = _rewritten(
            data,
            lambda name, content: content.replace(b"print shop", b"bakery")
            if name == "memory.jsonl"
            else content,
        )
        refusals = {}
        current = make(tmp_path / "second", clock)
        for label, save in (("newer", newer), ("damaged", damaged), ("not a save", b"PK nonsense")):
            with pytest.raises(StateFormatError) as raised:
                current.import_state(save)
            refusals[label] = str(raised.value)
        unchanged = current.memories("a")

        await current.reply("hi", conversation_id="a")
        with pytest.raises(RuntimeError):
            current.import_state(data)
        await current.close()

        other = make(tmp_path / "other", clock, character="yura")
        with pytest.raises(ValueError):
            other.import_state(data)
        other.import_state(data, allow_other_character=True)
        hers = (other.memories("a"), other.self_memories())
        await other.close()
        return refusals, unchanged, hers

    refusals, unchanged, hers = asyncio.run(scenario())
    assert "newer engine" in refusals["newer"]
    assert "memory.jsonl" in refusals["damaged"]
    assert "not a save" in refusals["not a save"]
    assert unchanged == []
    assert hers == (["Dawn works at a print shop."], ["Mei likes strong black tea."])


def test_a_save_replaces_what_her_storage_held_and_stays_after_a_restart(tmp_path):
    clock = GameClock()

    async def scenario():
        first = lived(tmp_path / "first", clock)
        await a_day(first, clock)
        data = await first.export_state()
        await first.close()

        storage = tmp_path / "second"
        other_life = make(
            storage, clock, workers={"memory": Worker(dawns_work)}, memory_every=1
        )
        await other_life.reply("I am Dawn and I work at a print shop. I also bake.", conversation_id="z")
        await other_life.settle()
        await other_life.close()

        loaded = make(storage, clock)
        loaded.import_state(data)
        await loaded.close()
        restarted = make(storage, clock)
        kept = everything(restarted), restarted.memories("z")
        await restarted.close()
        return kept

    kept, other_conversation = asyncio.run(scenario())
    assert kept["memories of a"] == ["Dawn works at a print shop."]
    assert kept["about herself"] == ["Mei likes strong black tea."]
    assert kept["diary"]
    assert other_conversation == []
    assert not any(path.name.endswith(".tmp") for path in (tmp_path / "second").iterdir())


def test_a_save_written_to_a_file_loads_back(tmp_path):
    clock = GameClock()

    async def scenario():
        first = lived(tmp_path / "first", clock)
        await a_day(first, clock)
        path = tmp_path / "slot1.sav"
        await save_state_file(first, path)
        await first.close()
        second = make(None, clock)
        load_state_file(second, path)
        remembered = second.memories("a")
        await second.close()
        return remembered, sorted(p.name for p in tmp_path.iterdir())

    remembered, names = asyncio.run(scenario())
    assert remembered == ["Dawn works at a print shop."]
    assert "slot1.sav" in names and "slot1.sav.tmp" not in names


def test_a_save_loaded_after_her_clock_was_turned_back_keeps_her_mood_fading(tmp_path):
    clock = GameClock()

    async def scenario():
        first = lived(tmp_path / "first", clock)
        await a_day(first, clock)
        data = await first.export_state()
        await first.close()
        clock.hours(-48)
        second = make(None, clock)
        second.import_state(data)
        dated = second.snapshot().mood_updated_at
        await second.close()
        return dated

    assert asyncio.run(scenario()) == clock()


def test_what_the_host_knows_is_taken_back_after_a_load_as_before(tmp_path):
    clock = GameClock()

    async def scenario():
        first = make(tmp_path / "first", clock, llm=Her("Hm.", "Oh no."))
        await first.reply("is the bridge there?", conversation_id="a", notes=["- The bridge is intact."])
        data = await first.export_state()
        second = make(tmp_path / "second", clock, llm=Her("Oh no."))
        second.import_state(data)
        prompts = []
        for current in (first, second):
            await current.reply("and now?", conversation_id="a", notes=["- The bridge is destroyed."])
            prompts.append("\n".join(m.content for m in current.runtime.llm.calls[-1]))
            await current.close()
        return prompts

    first_prompt, second_prompt = asyncio.run(scenario())
    assert "The bridge is intact." not in first_prompt
    assert "The bridge is intact." not in second_prompt
    assert "The bridge is destroyed." in second_prompt
