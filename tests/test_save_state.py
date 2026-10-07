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


def on_disk(storage):
    return {path.name: path.read_bytes() for path in sorted(storage.iterdir()) if path.is_file()}


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
        # She has a life of her own, on disk, that a refused save must not touch.
        life = lived(tmp_path / "second", clock)
        await a_day(life, clock)
        await life.close()

        refusals, kept = {}, []
        current = make(tmp_path / "second", clock)
        before = (everything(current), on_disk(tmp_path / "second"))
        for label, save in (("newer", newer), ("damaged", damaged), ("not a save", b"PK nonsense")):
            with pytest.raises(StateFormatError) as raised:
                current.import_state(save)
            refusals[label] = str(raised.value)
            kept.append((everything(current), on_disk(tmp_path / "second")) == before)

        await current.reply("hi", conversation_id="a")
        await current.settle()
        spoken = (everything(current), on_disk(tmp_path / "second"))
        with pytest.raises(RuntimeError):
            current.import_state(data)
        kept.append((everything(current), on_disk(tmp_path / "second")) == spoken)
        await current.close()

        other = make(tmp_path / "other", clock, character="yura")
        alone = everything(other)
        with pytest.raises(ValueError):
            other.import_state(data)
        kept.append(everything(other) == alone)
        other.import_state(data, allow_other_character=True)
        hers = (other.memories("a"), other.self_memories())
        await other.close()
        return refusals, kept, hers

    refusals, kept, hers = asyncio.run(scenario())
    assert "newer engine" in refusals["newer"]
    assert "memory.jsonl" in refusals["damaged"]
    assert "not a save" in refusals["not a save"]
    assert kept == [True, True, True, True, True]
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



def test_a_save_whose_manifest_is_not_one_is_refused(tmp_path):
    clock = GameClock()

    async def scenario():
        first = lived(tmp_path / "first", clock)
        await a_day(first, clock)
        data = await first.export_state()
        await first.close()
        current = make(None, clock)
        refused = []
        for change in (
            lambda manifest: {**manifest, "files": list(manifest["files"])},
            lambda manifest: {**manifest, "format_version": 0},
        ):
            save = _rewritten(
                data,
                lambda name, content: json.dumps(change(json.loads(content))).encode()
                if name == "manifest.json"
                else content,
            )
            with pytest.raises(StateFormatError):
                current.import_state(save)
            refused.append(True)
        return refused

    assert asyncio.run(scenario()) == [True, True]


class HeldDiary(DiaryWorker):
    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()


def test_a_diary_being_written_is_waited_for_and_keeps_a_save_from_loading(tmp_path):
    clock = GameClock()

    async def scenario():
        diary = HeldDiary()
        first = make(
            tmp_path / "first",
            clock,
            workers={"memory": Worker(dawns_work), "self_memory": Worker(her_tea), "diary": diary},
            memory_every=1,
            self_memory_every=1,
        )
        await first.reply(DAWN, conversation_id="a")
        await first.settle()
        clock.hours(1)
        writing = asyncio.create_task(first.write_diary())
        await asyncio.sleep(0.05)
        with pytest.raises(StateBusy):
            await first.export_state(timeout=0.05)
        diary.gate.set()
        await writing
        data = await first.export_state()
        clock.hours(1)
        await first.reply("good night", conversation_id="a")  # a day for the next entry
        await first.settle()
        await first.close()

        held = HeldDiary()
        before_she_speaks = make(
            tmp_path / "first", clock, workers={"diary": held}, diary_every_hours=0
        )
        clock.hours(25)
        pending = asyncio.create_task(before_she_speaks.write_diary())
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError):
            before_she_speaks.import_state(data)
        held.gate.set()
        await pending
        await before_she_speaks.close()
        loaded = make(None, clock)
        loaded.import_state(data)
        return loaded.diary()

    assert len(asyncio.run(scenario())) == 1


def test_her_pending_fix_and_how_the_user_seemed_survive_a_load(tmp_path):
    from collections import deque

    clock = GameClock()

    async def scenario():
        first = make(tmp_path / "first", clock)
        await first.reply("hello", conversation_id="a")
        first._reply_note = ("a", "Hello there.", ("Say it in fewer words.",))
        first._emotions = {"a": deque([{"emotion": "tired", "valence": -0.4, "stance": 0.0}], maxlen=8)}
        data = await first.export_state()
        await first.close()
        loaded = make(None, clock)
        loaded.import_state(data)
        return loaded._reply_note, {key: list(value) for key, value in loaded._emotions.items()}

    note, emotions = asyncio.run(scenario())
    assert note == ("a", "Hello there.", ("Say it in fewer words.",))
    assert emotions == {"a": [{"emotion": "tired", "valence": -0.4, "stance": 0.0}]}


def test_how_the_user_has_been_is_not_dated_after_her_clock_on_a_load(tmp_path):
    clock = GameClock()

    async def scenario():
        first = make(tmp_path / "first", clock)
        first.runtime.state.custom["user_state"] = {
            "energy": "low", "mood_trend": "down", "concerns": [], "evidence": [], "updated_at": clock(),
        }
        data = await first.export_state()
        await first.close()
        clock.hours(-48)
        loaded = make(None, clock)
        loaded.import_state(data)
        return loaded.snapshot().user_state

    lately = asyncio.run(scenario())
    assert lately is not None and lately.updated_at == clock()


def test_a_save_waits_for_her_line_and_then_her_background_work(tmp_path):
    clock = GameClock()

    async def scenario():
        speaking = asyncio.Event()
        thinking = asyncio.Event()
        current = make(
            tmp_path,
            clock,
            llm=Foreground(("Hello. ", "Welcome back."), gate=speaking),
            workers={"memory": GatedWorker(dawns_work, gate=thinking)},
            memory_every=1,
        )
        replying = asyncio.create_task(current.reply(DAWN, conversation_id="a"))
        await asyncio.sleep(0.02)
        saving = asyncio.create_task(current.export_state())
        await asyncio.sleep(0.02)
        during_her_line = saving.done()
        speaking.set()
        await replying
        await asyncio.sleep(0.02)
        during_her_work = saving.done()
        thinking.set()
        data = await saving
        later = make(None, clock)
        later.import_state(data)
        await current.close()
        return during_her_line, during_her_work, later.memories("a")

    assert asyncio.run(scenario()) == (False, False, ["Dawn works at a print shop."])
