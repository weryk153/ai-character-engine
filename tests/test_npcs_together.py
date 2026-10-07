"""Several NPCs in one process: one model, one memory across runs each.

A game server runs every NPC of a scene against one local model and may hold
the same NPC open in two save slots at once.
"""
from __future__ import annotations

import asyncio

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import CharacterCompanion, CompanionSettings, ModelAccess
from tests.test_companion import DAWN, Foreground, dawns_work
from tests.test_companion import Worker as CountingWorker
from tests.test_conversation_plans import Her
from tests.test_game_time import GameClock

SILENT = {
    "emotion_every": 0,
    "mood_every": 0,
    "memory_every": 0,
    "self_memory_every": 0,
    "goal_every": 0,
    "reflection_every": 0,
    "summary_every": 0,
}


def npc(storage, *, llm, workers=None, meta=None, access=None, character="mira", **settings):
    return CharacterCompanion(
        character=CharacterProfile(id=character, name=character.title(), description="A keeper."),
        llm=llm,
        background_llm=workers or {},
        storage_dir=storage,
        settings=CompanionSettings(**{**SILENT, **settings}),
        clock=GameClock(),
        meta_dir=meta,
        model_access=access,
    )


def test_one_npcs_background_work_waits_while_another_npc_replies(tmp_path):
    async def scenario(shared: bool):
        access = ModelAccess(120.0) if shared else None
        gate = asyncio.Event()
        talking = Foreground(("Hello. ", "Welcome."), gate=gate)
        first = npc(tmp_path / f"a{shared}", llm=talking, access=access)
        worker = CountingWorker(dawns_work)
        second = npc(
            tmp_path / f"b{shared}",
            llm=Foreground(("Mm. ", "I see.")),
            workers={"memory": worker},
            access=access,
            memory_every=1,
        )
        replying = asyncio.create_task(first.reply("hi", conversation_id="a"))
        await talking.started.wait()
        await second.reply(DAWN, conversation_id="a")
        await asyncio.sleep(0.05)
        while_first_replied = worker.calls
        gate.set()
        await replying
        await second.settle()
        after = worker.calls
        await first.close()
        await second.close()
        return while_first_replied, after

    assert asyncio.run(scenario(True)) == (0, 1)
    assert asyncio.run(scenario(False)) == (1, 1)


def test_a_memory_across_runs_written_elsewhere_is_in_her_next_reply(tmp_path):
    meta = tmp_path / "meta"

    async def scenario():
        her = Her("Hello.", "Hello again.")
        in_slot_one = npc(tmp_path / "one", llm=her, meta=meta)
        in_slot_two = npc(tmp_path / "two", llm=Her("Mm."), meta=meta)
        await in_slot_one.reply("hi")
        before = her.calls[-1][0].content
        in_slot_two.remember_across_runs("The fire.")
        listed = [memory.text for memory in in_slot_one.across_runs()]
        await in_slot_one.reply("hi again")
        after = her.calls[-1][0].content
        await in_slot_one.close()
        await in_slot_two.close()
        return before, listed, after

    before, listed, after = asyncio.run(scenario())
    assert "The fire." not in before
    assert listed == ["The fire."]
    assert "- The fire." in after
