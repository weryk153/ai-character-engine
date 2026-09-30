"""Offline v0.39 multi-character isolation and sharing demonstration."""
from __future__ import annotations

import asyncio

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.multi_character import KnowledgeVisibility, MultiCharacterRuntime
from ai_character_engine.runtime import CharacterRuntime


class DemoLLM:
    def __init__(self, character_id: str) -> None:
        self.character_id = character_id

    async def generate(self, messages, *, tools=None):
        last = messages[-1]
        return LLMResponse(
            text=f"{self.character_id} observed: {last.content.splitlines()[-1]}",
            model="offline-demo",
        )


def make_runtime(character_id: str) -> CharacterRuntime:
    return CharacterRuntime(
        character=CharacterProfile(
            id=character_id,
            name=character_id.title(),
            description="Offline multi-character demo character",
        ),
        llm=DemoLLM(character_id),
        memory_scope_id=f"memory:{character_id}",
        cognition_scope_id=f"cognition:{character_id}",
        goal_scope_id=f"goal:{character_id}",
    )


async def main() -> None:
    multi = MultiCharacterRuntime(
        {"alice": make_runtime("alice"), "bob": make_runtime("bob")}
    )

    alice_turn = await multi.run_turn("alice", "hello Alice")
    print("alice:", alice_turn.text)
    print("bob history before exchange:", len(multi.runtime_for("bob").history))

    interaction = await multi.send_message(
        sender_character_id="alice",
        recipient_character_id="bob",
        content="I noticed the north door is open.",
    )
    print("exchange:", interaction.exchange.status.value)
    print("bob:", interaction.recipient_result.text)

    shared = multi.publish_shared_cognition(
        owner_character_id="alice",
        kind="observation",
        content="The lobby clock reads 12:00.",
        visibility=KnowledgeVisibility.PUBLIC,
        source_type="event",
        source_id="demo-clock",
    )
    observed = await multi.deliver_shared_cognition(
        shared.id, recipient_character_id="bob"
    )
    print("shared observation authoritative:", observed.event.payload["authoritative"])
    print("scheduler:", multi.scheduler.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
