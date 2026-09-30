"""Deterministic state transitions before character generation."""

import asyncio
import os

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.events import CharacterEvent
from ai_character_engine.llm.provider import OpenAIResponsesClient
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import (
    CharacterState,
    EventStateRule,
    RuleBasedStatePolicy,
    StatePatch,
)


async def main() -> None:
    character = CharacterProfile(
        id="demo",
        name="Mira",
        description="A calm virtual companion who hides embarrassment with dry humor.",
        personality=["observant", "dry humor", "slow to trust"],
        speaking_style=["short natural replies", "avoid exaggerated enthusiasm"],
    )

    policy = RuleBasedStatePolicy(
        event_rules=[
            EventStateRule(
                event_type="compliment_received",
                patch=StatePatch(
                    emotion="pleased",
                    trust_delta=1,
                    favorability_delta=3,
                    reason="compliment received",
                ),
            ),
            EventStateRule(
                event_type="late_night",
                patch=StatePatch(
                    emotion="tired",
                    energy_delta=-20,
                    reason="late-night fatigue",
                ),
            ),
        ]
    )

    llm = OpenAIResponsesClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    )
    runtime = CharacterRuntime(
        character=character,
        llm=llm,
        state=CharacterState(energy=75, trust=40, favorability=45),
        state_policy=policy,
    )

    event = CharacterEvent(
        type="compliment_received",
        source="user",
        content="You handled that really well.",
    )
    result = await runtime.process_event(event)

    print("state before:", result.state_before)
    print("state after :", result.state_after)
    print(f"{character.name}: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
