"""Offline Goal / Motivation Runtime demonstration.

No network/model dependency is required. The example feeds already-structured
TaskProposal values through the same commit boundary used by background cognition.
"""
from __future__ import annotations

import asyncio

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.commit import CognitiveCommitCoordinator
from ai_character_engine.goals import GoalManager, GoalStatus
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.tasks import MultiTaskRuntime, TaskProposal


class OfflineLLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="offline", model="offline")


def proposal(event_id: str, objective: str, *, conflict_key: str | None = None) -> TaskProposal:
    return TaskProposal(
        target="cognition.goal_candidate",
        payload={
            "objective": objective,
            "horizon": "short_term",
            "urgency": 0.75,
            "conflict_key": conflict_key,
            "motivation_signals": [
                {
                    "kind": "explicit_request",
                    "strength": 0.8,
                    "source_type": "event",
                    "source_id": event_id,
                    "rationale": "The current authoritative event explicitly supports this action intention.",
                }
            ],
        },
        base_revision=0,
        source_task_id=f"task-{event_id}",
        confidence=0.9,
        provenance={
            "worker_kind": "goal_motivation",
            "foreground_event_id": event_id,
            "foreground_event_type": "user_message",
            "foreground_event_source": "user",
            "foreground_event_content": "Please keep helping with this objective.",
            "evidence_type": "user_instruction",
        },
    )


async def main() -> None:
    goals = GoalManager()
    runtime = CharacterRuntime(
        character=CharacterProfile(id="demo", name="Demo", description="Offline v0.35 demo."),
        llm=OfflineLLM(),
        goal_manager=goals,
    )
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime))

    first = await commits.commit(proposal("evt-1", "Finish deployment"))
    initial = goals.active_goals(character_id="demo")[0]
    print("first:", first.status.value, initial.id, "support=", initial.support_count,
          "motivation=", round(initial.motivation_score, 3))

    second = await commits.commit(proposal("evt-2", "Finish deployment"))
    reinforced = goals.active_goals(character_id="demo")[0]
    print("reinforced:", second.status.value, reinforced.id, "support=", reinforced.support_count,
          "same_id=", reinforced.id == initial.id,
          "motivation=", round(reinforced.motivation_score, 3))

    await commits.commit(proposal("evt-3", "Deploy now", conflict_key="deployment_strategy"))
    await commits.commit(proposal("evt-4", "Delay deployment", conflict_key="deployment_strategy"))
    conflict_group = [g for g in goals.goals(character_id="demo") if g.conflict_key == "deployment_strategy"]
    print("conflict:", [(g.objective, g.status.value) for g in conflict_group])

    delay = next(g for g in conflict_group if g.objective == "Delay deployment")
    goals.transition(
        character_id="demo",
        goal_id=delay.id,
        status=GoalStatus.RETIRED,
        reason="Host resolved the alternative action direction.",
    )
    print("resolved active:", [g.objective for g in goals.active_goals(character_id="demo")])


if __name__ == "__main__":
    asyncio.run(main())
