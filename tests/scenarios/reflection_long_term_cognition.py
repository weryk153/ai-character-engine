"""Offline v0.34 reflection -> deterministic long-term belief demo."""

from ai_character_engine.long_term_cognition import (
    BeliefClaim,
    CognitionEvidenceRef,
    LongTermCognitionManager,
    ReflectionRecord,
)


CHARACTER_ID = "demo-character"
CLAIM_CONCISE = BeliefClaim("user", "prefers_response_style", "concise")
CLAIM_DETAILED = BeliefClaim("user", "prefers_response_style", "detailed")


def reflection(source_id: str, claim: BeliefClaim, insight: str) -> ReflectionRecord:
    return ReflectionRecord(
        character_id=CHARACTER_ID,
        insight=insight,
        confidence=0.90,
        evidence=(
            CognitionEvidenceRef(
                source_type="event",
                source_id=source_id,
                evidence_type="asserted_fact",
                excerpt=f"direct evidence from {source_id}",
                confidence=0.90,
            ),
        ),
        claim=claim,
        source_proposal_id=f"proposal-{source_id}",
        source_task_id=f"task-{source_id}",
    )


def show(manager: LongTermCognitionManager, label: str) -> None:
    print(f"\n{label}")
    print(f"reflections={len(manager.reflections(character_id=CHARACTER_ID))}")
    for belief in manager.beliefs(character_id=CHARACTER_ID):
        print(
            "belief",
            belief.id[:8],
            belief.status.value,
            f"{belief.claim.subject} {belief.claim.predicate} {belief.claim.object}",
            f"support={belief.support_count}",
            f"confidence={belief.confidence:.2f}",
        )
    active = manager.active_beliefs(character_id=CHARACTER_ID)
    print("active_context_values=", [item.claim.object for item in active])


def main() -> None:
    manager = LongTermCognitionManager()

    manager.commit_reflection(
        reflection("evt-1", CLAIM_CONCISE, "The user may prefer concise answers.")
    )
    show(manager, "1) one source: reflection only")

    manager.commit_reflection(
        reflection("evt-2", CLAIM_CONCISE, "Concise answers appear to be preferred.")
    )
    show(manager, "2) two independent sources: active belief")
    first_belief_id = manager.active_beliefs(character_id=CHARACTER_ID)[0].id

    manager.commit_reflection(
        reflection("evt-3", CLAIM_CONCISE, "The concise preference is reinforced.")
    )
    reinforced = manager.active_beliefs(character_id=CHARACTER_ID)[0]
    assert reinforced.id == first_belief_id
    show(manager, "3) reinforcement: same belief id, support grows")

    manager.commit_reflection(
        reflection("evt-4", CLAIM_DETAILED, "The user may instead prefer detailed answers.")
    )
    manager.commit_reflection(
        reflection("evt-5", CLAIM_DETAILED, "Detailed answers are independently supported.")
    )
    show(manager, "4) competing qualified value: beliefs become contested")
    assert manager.active_beliefs(character_id=CHARACTER_ID) == ()


if __name__ == "__main__":
    main()
