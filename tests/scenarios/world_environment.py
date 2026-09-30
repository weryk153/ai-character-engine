from __future__ import annotations

from ai_character_engine.world import (
    WorldPatch,
    WorldPerceptionDeniedError,
    WorldPerceptionScope,
    WorldRuntime,
)


def main() -> None:
    world = WorldRuntime()
    event = world.apply_event(
        type="door_opened",
        content="The studio door opened.",
        patch=WorldPatch(set_values={"studio.door": "open"}, expected_revision=0),
        perception_scope=WorldPerceptionScope.DIRECT,
        observer_character_ids=("alice",),
    )

    alice = world.observe_event(event.id, character_id="alice")
    alice_event = world.character_event(alice)

    denied = False
    try:
        world.observe_event(event.id, character_id="bob")
    except WorldPerceptionDeniedError:
        denied = True

    print("world revision:", world.snapshot().revision)
    print("alice observed:", alice_event.type)
    print("bob denied:", denied)
    print("authoritative:", alice_event.payload["authoritative"])


if __name__ == "__main__":
    main()
