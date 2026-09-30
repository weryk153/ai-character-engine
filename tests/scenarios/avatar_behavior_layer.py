"""Offline avatar-behavior demo; no network, audio device, or renderer."""

from __future__ import annotations

import json
import random

from ai_character_engine.avatar import (
    AvatarBehaviorConfig,
    AvatarBehaviorPhase,
    AvatarBehaviorRuntime,
    GazeRequest,
    GazeTarget,
    GestureRequest,
)


def show(label: str, bundle) -> None:
    print(f"\n[{label}]")
    if bundle is None:
        print("(no cue due)")
        return
    print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))


def main() -> None:
    behavior = AvatarBehaviorRuntime(
        AvatarBehaviorConfig(
            blink_interval_min_ms=900,
            blink_interval_max_ms=900,
            head_interval_min_ms=1200,
            head_interval_max_ms=1200,
        ),
        rng=random.Random(7),
    )
    behavior.start(now_ms=0)
    show("session start", behavior.tick(now_ms=0))

    behavior.schedule_gaze(
        GazeRequest(GazeTarget("user", yaw_deg=6, pitch_deg=-2), duration_ms=2500),
        now_ms=100,
    )
    behavior.begin_turn("demo-turn", now_ms=100)
    show("thinking", behavior.tick(now_ms=100, force=True))

    behavior.schedule_gesture(
        GestureRequest(
            "small_explain",
            duration_ms=800,
            allowed_phases=(AvatarBehaviorPhase.SPEAKING,),
        ),
        now_ms=100,
    )
    behavior.set_phase(AvatarBehaviorPhase.SPEAKING, now_ms=500)
    show("speaking + gesture", behavior.tick(now_ms=500, force=True))

    show("natural blink", behavior.tick(now_ms=900))
    show("natural head motion", behavior.tick(now_ms=1200))

    behavior.end_turn(now_ms=1800)
    show("back to idle", behavior.tick(now_ms=1800, force=True))


if __name__ == "__main__":
    main()
