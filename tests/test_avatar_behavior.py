import random

import pytest

from ai_character_engine.avatar import (
    AvatarBehaviorConfig,
    AvatarBehaviorPhase,
    AvatarBehaviorRuntime,
    GazeRequest,
    GazeTarget,
    GestureRequest,
)
from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import LiveCharacterOrchestrator, LiveEventType, LiveRuntimeConfig
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.state import CharacterState
from ai_character_engine.voice import AudioChunk, AudioFormat, SynthesizedAudio


class Clock:
    def __init__(self, seconds=0.0):
        self.value = seconds

    def __call__(self):
        return self.value

    def advance_ms(self, ms):
        self.value += ms / 1000.0


class LLM:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="hello")


class TTS:
    async def synthesize(self, text, *, voice=None):
        return SynthesizedAudio(b"\x00\x00" * 320, AudioFormat(), duration_ms=20)


class Sink:
    async def play(self, chunk):
        pass


def runtime():
    return CharacterRuntime(
        character=CharacterProfile("v028", "V028", "behavior test"),
        llm=LLM(),
    )


def fixed_config(**overrides):
    values = dict(
        blink_interval_min_ms=100.0,
        blink_interval_max_ms=100.0,
        head_interval_min_ms=200.0,
        head_interval_max_ms=200.0,
        idle_motion_refresh_ms=500.0,
    )
    values.update(overrides)
    return AvatarBehaviorConfig(**values)


def test_initial_tick_emits_default_gaze_and_idle_motion():
    behavior = AvatarBehaviorRuntime(fixed_config(), rng=random.Random(1))
    behavior.start(now_ms=1000)
    bundle = behavior.tick(now_ms=1000)
    assert bundle is not None
    assert bundle.phase is AvatarBehaviorPhase.IDLE
    assert bundle.gaze.target.name == "camera"
    assert bundle.idle_motion.motion == "breathing"
    assert bundle.blink is None


def test_natural_blink_and_head_motion_use_due_time_and_bounds():
    behavior = AvatarBehaviorRuntime(fixed_config(), rng=random.Random(7))
    behavior.start(now_ms=0)
    behavior.tick(now_ms=0)
    blink = behavior.tick(now_ms=100)
    assert blink is not None and blink.blink is not None
    head = behavior.tick(now_ms=200)
    assert head is not None and head.head_motion is not None
    assert abs(head.head_motion.yaw_delta_deg) <= behavior.config.head_max_yaw_deg
    assert abs(head.head_motion.pitch_delta_deg) <= behavior.config.head_max_pitch_deg
    assert abs(head.head_motion.roll_delta_deg) <= behavior.config.head_max_roll_deg


def test_gaze_priority_wins_then_returns_to_default():
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    low = GazeRequest(GazeTarget("desk", yaw_deg=-15), duration_ms=500, priority=10)
    high = GazeRequest(GazeTarget("user", yaw_deg=8), duration_ms=200, priority=100)
    behavior.schedule_gaze(low, now_ms=0)
    behavior.schedule_gaze(high, now_ms=0)
    first = behavior.tick(now_ms=0, force=True)
    assert first.gaze.target.name == "user"
    later = behavior.tick(now_ms=250, force=True)
    assert later.gaze.target.name == "desk"
    expired = behavior.tick(now_ms=600, force=True)
    assert expired.gaze.target.name == "camera"


def test_gesture_group_arbitration_uses_priority_and_is_one_shot():
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    behavior.schedule_gesture(GestureRequest("small_wave", priority=10), now_ms=0)
    behavior.schedule_gesture(GestureRequest("big_wave", priority=90), now_ms=0)
    bundle = behavior.tick(now_ms=0, force=True)
    assert [gesture.gesture for gesture in bundle.gestures] == ["big_wave"]
    next_bundle = behavior.tick(now_ms=1)
    assert next_bundle is None or not next_bundle.gestures


def test_gesture_can_wait_for_allowed_phase():
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    behavior.set_phase(AvatarBehaviorPhase.LISTENING, now_ms=0)
    behavior.schedule_gesture(
        GestureRequest(
            "explain",
            duration_ms=500,
            allowed_phases=(AvatarBehaviorPhase.SPEAKING,),
        ),
        now_ms=0,
    )
    listening = behavior.tick(now_ms=0, force=True)
    assert listening.gestures == ()
    behavior.set_phase(AvatarBehaviorPhase.SPEAKING, now_ms=100)
    speaking = behavior.tick(now_ms=100, force=True)
    assert [gesture.gesture for gesture in speaking.gestures] == ["explain"]


def test_turn_interruption_clears_turn_scoped_requests_but_keeps_session_gaze():
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    behavior.begin_turn("turn-1", now_ms=0)
    behavior.schedule_gaze(
        GazeRequest(GazeTarget("user"), duration_ms=2000, turn_scoped=False),
        now_ms=0,
    )
    behavior.schedule_gesture(GestureRequest("point", duration_ms=1000, turn_scoped=True), now_ms=0)
    behavior.interrupt_turn(now_ms=10)
    behavior.end_turn(now_ms=20, reason="interrupted")
    bundle = behavior.tick(now_ms=20, force=True)
    assert bundle.gaze.target.name == "user"
    assert bundle.gestures == ()
    assert bundle.phase is AvatarBehaviorPhase.IDLE


def test_behavior_runtime_does_not_mutate_character_state():
    state = CharacterState(emotion="neutral", trust=40)
    before = (state.emotion, state.trust)
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    behavior.schedule_gesture(GestureRequest("nod"), now_ms=0)
    behavior.tick(now_ms=0, force=True)
    assert (state.emotion, state.trust) == before


def test_behavior_bundle_is_renderer_data_without_audio_bytes():
    behavior = AvatarBehaviorRuntime(fixed_config())
    behavior.start(now_ms=0)
    data = behavior.tick(now_ms=0).to_dict()

    def walk(value):
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, list):
            return all(walk(item) for item in value)
        return not isinstance(value, bytes)

    assert walk(data)
    assert "vrm" not in data


async def test_idle_run_once_can_emit_behavior_without_character_turn():
    clock = Clock()
    behavior = AvatarBehaviorRuntime(fixed_config())
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        behavior_runtime=behavior,
        monotonic=clock,
    )
    events = await live.run_once()
    assert [event.type for event in events] == [LiveEventType.AVATAR_BEHAVIOR_CUE]
    assert events[0].data["phase"] == "idle"
    assert behavior.turn_id is None


async def test_foreground_turn_emits_thinking_and_idle_behavior_phases():
    clock = Clock()
    behavior = AvatarBehaviorRuntime(fixed_config())
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        behavior_runtime=behavior,
        monotonic=clock,
    )
    live.submit_text("hi")
    events = await live.run_once()
    phases = [e.data["phase"] for e in events if e.type is LiveEventType.AVATAR_BEHAVIOR_CUE]
    assert phases[0] == "thinking"
    assert phases[-1] == "idle"
    assert behavior.turn_id is None


async def test_tts_turn_enters_speaking_behavior_phase():
    clock = Clock()
    behavior = AvatarBehaviorRuntime(fixed_config())
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        tts=TTS(),
        audio_sink=Sink(),
        behavior_runtime=behavior,
        monotonic=clock,
    )
    live.submit_text("hi")
    events = await live.run_once()
    phases = [e.data["phase"] for e in events if e.type is LiveEventType.AVATAR_BEHAVIOR_CUE]
    assert phases == ["thinking", "speaking", "idle"]


async def test_no_behavior_runtime_preserves_v027_nonstreaming_contract():
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        tts=TTS(),
        audio_sink=Sink(),
        config=LiveRuntimeConfig(streaming_output=False),
    )
    live.submit_text("hi")
    events = await live.run_once()
    assert [e.type for e in events] == [
        LiveEventType.TURN_STARTED,
        LiveEventType.REPLY,
        LiveEventType.TTS_AUDIO,
    ]


def test_poll_behavior_allows_host_render_loop_without_running_character_loop():
    clock = Clock()
    behavior = AvatarBehaviorRuntime(fixed_config())
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        behavior_runtime=behavior,
        monotonic=clock,
    )
    first = live.poll_behavior()
    assert first and first[0].data["gaze"]["target"]["name"] == "camera"
    clock.advance_ms(100)
    blink = live.poll_behavior()
    assert blink and blink[0].data["blink"] is not None


def test_invalid_behavior_ranges_are_rejected():
    with pytest.raises(ValueError):
        AvatarBehaviorConfig(blink_interval_min_ms=500, blink_interval_max_ms=100)
    with pytest.raises(ValueError):
        GazeTarget("bad", yaw_deg=100)


def test_orchestrator_schedule_helpers_and_behavior_reset_contract():
    clock = Clock()
    behavior = AvatarBehaviorRuntime(fixed_config())
    live = LiveCharacterOrchestrator(
        CharacterHostBridge(runtime()),
        behavior_runtime=behavior,
        monotonic=clock,
    )
    gaze_id = live.schedule_gaze(GazeRequest(GazeTarget("screen"), duration_ms=500))
    gesture_id = live.schedule_gesture(GestureRequest("nod", duration_ms=500, turn_scoped=False))
    assert gaze_id and gesture_id
    event = live.reset_behavior(reason="renderer_unload")
    assert event.type is LiveEventType.AVATAR_BEHAVIOR_RESET
    assert event.data["reason"] == "renderer_unload"
    assert set(event.data["reset"]) == {"gaze", "blink", "head", "idle_motion", "gestures"}
    assert not behavior.active
