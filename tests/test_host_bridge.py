import asyncio
import base64
import struct

import pytest

from ai_character_engine import CharacterProfile, CharacterRuntime
from ai_character_engine.host import (
    CharacterHostBridge,
    HostBridgeConfig,
    HostBridgeError,
    image_from_host,
)
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.llm.local import OpenAICompatibleChatClient
from ai_character_engine.state import StatePatch
from ai_character_engine.tools.models import ToolCall, ToolDefinition
from ai_character_engine.vision import VisionAnalysis, VisionFrame, VisionPipeline
from ai_character_engine.vision.providers import CallableVisionProvider
from ai_character_engine.vision.sampling import FrameGate


class LLM:
    def __init__(self):
        self.messages = []
        self.fail = False
        self.wait = False
        self.started = asyncio.Event()

    async def generate(self, messages, *, tools=None):
        self.messages = messages
        self.started.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError("secret API_KEY must not appear")
        return LLMResponse(text="你好！")


def make_bridge(llm=None, **kwargs):
    runtime = CharacterRuntime(
        character=CharacterProfile("mei", "Mei", "A character"), llm=llm or LLM()
    )
    return CharacterHostBridge(runtime, **kwargs)


def frame():
    raw = b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 1, 1)
    image = image_from_host(
        base64.b64encode(raw).decode(), "image/png", max_bytes=1024
    )
    return VisionFrame(image=image, source_type="camera")


async def test_text_history_and_connection_isolation():
    first, second = make_bridge(), make_bridge()
    await first.process("我叫 Alex")
    await first.process("記得嗎？")
    assert any(m.content == "我叫 Alex" for m in first.runtime.llm.messages)
    assert second.runtime.history == []
    assert len(first.runtime.history) == 4


async def test_vision_is_transient_and_one_character_turn():
    llm = LLM()
    vision = VisionPipeline(
        provider=CallableVisionProvider(
            lambda image, prompt: VisionAnalysis("red mug", "fake")
        ),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=False),
    )
    bridge = make_bridge(llm, vision=vision)
    result = await bridge.process("看到了什麼？", frames=(frame(), frame()))
    assert result.event.type == "multimodal_user_message"
    assert len(result.event.payload["vision"]) == 2
    assert result.event.payload["vision_memory"] == "ephemeral"
    assert "red mug" in llm.messages[-2].content
    assert llm.messages[-1].content == "看到了什麼？"
    assert all("red mug" not in m.content for m in bridge.runtime.history)
    assert len(bridge.runtime.history) == 2
    await bridge.process("同一张圖呢？", frames=(frame(),))


async def test_vision_disabled_and_bad_images():
    with pytest.raises(HostBridgeError, match="disabled"):
        await make_bridge().process("picture", frames=(frame(),))
    for data in (
        "https://private/image.png",
        "/etc/passwd",
        "%%%",
        "",
        "data:image/jpeg;base64,YQ==",
    ):
        with pytest.raises(HostBridgeError):
            image_from_host(data, "image/png", max_bytes=128)
    with pytest.raises(HostBridgeError, match="size"):
        image_from_host("A" * 1024, "image/png", max_bytes=10)


async def test_error_restores_history_and_state():
    bridge = make_bridge()
    await bridge.process("hello")
    history = list(bridge.runtime.history)

    class Policy:
        def on_event(self, event, state):
            return StatePatch(energy_delta=-10)

    bridge.runtime.state_policy = Policy()
    bridge.runtime.llm.fail = True
    with pytest.raises(HostBridgeError, match="processing failed") as caught:
        await bridge.process("fail")
    assert "secret" not in str(caught.value)
    assert bridge.runtime.history == history
    assert bridge.runtime.state.energy == 100


async def test_timeout_then_recovery():
    llm = LLM()
    llm.wait = True
    bridge = make_bridge(
        llm, config=HostBridgeConfig(turn_timeout_seconds=0.02)
    )
    with pytest.raises(HostBridgeError, match="timed out"):
        await bridge.process("timeout")
    assert bridge.runtime.history == []
    llm.wait = False
    assert (await bridge.process("retry")).text == "你好！"


async def test_cancel_busy_close_and_no_old_reply_corruption():
    llm = LLM()
    bridge = make_bridge(llm)
    await bridge.process("first")
    history = list(bridge.runtime.history)
    llm.wait = True
    llm.started.clear()
    task = asyncio.create_task(bridge.process("second"))
    await llm.started.wait()
    with pytest.raises(HostBridgeError, match="already running"):
        await bridge.process("third")
    bridge.interrupt("unheard")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.runtime.history == history
    await bridge.close()
    await bridge.close()
    with pytest.raises(HostBridgeError, match="closed"):
        await bridge.process("fourth")


async def test_interrupt_keeps_only_heard_reply_once():
    bridge = make_bridge()
    await bridge.process("hello")
    bridge.interrupt("你")
    bridge.interrupt("changed")
    assert bridge.runtime.history[-1].content == "你 [Interrupted by user]"


async def test_interrupt_without_heard_audio_records_unpadded_marker():
    bridge = make_bridge()
    await bridge.process("hello")
    bridge.interrupt("  ")
    assert bridge.runtime.history[-1].content == "[Interrupted by user]"


def roles(bridge):
    return [message.role for message in bridge.runtime.history]


async def test_restored_history_starts_with_the_user_side_of_an_exchange():
    bridge = make_bridge()
    bridge.restore_history(
        [Message("assistant", "orphan"), Message("assistant", "again"),
         Message("user", "hi"), Message("assistant", "hello")]
    )
    assert [m.content for m in bridge.runtime.history] == ["hi", "hello"]
    bridge.runtime.max_history_messages = 5
    bridge.restore_history(
        [Message(role, str(index)) for index in range(3) for role in ("user", "assistant")]
    )
    assert [m.content for m in bridge.runtime.history] == ["1", "1", "2", "2"]


async def test_interrupted_turn_at_history_cap_keeps_complete_exchanges():
    llm = LLM()
    bridge = make_bridge(llm)
    bridge.runtime.max_history_messages = 5
    for text in ("one", "two"):
        await bridge.process(text)
    bridge.record_interrupted_turn("three", "heard")
    assert roles(bridge) == ["user", "assistant", "user", "assistant"]
    assert bridge.runtime.history[0].content == "two"
    await bridge.process("four")
    assert [m.role for m in llm.messages[:2]] == ["system", "user"]


async def test_vision_turn_at_history_cap_keeps_tool_cycle_intact():
    class ToolLLM:
        async def generate(self, messages, *, tools=None):
            if messages[-1].role == "tool":
                return LLMResponse(text="12:34")
            if tools and "time" in messages[-1].content:
                return LLMResponse(tool_calls=(ToolCall("one", "clock", {}),))
            return LLMResponse(text="seen")

    vision = VisionPipeline(
        provider=CallableVisionProvider(
            lambda image, prompt: VisionAnalysis("red mug", "fake")
        ),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=False),
    )
    bridge = make_bridge(ToolLLM(), vision=vision)
    bridge.runtime.tool_registry.register(
        ToolDefinition("clock", "Read clock", {"type": "object", "properties": {}}),
        lambda: "12:34",
    )
    bridge.runtime.max_history_messages = 5
    await bridge.process("time?")
    assert roles(bridge) == ["user", "assistant", "tool", "assistant"]
    await bridge.process("look", frames=(frame(),))
    assert roles(bridge) == ["user", "assistant"]


async def test_oversized_context_reports_budget_not_provider():
    bridge = make_bridge()
    bridge.runtime.character.description = "燈" * 8000
    with pytest.raises(HostBridgeError, match="context budget") as caught:
        await bridge.process("hello")
    assert "燈" not in str(caught.value)
    assert bridge.runtime.history == []


async def test_reply_cut_off_before_visible_text_reports_output_limit():
    from types import SimpleNamespace as NS

    async def create(**kwargs):
        if not kwargs.get("stream"):
            return NS(choices=[NS(message=NS(content=""), finish_reason="length")])

        async def chunks():
            yield NS(choices=[NS(index=0, delta=NS(content=None), finish_reason=None)])
            yield NS(choices=[NS(index=0, delta=NS(content=None), finish_reason="length")])

        return chunks()

    llm = OpenAICompatibleChatClient(
        model="local", base_url="http://localhost/v1",
        client=NS(chat=NS(completions=NS(create=create))),
    )
    assert (await llm.generate([Message("user", "hello")])).metadata["finish_reason"] == "length"

    async def ignore(_):
        pass

    for callback in (None, ignore):
        bridge = make_bridge(llm)
        with pytest.raises(HostBridgeError, match="output limit"):
            await bridge.process("hello", on_text_delta=callback)
        assert bridge.runtime.history == []


async def test_empty_reply_without_output_limit_keeps_generic_message():
    class Empty:
        async def generate(self, messages, *, tools=None):
            return LLMResponse(text=" ")

    with pytest.raises(HostBridgeError, match="empty reply"):
        await make_bridge(Empty()).process("hello")


async def test_proactive_skip_memory_and_zero_history():
    bridge = make_bridge()
    result = await bridge.process("observe", proactive=True, skip_memory=True)
    assert result.event.type == "proactive_observation"
    assert bridge.runtime.history == []
    bridge.runtime.max_history_messages = 0
    bridge.restore_history([Message("user", "secret")])
    assert bridge.runtime.history == []


async def test_tool_call_uses_real_runtime_executor():
    class ToolLLM:
        async def generate(self, messages, *, tools=None):
            if messages[-1].role == "tool":
                return LLMResponse(text=messages[-1].content)
            return LLMResponse(tool_calls=(ToolCall("one", "clock", {}),))

    bridge = make_bridge(ToolLLM())
    bridge.runtime.tool_registry.register(
        ToolDefinition("clock", "Read clock", {"type": "object", "properties": {}}),
        lambda: "12:34",
    )
    result = await bridge.process("time?")
    assert result.tool_results[0].output == "12:34"
    assert result.text == "12:34"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        HostBridgeConfig(turn_timeout_seconds=timeout)


async def test_request_options_preserved_and_reserved_fields_rejected():
    from types import SimpleNamespace

    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    llm = OpenAICompatibleChatClient(
        model="local",
        base_url="http://localhost/v1",
        client=client,
        request_options={
            "temperature": 0.2,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    await llm.generate([Message("user", "hello")])
    assert requests[0]["temperature"] == 0.2
    assert requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    for options in ({"messages": []}, {"extra_body": {"stream": True}}):
        with pytest.raises(ValueError):
            OpenAICompatibleChatClient(
                model="local",
                base_url="http://localhost/v1",
                client=client,
                request_options=options,
            )


@pytest.mark.asyncio
async def test_a_turn_that_is_taken_back_takes_its_note_with_it():
    """Left behind, the note would be put in front of the same words when the
    user says them again, in place of the note that belongs there."""
    bridge = make_bridge()
    await bridge.process("hello")
    kept = list(bridge.runtime.context_notes)
    bridge.runtime.state.apply(StatePatch(emotion="happy", reason="test"))

    await bridge.process("a remark of her own", skip_memory=True, proactive=True)

    assert bridge.runtime.context_notes == kept


async def test_the_turn_after_a_picture_extends_the_prompt_of_the_picture_turn():
    """The picture's description comes right before the user's words, which stay
    the last message (said last, the picture was answered instead of them), and
    is left out afterwards; everything before it stays as it was sent. An
    inference server that continues from the end of the previous prompt then
    reads again only from the description on. Rewriting the user's message after
    the turn made it read the conversation again on every turn with a camera on."""
    llm = LLM()
    vision = VisionPipeline(
        provider=CallableVisionProvider(
            lambda image, prompt: VisionAnalysis("red mug", "fake")
        ),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=False),
    )
    bridge = make_bridge(llm, vision=vision)
    await bridge.process("hello")
    await bridge.process("look", frames=(frame(),))
    picture_turn = list(llm.messages)
    assert picture_turn[-1] == Message("user", "look")
    assert "red mug" in picture_turn[-2].content

    await bridge.process("and now?")
    sent_again = picture_turn[:-2]
    assert llm.messages[: len(sent_again)] == sent_again
    assert all("red mug" not in m.content for m in llm.messages)


async def test_a_picture_the_host_says_is_unchanged_reuses_the_last_description():
    """A camera mostly shows the same thing turn after turn. When the host marks a
    frame unchanged, the last description of that source is told again instead of
    asking the vision model (about two seconds on a local 9B)."""
    calls = []

    def look(image, prompt):
        calls.append(1)
        return VisionAnalysis(f"red mug {len(calls)}", "fake")

    llm = LLM()
    vision = VisionPipeline(
        provider=CallableVisionProvider(look),
        frame_gate=FrameGate(min_interval_seconds=0, deduplicate=False),
    )
    bridge = make_bridge(llm, vision=vision)
    await bridge.process("look", frames=(frame(),))
    unchanged = frame()
    unchanged.metadata["unchanged"] = True
    await bridge.process("still?", frames=(unchanged,))
    assert len(calls) == 1
    assert "red mug 1" in llm.messages[-2].content

    await bridge.process("and now?", frames=(frame(),))
    assert len(calls) == 2
