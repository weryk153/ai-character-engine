import pytest

from ai_character_engine.character.profile import CharacterProfile
from ai_character_engine.runtime.character_runtime import CharacterRuntime
from tests.fakes import FakeLLMClient


@pytest.fixture
def character() -> CharacterProfile:
    return CharacterProfile(id="test", name="Test", description="Test character")


@pytest.mark.asyncio
async def test_runtime_saves_history(character: CharacterProfile) -> None:
    fake = FakeLLMClient("hello")
    runtime = CharacterRuntime(character=character, llm=fake)

    response = await runtime.run_turn("Hi")

    assert response.text == "hello"
    assert [m.role for m in runtime.history] == ["user", "assistant"]
    assert runtime.history[0].content == "Hi"
    assert runtime.history[1].content == "hello"


@pytest.mark.asyncio
async def test_runtime_uses_previous_history(character: CharacterProfile) -> None:
    fake = FakeLLMClient("ok")
    runtime = CharacterRuntime(character=character, llm=fake)

    await runtime.run_turn("first")
    await runtime.run_turn("second")

    second_call = fake.calls[1]
    assert any(m.role == "user" and m.content == "first" for m in second_call)
    assert any(m.role == "assistant" and m.content == "ok" for m in second_call)


@pytest.mark.asyncio
async def test_runtime_trims_history(character: CharacterProfile) -> None:
    fake = FakeLLMClient("ok")
    runtime = CharacterRuntime(character=character, llm=fake, max_history_messages=2)

    await runtime.run_turn("first")
    await runtime.run_turn("second")

    assert len(runtime.history) == 2
    assert runtime.history[0].content == "second"
