"""Memory correction / forgetting example using the local memory layer."""

from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager
from ai_character_engine.state.models import CharacterState


def record(manager: MemoryManager, text: str):
    state = CharacterState().snapshot()
    return manager.record_interaction(
        character_id="demo",
        event=CharacterEvent.user_message(text),
        response=LLMResponse(text="知道了。"),
        state_before=state,
        state_after=state,
    )


store = InMemoryMemoryStore()
memory = MemoryManager(store=store, auto_consolidate_threshold=None)

record(memory, "我最喜歡七武士")
record(memory, "其實我現在不再那麼喜歡七武士了")

print("After correction:")
for item in store.list_for_character("demo"):
    print(item.status, item.summary, "supersedes=", item.supersedes)

record(memory, "我喜歡咖啡")
record(memory, "忘記我喜歡咖啡這件事")

print("\nAfter forgetting:")
for item in store.list_for_character("demo"):
    print(item.status, item.summary)
