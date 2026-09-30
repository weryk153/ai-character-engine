"""Regressions from real-model twenty-turn text acceptance."""
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import LLMResponse
from ai_character_engine.memory import JsonlMemoryStore, MemoryManager
from ai_character_engine.state.models import CharacterState


def record(manager, text, reply):
    state = CharacterState().snapshot()
    return manager.record_interaction(character_id="test",
        event=CharacterEvent.user_message(text), response=LLMResponse(text=reply),
        state_before=state, state_after=state)


def test_hallucinated_answer_is_not_persisted_as_user_evidence(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = MemoryManager(store=JsonlMemoryStore(path))
    record(memory, "我的生日是哪天？", "你的生日是四月十二日。")
    reloaded = JsonlMemoryStore(path).list_for_character("test")
    assert reloaded == []
    assert memory.ledger.list_for_character("test")[0].response_text == "你的生日是四月十二日。"


def test_forget_cannot_be_bypassed_through_saved_recall_answer(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = MemoryManager(store=JsonlMemoryStore(path))
    original = record(memory, "我的測試代號是紫鷺731。", "知道了。")
    record(memory, "我的測試代號是什麼？", "紫鷺731。")
    record(memory, "請忘記我的測試代號紫鷺731。", "好的。")
    reloaded = MemoryManager(store=JsonlMemoryStore(path))
    records = reloaded.store.list_for_character("test")
    assert next(r for r in records if r.id == original.id).status == "forgotten"
    selected = reloaded.retrieve_for_event(character_id="test",
        event=CharacterEvent.user_message("我的測試代號是什麼？"))
    assert all("紫鷺731" not in r.record.summary for r in selected)
    assert len(memory.ledger.list_for_character("test")) == 3


def test_correction_does_not_leave_old_preference_in_recall_answer(tmp_path):
    memory = MemoryManager(store=JsonlMemoryStore(tmp_path / "memory.jsonl"))
    old = record(memory, "我最喜歡的飲料是無糖烏龍茶。", "好。")
    record(memory, "我平常最喜歡喝什麼？", "無糖烏龍茶。")
    record(memory, "更正，我最喜歡的飲料改成無糖紅茶，原本的無糖烏龍茶偏好已經過時。", "好。")
    records = memory.store.list_for_character("test")
    assert next(r for r in records if r.id == old.id).status == "superseded"
    assert all(r.metadata['source_content'] != "我平常最喜歡喝什麼？" for r in records)
    assert "無糖紅茶" in next(r for r in records if r.supersedes).summary


def test_repeated_name_questions_do_not_crowd_out_original_fact(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = MemoryManager(store=JsonlMemoryStore(path))
    original = record(memory, "我叫林澄，請叫我阿澄。", "好的。")
    for question in ("我的名字是什麼？", "你記得我的名字嗎？", "我的名字和稱呼是什麼？",
                     "你應該怎麼稱呼我？", "我之前說的名字是什麼？", "我的姓名是什麼？"):
        record(memory, question, "阿澄。")
    reloaded = MemoryManager(store=JsonlMemoryStore(path))
    selected = reloaded.retrieve_for_event(character_id="test",
        event=CharacterEvent.user_message("我的名字和你應該怎麼稱呼我？"))
    assert original.id in {r.record.id for r in selected}
    # Recall questions remain in the append-only ledger/history, not long-term fact memory.
    assert len(reloaded.store.list_for_character("test")) == 1


def test_user_fact_with_question_is_downranked_but_not_discarded():
    memory = MemoryManager()
    original = record(memory, "我喜歡黑澤明的七武士，你呢？", "我知道這部電影。")
    selected = memory.retrieve_for_event(character_id="test",
        event=CharacterEvent.user_message("我喜歡哪部黑澤明電影？"))
    assert selected[0].record.id == original.id
