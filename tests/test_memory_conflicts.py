"""Memory conflicts: a new fact about the user against what she already holds.

"I changed jobs" next to "works at X" is not two facts to keep side by side.
A fact that moved on replaces the old one; two that cannot both be true and
cannot be put in order are asked about, once; a detail added to a fact leaves
both as they are.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ai_character_engine.memory.conflicts import (
    CONFLICT_RELATIONS,
    apply_conflicts,
    conflict_candidates,
    resolve_conflict,
    take_unasked_conflicts,
    unresolved_conflicts,
)
from ai_character_engine.memory.models import MemoryRecord

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def fact(summary, *, minutes=0, kind="fact", **kwargs):
    return MemoryRecord(
        character_id="mei:a",
        summary=summary,
        kind=kind,
        created_at=T0 + timedelta(minutes=minutes),
        metadata={"evidence_type": "asserted_fact"},
        **kwargs,
    )


# --- candidates: cheap, before any model is asked --------------------------------


def test_the_relations_are_three_and_fixed():
    assert CONFLICT_RELATIONS == ("supersedes", "contradicts", "refines")


def test_a_fact_on_the_same_topic_is_a_candidate_though_no_word_is_shared():
    job = fact("使用者在台積電上班")
    cat = fact("使用者養了一隻叫小麥的貓", minutes=1)
    new = fact("使用者換工作了", minutes=2)
    assert conflict_candidates(new, [job, cat, new]) == [job]


def test_a_shared_name_makes_a_candidate():
    old = fact("The user's sister Anna lives in Berlin")
    other = fact("The user likes rainy days", minutes=1)
    new = fact("Anna is studying medicine", minutes=2)
    assert conflict_candidates(new, [old, other, new]) == [old]


@pytest.mark.parametrize(
    "earlier, newer",
    [
        ("使用者在準備考試", "使用者考完試了"),
        ("使用者叫小明", "使用者叫阿傑"),
    ],
)
def test_two_characters_in_common_or_a_name_make_a_candidate(earlier, newer):
    old = fact(earlier)
    new = fact(newer, minutes=1)
    assert conflict_candidates(new, [old, new]) == [old]


def test_the_user_as_the_subject_of_every_fact_shares_nothing():
    old = fact("使用者喜歡下雨天")
    new = fact("使用者昨天去了海邊", minutes=1)
    assert conflict_candidates(new, [old, new]) == []
    old = fact("The user likes rainy days")
    new = fact("The user went to the beach yesterday", minutes=1)
    assert conflict_candidates(new, [old, new]) == []


def test_only_earlier_active_facts_about_the_user_are_candidates():
    gone = replace(fact("使用者在台積電上班"), status="superseded")
    forgotten = replace(fact("使用者在聯發科上班", minutes=1), status="forgotten")
    summary = fact("使用者聊了上班的事", minutes=2, kind="conversation_summary")
    later = fact("使用者在鴻海上班", minutes=9)
    new = fact("使用者換工作了", minutes=5)
    assert conflict_candidates(new, [gone, forgotten, summary, later, new]) == []


def test_at_most_five_candidates_most_alike_first():
    olds = [fact(f"使用者在第{n}家公司上班", minutes=n) for n in range(7)]
    closest = fact("使用者換工作，現在在新公司上班", minutes=8)
    new = fact("使用者換工作了，在新公司上班", minutes=9)
    found = conflict_candidates(new, [*olds, closest, new])
    assert len(found) == 5
    assert found[0] is closest


def test_embeddings_alike_make_a_candidate_when_both_have_one():
    old = fact("A", embedding=(1.0, 0.0, 0.1))
    far = fact("B", minutes=1, embedding=(0.0, 1.0, 0.0))
    new = fact("C", minutes=2, embedding=(0.95, 0.05, 0.1))
    assert conflict_candidates(new, [old, far, new]) == [old]


# --- what each relation does -----------------------------------------------------


def test_a_fact_that_moved_on_replaces_the_old_one():
    old = fact("使用者在台積電上班")
    new = fact("使用者換工作了", minutes=1)
    records, applied = apply_conflicts(
        [old, new], new.id, [{"old_id": old.id, "relation": "supersedes", "reason": "換工作"}]
    )
    by_id = {record.id: record for record in records}
    assert by_id[old.id].status == "superseded"
    assert by_id[old.id].superseded_by == new.id
    assert by_id[new.id].is_active
    assert by_id[new.id].supersedes == (old.id,)
    assert applied == ({"old_id": old.id, "relation": "supersedes", "reason": "換工作"},)
    assert unresolved_conflicts(records) == []


def test_two_facts_that_cannot_both_be_true_are_both_kept_and_marked():
    old = fact("使用者27歲")
    new = fact("使用者25歲", minutes=1)
    records, _ = apply_conflicts(
        [old, new], new.id, [{"old_id": old.id, "relation": "contradicts", "reason": "年齡不同"}]
    )
    by_id = {record.id: record for record in records}
    assert by_id[old.id].is_active and by_id[new.id].is_active
    assert by_id[new.id].metadata["conflict_with"] == [old.id]
    ((later, earlier, reason),) = unresolved_conflicts(records)
    assert (later.id, earlier.id, reason) == (new.id, old.id, "年齡不同")


def test_a_detail_added_leaves_both_facts_as_they_are():
    old = fact("使用者養貓")
    new = fact("使用者的貓叫小麥", minutes=1)
    records, applied = apply_conflicts(
        [old, new], new.id, [{"old_id": old.id, "relation": "refines", "reason": ""}]
    )
    by_id = {record.id: record for record in records}
    assert by_id[old.id] == old
    assert by_id[new.id].is_active and by_id[new.id].status == "active"
    assert by_id[new.id].metadata["refines"] == [old.id]
    assert "conflict_with" not in by_id[new.id].metadata
    assert len(applied) == 1
    assert unresolved_conflicts(records) == []


@pytest.mark.parametrize(
    "item",
    [
        {"old_id": "nobody", "relation": "supersedes"},
        {"old_id": "OLD", "relation": "replaces"},
        {"old_id": "NEW", "relation": "supersedes"},
        {"relation": "supersedes"},
        "supersedes",
    ],
)
def test_an_answer_off_the_list_changes_nothing(item):
    old = fact("使用者在台積電上班")
    new = fact("使用者換工作了", minutes=1)
    if isinstance(item, dict) and "old_id" in item:
        item = {**item, "old_id": {"OLD": old.id, "NEW": new.id}.get(item["old_id"], item["old_id"])}
    records, applied = apply_conflicts([old, new], new.id, [item])
    assert records == [old, new]
    assert applied == ()


def test_a_fact_already_gone_is_not_replaced_again():
    old = replace(fact("使用者在台積電上班"), status="forgotten")
    new = fact("使用者換工作了", minutes=1)
    records, applied = apply_conflicts(
        [old, new], new.id, [{"old_id": old.id, "relation": "supersedes"}]
    )
    assert records == [old, new]
    assert applied == ()


# --- asking about it, once, and settling it --------------------------------------


def contradicting():
    old = fact("使用者27歲")
    new = fact("使用者25歲", minutes=1)
    records, _ = apply_conflicts(
        [old, new], new.id, [{"old_id": old.id, "relation": "contradicts", "reason": ""}]
    )
    return old, new, records


def test_a_contradiction_is_asked_about_once():
    old, new, records = contradicting()
    records, asked = take_unasked_conflicts(records)
    assert [(later.id, earlier.id) for later, earlier, _ in asked] == [(new.id, old.id)]
    records, asked = take_unasked_conflicts(records)
    assert asked == []
    # Asked about, it is still not settled.
    assert len(unresolved_conflicts(records)) == 1


@pytest.mark.parametrize("keep", ["old", "new"])
def test_settling_a_contradiction_keeps_the_one_chosen(keep):
    old, new, records = contradicting()
    kept, other = (old, new) if keep == "old" else (new, old)
    records, settled = resolve_conflict(records, kept.id)
    by_id = {record.id: record for record in records}
    assert settled
    assert by_id[kept.id].is_active
    assert by_id[other.id].status == "superseded"
    assert by_id[other.id].superseded_by == kept.id
    assert unresolved_conflicts(records) == []


def test_settling_what_is_in_no_contradiction_changes_nothing():
    old, new, records = contradicting()
    unrelated = fact("使用者養貓", minutes=2)
    records = [*records, unrelated]
    assert resolve_conflict(records, unrelated.id) == (records, False)


def test_a_newer_fact_that_replaces_one_side_settles_it():
    old, new, records = contradicting()
    newest = fact("使用者其實26歲", minutes=3)
    records, _ = apply_conflicts(
        [*records, newest], newest.id, [{"old_id": new.id, "relation": "supersedes"}]
    )
    assert unresolved_conflicts(records) == []


# --- the worker: asked only about the candidates ---------------------------------

import asyncio  # noqa: E402
import json  # noqa: E402

from ai_character_engine import (  # noqa: E402
    BackgroundCognitionConfig,
    BackgroundCognitionKind,
    BackgroundCognitionRuntime,
    BackgroundWorkerSpec,
    CharacterProfile,
    CharacterRuntime,
    CognitiveModelRouter,
    CognitiveModelRuntime,
    CognitiveRole,
    CognitiveRolePolicy,
    ModelEndpoint,
    MultiTaskRuntime,
    TaskStatus,
)
from ai_character_engine.llm.models import LLMResponse, Message  # noqa: E402


CHANGED = {"changed": True, "word": "換工作", "both_true": False}


class Judge:
    """The conflict model: its answer, then ``confirm`` for each relation
    asked about once more; a str is answered as it is."""

    def __init__(self, answer, confirm=CHANGED):
        self.answer = answer
        self.confirm = confirm
        self.calls: list[list[Message]] = []

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        answer = self.answer if len(self.calls) == 1 else self.confirm
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return LLMResponse(text=text, model="judge")


class Says:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="嗯。", model="foreground")


def conflict_runtime(judge, *, language=""):
    runtime = CharacterRuntime(
        character=CharacterProfile(id="mei", name="Mei", description="test"), llm=Says()
    )
    tasks = MultiTaskRuntime(runtime)
    models = CognitiveModelRuntime(
        endpoints=(ModelEndpoint(endpoint_id="judge", client=judge),),
        router=CognitiveModelRouter(
            policies={CognitiveRole.MEMORY_CONFLICT: CognitiveRolePolicy(primary_endpoint_ids=("judge",))}
        ),
    )
    background = BackgroundCognitionRuntime(
        tasks,
        models,
        config=BackgroundCognitionConfig(
            worker_specs=(BackgroundWorkerSpec(BackgroundCognitionKind.MEMORY_CONFLICT),)
        ),
    )
    background.output_language = language
    return tasks, background


def said(summary, words, *, minutes=0):
    record = fact(summary, minutes=minutes)
    return replace(
        record,
        metadata={**record.metadata, "background_provenance": {"evidence": [words]}},
    )


async def judged(answer, *, candidates=None, language="", confirm=CHANGED):
    old = fact("使用者在台積電上班")
    new = said("使用者換工作了", "我上個月換工作了", minutes=1)
    judge = Judge(answer, confirm)
    tasks, background = conflict_runtime(judge, language=language)
    async with tasks:
        handle = await background.schedule_memory_conflict(
            new, [old] if candidates is None else candidates
        )
        if handle is None:
            return judge, old, new, None
        result = await handle.wait()
    return judge, old, new, result


@pytest.mark.asyncio
async def test_the_model_is_asked_about_the_new_fact_against_the_candidates_only():
    from ai_character_engine.cognition.background import MEMORY_CONFLICT_TARGET

    judge, old, new, result = await judged(
        {"conflicts": [{"old_id": "m1", "relation": "supersedes", "reason": "換工作"}]}
    )
    assert result.status is TaskStatus.SUCCEEDED, result.error
    (proposal,) = result.output.proposals
    assert proposal.target == MEMORY_CONFLICT_TARGET
    assert proposal.payload == {
        "new_id": new.id,
        "conflicts": [{"old_id": old.id, "relation": "supersedes", "reason": "換工作"}],
    }
    assert proposal.provenance["worker_kind"] == "memory_conflict"
    assert proposal.provenance["candidates"] == [old.id]
    assert proposal.confidence == 1.0
    (system, user), (_, question) = judge.calls
    assert "使用者在台積電上班" in question.content and "我上個月換工作了" in question.content
    assert "supersedes" in system.content and "contradicts" in system.content
    assert "使用者換工作了" in user.content
    assert "我上個月換工作了" in user.content
    assert "m1: 使用者在台積電上班" in user.content


@pytest.mark.asyncio
async def test_an_id_that_was_not_a_candidate_or_a_relation_off_the_list_is_dropped():
    other = fact("使用者養貓")
    _, old, _, result = await judged(
        {
            "conflicts": [
                {"old_id": "m2", "relation": "supersedes"},
                {"old_id": other.id, "relation": "supersedes"},
                {"old_id": "m1", "relation": "replaces"},
                {"old_id": "m1", "relation": "Contradicts", "reason": 3},
            ]
        }
    )
    (proposal,) = result.output.proposals
    assert proposal.payload["conflicts"] == [
        {"old_id": old.id, "relation": "contradicts", "reason": "3"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [{"conflicts": []}, {"conflicts": "none"}, {}])
async def test_no_relation_is_no_proposal(answer):
    _, _, _, result = await judged(answer)
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output.proposals == ()


@pytest.mark.asyncio
async def test_a_bad_answer_fails_the_job_alone():
    _, _, _, result = await judged("not json")
    assert result.status is TaskStatus.FAILED


@pytest.mark.asyncio
async def test_without_candidates_no_model_is_asked():
    judge, _, _, result = await judged({"conflicts": []}, candidates=[])
    assert result is None
    assert judge.calls == []


@pytest.mark.asyncio
async def test_the_conflict_worker_is_not_run_after_every_turn():
    judge = Judge({"conflicts": []})
    tasks, background = conflict_runtime(judge)
    async with tasks:
        result = await background.run_turn("我換工作了")
        await background.collect_all()
    assert result.text == "嗯。"
    assert judge.calls == []
    assert [event.action for event in background.events()] == ["not_applicable"]


# --- the coordinator: candidates on commit, the answer applied --------------------

from ai_character_engine.commit import CognitiveCommitCoordinator, CommitStatus  # noqa: E402
from ai_character_engine.memory import InMemoryMemoryStore, MemoryManager  # noqa: E402
from ai_character_engine.tasks import TaskProposal  # noqa: E402


def committing():
    store = InMemoryMemoryStore()
    engine = CharacterRuntime(
        character=CharacterProfile(id="mei", name="Mei", description="test"),
        llm=Says(),
        memory_manager=MemoryManager(store=store),
    )
    tasks = MultiTaskRuntime(engine)
    return engine, store, CognitiveCommitCoordinator(tasks)


def memory_proposal(summary):
    return TaskProposal(
        target="memory.append_candidate",
        payload={"summary": summary, "kind": "fact", "importance": 0.7},
        base_revision=0,
        source_task_id="task-1",
        confidence=0.9,
        provenance={
            "worker_kind": "memory_extraction",
            "foreground_event_id": f"evt-{summary}",
            "evidence_type": "asserted_fact",
            "evidence": [summary],
        },
    )


def conflict_proposal(new_id, conflicts, candidates):
    from ai_character_engine.cognition.background import MEMORY_CONFLICT_TARGET

    return TaskProposal(
        target=MEMORY_CONFLICT_TARGET,
        payload={"new_id": new_id, "conflicts": conflicts},
        base_revision=0,
        source_task_id="task-2",
        confidence=1.0,
        provenance={
            "worker_kind": "memory_conflict",
            "evidence_type": "asserted_fact",
            "new_id": new_id,
            "candidates": list(candidates),
        },
    )


@pytest.mark.asyncio
async def test_a_memory_committed_names_the_earlier_facts_it_may_change():
    engine, store, commits = committing()
    first = await commits.commit(memory_proposal("使用者在台積電上班"))
    other = await commits.commit(memory_proposal("使用者養貓"))
    assert first.metadata["conflict_candidates"] == []
    assert other.metadata["conflict_candidates"] == []
    new = await commits.commit(memory_proposal("使用者換工作了"))
    assert new.metadata["conflict_candidates"] == [first.applied_record_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", CONFLICT_RELATIONS)
async def test_the_answer_is_applied_to_the_memories_held(relation):
    engine, store, commits = committing()
    old = (await commits.commit(memory_proposal("使用者在台積電上班"))).applied_record_id
    new = (await commits.commit(memory_proposal("使用者換工作了"))).applied_record_id
    result = await commits.commit(
        conflict_proposal(new, [{"old_id": old, "relation": relation, "reason": ""}], [old])
    )
    assert result.status is CommitStatus.COMMITTED, result.reason
    by_id = {record.id: record for record in store.list_for_character(engine.memory_scope_id)}
    assert by_id[old].is_active is (relation != "supersedes")
    assert by_id[new].is_active
    assert result.metadata["applied"] == [{"old_id": old, "relation": relation, "reason": ""}]


@pytest.mark.asyncio
async def test_a_fact_the_model_was_not_shown_is_not_touched():
    engine, store, commits = committing()
    old = (await commits.commit(memory_proposal("使用者在台積電上班"))).applied_record_id
    new = (await commits.commit(memory_proposal("使用者換工作了"))).applied_record_id
    before = store.list_for_character(engine.memory_scope_id)
    result = await commits.commit(
        conflict_proposal(new, [{"old_id": old, "relation": "supersedes"}], [])
    )
    assert (result.status, result.reason) == (CommitStatus.REJECTED, "memory_conflict_without_relation")
    assert store.list_for_character(engine.memory_scope_id) == before


@pytest.mark.asyncio
async def test_a_conflict_about_a_memory_no_longer_held_is_rejected():
    engine, store, commits = committing()
    old = (await commits.commit(memory_proposal("使用者在台積電上班"))).applied_record_id
    new = (await commits.commit(memory_proposal("使用者換工作了"))).applied_record_id
    store.replace_for_character(
        engine.memory_scope_id,
        [
            replace(record, status="forgotten") if record.id == new else record
            for record in store.list_for_character(engine.memory_scope_id)
        ],
    )
    result = await commits.commit(
        conflict_proposal(new, [{"old_id": old, "relation": "supersedes"}], [old])
    )
    assert result.status is CommitStatus.REJECTED
    assert all(
        record.status != "superseded"
        for record in store.list_for_character(engine.memory_scope_id)
    )


@pytest.mark.asyncio
async def test_the_same_answer_is_applied_once():
    engine, store, commits = committing()
    old = (await commits.commit(memory_proposal("使用者27歲"))).applied_record_id
    new = (await commits.commit(memory_proposal("使用者25歲"))).applied_record_id
    conflicts = [{"old_id": old, "relation": "contradicts", "reason": ""}]
    assert (await commits.commit(conflict_proposal(new, conflicts, [old]))).committed
    again = await commits.commit(conflict_proposal(new, conflicts, [old]))
    assert again.status is CommitStatus.DUPLICATE
    (record,) = [r for r in store.list_for_character(engine.memory_scope_id) if r.id == new]
    assert record.metadata["conflict_with"] == [old]


# --- the companion: asked after a memory is written, applied, asked about once ---

from ai_character_engine.companion import (  # noqa: E402
    CharacterCompanion,
    CompanionSettings,
    MemoryConflict,
)
from ai_character_engine.context.builder import is_turn_context  # noqa: E402
from tests.fakes import system_context  # noqa: E402
from tests.test_companion import Foreground, Worker, run  # noqa: E402

JOB = "我在台積電上班"
NEW_JOB = "我上個月換工作了"
FACTS = {JOB: "使用者在台積電上班", NEW_JOB: "使用者換工作了", "我27歲": "使用者27歲", "我25歲": "使用者25歲"}


def facts_said(messages):
    lines = messages[1].content.split("User lines to extract from:")[1]
    items = [
        {"summary": summary, "kind": "fact", "importance": 0.7, "confidence": 0.9, "evidence": said}
        for said, summary in FACTS.items()
        if f"- {said}\n" in lines + "\n"
    ]
    return {"items": items, "confidence": 0.9, "evidence": []}


def judging(relation):
    def answer(messages):
        if "Earlier fact:\n" in messages[1].content:
            # Asked once more: a change only where one was said.
            moved = relation == "supersedes"
            return {"changed": moved, "word": "換工作" if moved else "", "both_true": False}
        assert "Earlier facts about the user:\nm1:" in messages[1].content
        return {"conflicts": [{"old_id": "m1", "relation": relation, "reason": "r"}]}

    return answer


def a_companion(tmp_path, workers, *, llm=None, **settings):
    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=llm or Foreground(),
        background_llm=workers,
        storage_dir=tmp_path / "engine",
        settings=CompanionSettings(
            **{
                "emotion_every": 0,
                "reply_check_every": 0,
                "mood_every": 0,
                "memory_every": 1,
                "goal_every": 0,
                "reflection_every": 0,
                "self_memory_every": 0,
                **settings,
            }
        ),
    )


def newest_notes(call) -> str:
    notes = [message.content for message in call if is_turn_context(message)]
    return notes[-1] if notes else ""


async def two_facts(current, first, second, conversation_id="a"):
    await current.reply(first, conversation_id=conversation_id)
    await current.settle()
    await current.reply(second, conversation_id=conversation_id)
    await current.settle()


def test_memory_conflicts_are_on_by_default():
    assert CompanionSettings().memory_conflicts is True


def test_a_fact_that_moved_on_leaves_her_mind(tmp_path):
    async def scenario():
        llm = Foreground()
        judge = Worker(judging("supersedes"))
        current = a_companion(
            tmp_path, {"memory": Worker(facts_said), "memory_conflict": judge}, llm=llm
        )
        await two_facts(current, JOB, NEW_JOB)
        await current.reply("你記得我在哪上班嗎", conversation_id="a")
        context = system_context(llm.calls[-1])
        await current.close()
        return judge.calls, current.memories("a"), current.memory_conflicts("a"), context

    calls, memories, conflicts, context = run(scenario())
    assert calls == 2
    assert memories == ["使用者換工作了"]
    assert conflicts == []
    assert "台積電" not in context


def test_two_facts_that_cannot_both_be_true_are_asked_about_once(tmp_path):
    async def scenario():
        llm = Foreground()
        current = a_companion(
            tmp_path,
            {"memory": Worker(facts_said), "memory_conflict": Worker(judging("contradicts"))},
            llm=llm,
            language="繁體中文",
        )
        await two_facts(current, "我27歲", "我25歲")
        conflicts = current.memory_conflicts("a")
        await current.reply("今天好累", conversation_id="a")
        asked = newest_notes(llm.calls[-1])
        await current.settle()
        await current.reply("想睡了", conversation_id="a")
        # The note stays where it was said in the conversation; it is not
        # given again.
        again = system_context(llm.calls[-1]).count("關於使用者")
        await current.close()
        return conflicts, asked, again, current.memories("a")

    conflicts, asked, again, memories = run(scenario())
    (conflict,) = conflicts
    assert isinstance(conflict, MemoryConflict)
    assert (conflict.summary, conflict.earlier_summary, conflict.reason) == (
        "使用者25歲",
        "使用者27歲",
        "r",
    )
    assert "For the next reply only: 關於使用者：之前記得「使用者27歲」，現在聽到「使用者25歲」" in asked
    assert again == 1
    assert memories == ["使用者27歲", "使用者25歲"]


def test_the_note_is_in_english_without_a_language_and_with_latin_facts(tmp_path):
    async def scenario():
        llm = Foreground()
        facts = {"I am 27": "The user is 27 years old", "I am 25": "The user is 25 years old"}

        def said_in_english(messages):
            lines = messages[1].content.split("User lines to extract from:")[1]
            return {
                "items": [
                    {"summary": summary, "kind": "fact", "confidence": 0.9, "evidence": said}
                    for said, summary in facts.items()
                    if f"- {said}\n" in lines + "\n"
                ],
                "confidence": 0.9,
                "evidence": [],
            }

        current = a_companion(
            tmp_path,
            {"memory": Worker(said_in_english), "memory_conflict": Worker(judging("contradicts"))},
            llm=llm,
        )
        await two_facts(current, "I am 27", "I am 25")
        await current.reply("so tired", conversation_id="a")
        await current.close()
        return newest_notes(llm.calls[-1])

    assert (
        'About the user: earlier they said "The user is 27 years old", now "The user is 25 '
        'years old" — ask which is right, lightly, when it fits.'
    ) in run(scenario())


@pytest.mark.parametrize("keep", ["earlier", "newer"])
def test_the_host_settles_a_contradiction_with_the_fact_to_keep(tmp_path, keep):
    async def scenario():
        current = a_companion(
            tmp_path,
            {"memory": Worker(facts_said), "memory_conflict": Worker(judging("contradicts"))},
        )
        await two_facts(current, "我27歲", "我25歲")
        (conflict,) = current.memory_conflicts("a")
        kept = conflict.earlier_id if keep == "earlier" else conflict.id
        settled = current.resolve_conflict(kept, conversation_id="a")
        await current.close()
        return settled, current.memories("a"), current.memory_conflicts("a")

    settled, memories, conflicts = run(scenario())
    assert settled is True
    assert memories == (["使用者27歲"] if keep == "earlier" else ["使用者25歲"])
    assert conflicts == []


def test_a_detail_added_changes_nothing_she_holds(tmp_path):
    async def scenario():
        current = a_companion(
            tmp_path,
            {"memory": Worker(facts_said), "memory_conflict": Worker(judging("refines"))},
        )
        await two_facts(current, JOB, NEW_JOB)
        await current.close()
        return current.memories("a"), current.memory_conflicts("a")

    assert run(scenario()) == (["使用者在台積電上班", "使用者換工作了"], [])


@pytest.mark.parametrize("judge", ["down", "nonsense", "slow"])
def test_a_judge_that_fails_costs_nothing_of_the_memory(tmp_path, judge):
    async def scenario():
        gate = asyncio.Event()
        worker = {
            "down": Worker({}, fail=True),
            "nonsense": Worker(lambda messages: ["not", "an", "object"]),
            "slow": Worker({}, gate=gate),
        }[judge]
        current = a_companion(
            tmp_path,
            {"memory": Worker(facts_said), "memory_conflict": worker},
            call_timeout_seconds=0.05,
        )
        await two_facts(current, JOB, NEW_JOB)
        await current.close()
        return current.memories("a"), worker.calls

    memories, calls = run(scenario())
    assert memories == ["使用者在台積電上班", "使用者換工作了"]
    assert calls == 1


def test_turned_off_no_model_is_asked(tmp_path):
    async def scenario():
        judge = Worker(judging("supersedes"))
        current = a_companion(
            tmp_path,
            {"memory": Worker(facts_said), "memory_conflict": judge},
            memory_conflicts=False,
        )
        await two_facts(current, JOB, NEW_JOB)
        await current.close()
        return judge.calls, current.memories("a")

    assert run(scenario()) == (0, ["使用者在台積電上班", "使用者換工作了"])


def test_a_mapping_without_a_memory_conflict_client_asks_no_model(tmp_path):
    async def scenario():
        memory = Worker(facts_said)
        current = a_companion(tmp_path, {"memory": memory})
        await two_facts(current, JOB, NEW_JOB)
        await current.close()
        return memory.calls, current.memories("a")

    assert run(scenario()) == (2, ["使用者在台積電上班", "使用者換工作了"])


def test_one_client_for_every_worker_judges_conflicts_too(tmp_path):
    async def scenario():
        def answer(messages):
            if "User lines to extract from:" in messages[1].content:
                return facts_said(messages)
            return judging("supersedes")(messages)

        current = a_companion(tmp_path, Worker(answer))
        await two_facts(current, JOB, NEW_JOB)
        await current.close()
        return current.memories("a")

    assert run(scenario()) == ["使用者換工作了"]


def test_what_was_settled_and_asked_survives_a_restart(tmp_path):
    async def first_life():
        llm = Foreground()
        current = a_companion(
            tmp_path,
            {
                "memory": Worker(facts_said),
                "memory_conflict": Worker(
                    lambda messages: judging(
                        "supersedes" if "台積電" in messages[1].content else "contradicts"
                    )(messages)
                ),
            },
            llm=llm,
        )
        await two_facts(current, JOB, NEW_JOB)
        await two_facts(current, "我27歲", "我25歲")
        await current.reply("嗯", conversation_id="a")
        asked = newest_notes(llm.calls[-1])
        await current.close()
        return asked

    async def second_life():
        llm = Foreground()
        current = a_companion(tmp_path, {}, llm=llm)
        conflicts = current.memory_conflicts("a")
        await current.reply("嗯", conversation_id="a")
        await current.close()
        return current.memories("a"), conflicts, system_context(llm.calls[-1])

    asked = run(first_life())
    memories, conflicts, notes = run(second_life())
    assert "使用者27歲" in asked
    assert memories == ["使用者換工作了", "使用者27歲", "使用者25歲"]
    assert [(c.earlier_summary, c.summary) for c in conflicts] == [("使用者27歲", "使用者25歲")]
    assert "關於使用者" not in notes


def test_the_conflict_judge_comes_right_after_memory(tmp_path):
    from ai_character_engine.companion.companion import _WORKERS

    names = [worker.name for worker in _WORKERS]
    assert names.index("memory_conflict") == names.index("memory") + 1


def test_the_prompts_of_the_other_workers_are_unchanged():
    import hashlib

    from ai_character_engine.cognition.background import _SYSTEM_PROMPTS

    expected = {
        "memory_extraction": "551f13bbfb9cbba7ec135dae315933205ef9e8c1774bab7c186326b189a632ba",
        "emotion_analysis": "e649ea9e806653250ccd8ba7023352e41fdff93323a7a98931474ec2c48d323f",
        "conversation_summary": "49f94e82327cead4d3429ef55bf0de5d883686f176bee92b05d5263f6865d3f5",
        "reflection": "4a2296027b52bb378af083fc6a94fd53793eadac94bd5b11594256bbcedb812c",
        "goal_motivation": "35805251bc7b7db7a77304702d77c4991c6a056e22b60f92da8a6380abf976be",
        "self_memory_extraction": "90301ff751b540b3bd50949370dcefa952aee33a4b9b2613e03d5d450248f9e6",
        "character_mood": "31e5cd937cafcfbd9966fa73f680fb6ef6a0824a29f55f38b1bf3c2bcd5b40d2",
        "reply_check": "1800c77cd5b40bf356c19cc0951cef13509574f8481788fa9846342e280961fd",
        "vision_interpretation": "4729602b14eaf2cb844f9c569706955bf825a31dd90bbb83b737ec561dbaf952",
    }
    actual = {
        kind.value: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for kind, prompt in _SYSTEM_PROMPTS.items()
        if kind is not BackgroundCognitionKind.MEMORY_CONFLICT
    }
    assert actual == expected


def test_a_judgement_that_comes_after_another_turn_is_still_applied(tmp_path):
    async def scenario():
        gate = asyncio.Event()
        judge = Worker(judging("supersedes"), gate=gate)
        current = a_companion(
            tmp_path, {"memory": Worker(facts_said), "memory_conflict": judge}
        )
        await current.reply(JOB, conversation_id="a")
        await current.settle()
        await current.reply(NEW_JOB, conversation_id="a")
        while judge.calls == 0:
            await asyncio.sleep(0)
        await current.reply("對了", conversation_id="b")
        gate.set()
        await current.settle()
        await current.close()
        return current.memories("a"), current.memories("b")

    assert run(scenario()) == (["使用者換工作了"], [])


def one(relation):
    return {"conflicts": [{"old_id": "m1", "relation": relation, "reason": ""}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relation, confirm, kept",
    [
        # A change is said in the user's own words: the old fact is gone.
        ("supersedes", CHANGED, "supersedes"),
        # No change said, and both cannot be true: asked about instead.
        ("supersedes", {"changed": False, "word": "", "both_true": False}, "contradicts"),
        ("supersedes", {"changed": True, "word": "辭職", "both_true": False}, "contradicts"),
        ("supersedes", {"changed": "yes", "word": "換工作", "both_true": "no"}, "supersedes"),
        # Both can be true: nothing to do.
        ("supersedes", {"changed": False, "word": "", "both_true": True}, None),
        ("contradicts", {"changed": False, "word": "", "both_true": True}, None),
        ("contradicts", {"changed": False, "word": "", "both_true": False}, "contradicts"),
        # A change said does not make a contradiction a replacement.
        ("contradicts", CHANGED, "contradicts"),
        # No clear answer is no relation.
        ("supersedes", "not json", None),
        ("contradicts", {}, None),
    ],
)
async def test_a_relation_that_changes_her_memory_is_asked_about_once_more(relation, confirm, kept):
    judge, old, _, result = await judged(one(relation), confirm=confirm)
    assert result.status is TaskStatus.SUCCEEDED, result.error
    assert len(judge.calls) == 2
    relations = [item["relation"] for p in result.output.proposals for item in p.payload["conflicts"]]
    assert relations == ([kept] if kept else [])


@pytest.mark.asyncio
async def test_a_detail_added_is_not_asked_about_again():
    judge, _, _, result = await judged(one("refines"))
    assert len(judge.calls) == 1
    (proposal,) = result.output.proposals
    assert proposal.payload["conflicts"][0]["relation"] == "refines"


@pytest.mark.asyncio
async def test_without_the_users_words_nothing_is_replaced():
    old = fact("使用者在台積電上班")
    new = fact("使用者換工作了", minutes=1)
    judge = Judge(one("supersedes"), CHANGED)
    tasks, background = conflict_runtime(judge)
    async with tasks:
        result = await (await background.schedule_memory_conflict(new, [old])).wait()
    (proposal,) = result.output.proposals
    assert proposal.payload["conflicts"][0]["relation"] == "contradicts"


# --- what the user's words show, without a model --------------------------------

from ai_character_engine.memory.conflicts import (  # noqa: E402
    one_day_against_a_habit,
    says_a_change,
    says_a_plan,
)


@pytest.mark.parametrize(
    "words, change",
    [
        ("我上個月換工作了", True),
        ("我搬到台中了", True),
        ("我跟女朋友分手了", True),
        ("終於考完了", True),
        ("I changed jobs last month", True),
        ("I moved to Osaka", True),
        ("会社を辞めた", True),
        ("我25歲", False),
        ("我叫阿傑", False),
        ("I am 25", False),
        ("我下個月要搬去台中", False),
        ("I will move to Osaka next month", False),
    ],
)
def test_a_change_is_said_in_so_many_words_and_not_as_a_plan(words, change):
    assert says_a_change(words) is change


def test_a_plan_is_told_apart():
    assert says_a_plan("我下個月要搬去台中")
    assert says_a_plan("I'm going to quit next month")
    assert not says_a_plan("我上個月換工作了")


@pytest.mark.parametrize(
    "newer, earlier, apart",
    [
        ("使用者這週末沒去爬山 這週末沒去爬山", "使用者週末常去爬山", True),
        ("使用者今天沒跑步 今天沒去跑步", "使用者每天早上跑步", True),
        ("The user skipped the gym today", "The user goes to the gym every day", True),
        ("使用者今天換工作了", "使用者在台積電上班", False),
        ("使用者沒去爬山", "使用者週末常去爬山", False),
    ],
)
def test_one_day_does_not_go_against_a_habit(newer, earlier, apart):
    assert one_day_against_a_habit(newer, earlier) is apart


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["supersedes", "contradicts"])
async def test_the_words_of_the_user_decide_before_the_model_is_asked_again(relation):
    # 25 against 27: the model said time moved on, and so did the question
    # asked once more; no change is said, so it is asked about instead.
    old = fact("使用者27歲")
    new = said("使用者25歲", "我25歲", minutes=1)
    judge = Judge(one("supersedes"), {"changed": True, "word": "25歲", "both_true": False})
    tasks, background = conflict_runtime(judge)
    async with tasks:
        result = await (await background.schedule_memory_conflict(new, [old])).wait()
    assert [c["relation"] for c in result.output.proposals[0].payload["conflicts"]] == ["contradicts"]

    # A plan is no change yet, and no contradiction either.
    old = fact("使用者住在台北")
    new = said("使用者下個月要搬到台中", "我下個月要搬去台中", minutes=1)
    judge = Judge(one(relation), {"changed": True, "word": "搬去台中", "both_true": False})
    tasks, background = conflict_runtime(judge)
    async with tasks:
        result = await (await background.schedule_memory_conflict(new, [old])).wait()
    assert result.output.proposals == ()

    # One weekend against a habit: nothing, and no second question.
    old = fact("使用者週末常去爬山")
    new = said("使用者這週末沒去爬山", "這週末沒去爬山", minutes=1)
    judge = Judge(one(relation), {"changed": True, "word": "沒去", "both_true": False})
    tasks, background = conflict_runtime(judge)
    async with tasks:
        result = await (await background.schedule_memory_conflict(new, [old])).wait()
    assert result.output.proposals == ()
    assert len(judge.calls) == 1
