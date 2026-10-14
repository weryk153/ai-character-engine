"""Mechanism checks; model quality is tested separately with real conversations."""

import json

import pytest

from ai_character_engine.memory.experience import (
    experience_topics,
    normalize_experience,
)
from tests.fakes import system_context
from tests.test_companion import Foreground, Worker, companion


def lesson(messages):
    source = json.loads(messages[1].content)
    topics = source["topics"]
    return {
        "experience": {
            "topic": "nicknames",
            "topic_id": topics[0]["topic_id"] if topics else None,
            "lesson": "This person dislikes nicknames; I can use their name when we chat.",
            "outcome": "The user explicitly objected to a nickname.",
            "basis": "boundary",
            "character_quote": source["previous_reply"],
            "user_quote": source["feedback"],
            "revises": False,
        },
        "confidence": 0.95,
    }


@pytest.mark.asyncio
async def test_experience_survives_restart_and_new_conversation_but_not_another_user(
    tmp_path,
):
    worker = Worker(lesson)
    c = companion(tmp_path, {"interaction": worker}, character_development=True)
    await c.reply("Hi", conversation_id="old", user_id="alice")
    await c.settle()
    assert worker.calls == 0  # no preceding character reply
    await c.reply("Please don't use nicknames.", conversation_id="old", user_id="alice")
    await c.settle()
    records = c.interaction_experiences("alice")
    assert len(records) == 1
    assert records[0].metadata["user_quote"] == "Please don't use nicknames."
    assert c.interaction_experiences("bob") == ()
    await c.close()

    foreground = Foreground()
    c = companion(tmp_path, llm=foreground, character_development=True)
    await c.reply("Hello again", conversation_id="fresh", user_id="alice")
    assert "This person dislikes nicknames" in system_context(foreground.calls[-1])
    assert all(m.content != "Hi" for m in foreground.calls[-1])
    await c.reply("Hello", conversation_id="bob", user_id="bob")
    assert "This person dislikes nicknames" not in system_context(foreground.calls[-1])
    await c.reply("Hello", conversation_id="unknown")
    assert "This person dislikes nicknames" not in system_context(foreground.calls[-1])
    c.forget_interaction_experiences("alice", [records[0].id])
    await c.reply("Hi", conversation_id="another", user_id="alice")
    assert "This person dislikes nicknames" not in system_context(foreground.calls[-1])
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,user_id,skip",
    [(False, "alice", False), (True, None, False), (True, "alice", True)],
)
async def test_opt_in_identity_and_kept_dialogue_required(
    tmp_path, enabled, user_id, skip
):
    worker = Worker(lesson)
    c = companion(tmp_path, {"interaction": worker}, character_development=enabled)
    await c.reply("Hi", conversation_id="one", user_id=user_id)
    await c.reply(
        "Please don't use nicknames.",
        conversation_id="one",
        user_id=user_id,
        skip_memory=skip,
    )
    await c.settle()
    assert worker.calls == 0
    assert c.interaction_experiences("alice") == ()
    await c.close()


@pytest.mark.asyncio
async def test_repeated_support_and_revision_keep_original_episodes(tmp_path):
    def changing(messages):
        result = lesson(messages)
        if "changed my mind" in result["experience"]["user_quote"]:
            result["experience"].update(
                lesson="They now welcome this nickname when we are alone.",
                outcome="They explicitly revised the earlier boundary.",
                revises=True,
            )
        return result

    c = companion(
        tmp_path, {"interaction": Worker(changing)}, character_development=True
    )
    for text in [
        "Hello",
        "Please don't use nicknames.",
        "I still prefer my name.",
        "I changed my mind; that nickname is fine when we're alone.",
    ]:
        await c.reply(text, conversation_id="one", user_id="alice")
        await c.settle()
    records = c.interaction_experiences("alice")
    assert len(records) == 3
    before = experience_topics(records[:2])[0]
    assert before["distinct_exchanges"] == 2
    current = experience_topics(records)[0]
    assert current["distinct_exchanges"] == 1
    assert "dislikes" in current["previous_interpretation"]
    assert "now welcome" in current["interpretation"]
    await c.close()


def test_quotes_and_topic_ids_cannot_be_invented():
    source = {
        "previous_reply": "Hello",
        "feedback": "No nicknames please",
        "topics": [],
    }
    data = lesson([None, type("Msg", (), {"content": json.dumps(source)})()])
    assert normalize_experience(data, source)
    data["experience"]["user_quote"] = "I trust you completely"
    assert normalize_experience(data, source) is None
    data["experience"]["user_quote"] = source["feedback"]
    data["experience"]["topic_id"] = "someone-elses-topic"
    assert normalize_experience(data, source) is None


@pytest.mark.asyncio
async def test_delayed_learning_keeps_original_partner(tmp_path):
    import asyncio

    gate = asyncio.Event()
    worker = Worker(lesson, gate=gate)
    c = companion(tmp_path, {"interaction": worker}, character_development=True)
    await c.reply("Hi", conversation_id="alice-chat", user_id="alice")
    await c.reply("No nicknames", conversation_id="alice-chat", user_id="alice")
    await c.reply("Hello", conversation_id="bob-chat", user_id="bob")
    gate.set()
    await c.settle()
    assert len(c.interaction_experiences("alice")) == 1
    assert c.interaction_experiences("bob") == ()
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker", [Worker({"experience": None}), Worker({}, fail=True)]
)
async def test_no_observation_or_worker_failure_does_not_invent_growth(
    tmp_path, worker
):
    c = companion(tmp_path, {"interaction": worker}, character_development=True)
    await c.reply("Hi", conversation_id="one", user_id="alice")
    result = await c.reply("It is raining", conversation_id="one", user_id="alice")
    await c.settle()
    assert result.text
    assert c.interaction_experiences("alice") == ()
    await c.close()


@pytest.mark.asyncio
async def test_replayed_event_is_not_new_support_and_cannot_cross_partner_scope():
    from dataclasses import replace
    from uuid import uuid4
    from ai_character_engine.commit import CognitiveCommitCoordinator, CommitStatus
    from ai_character_engine.memory.experience import (
        EXPERIENCE_TARGET,
        experience_scope,
    )
    from ai_character_engine.tasks import MultiTaskRuntime
    from tests.test_commit_coordinator import proposal, runtime

    engine = runtime()
    source = {
        "user_id": "alice",
        "previous_reply": "Hello",
        "feedback": "No nicknames",
        "topics": [],
    }
    data = lesson([None, type("Msg", (), {"content": json.dumps(source)})()])
    p = proposal(
        EXPERIENCE_TARGET, data["experience"], worker_kind="interaction_learning"
    )
    p = replace(
        p,
        provenance={
            **p.provenance,
            "user_id": "alice",
            "interaction": source,
            "foreground_event_content": source["feedback"],
        },
    )
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    assert (await commits.commit(p)).status is CommitStatus.COMMITTED
    # A new coordinator cannot rely on its old in-memory deduplication cache.
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(engine))
    replay = replace(p, id=uuid4().hex)
    assert (await commits.commit(replay)).status is CommitStatus.DUPLICATE
    records = engine.memory_manager.store.list_for_character(
        experience_scope("c1", "alice")
    )
    assert len(records) == 1
    topic = records[0].metadata["topic_id"]
    forged = replace(
        p,
        id=uuid4().hex,
        payload={**p.payload, "topic_id": topic},
        provenance={
            **p.provenance,
            "user_id": "bob",
            "foreground_event_id": "evt-2",
            "interaction": {
                **source,
                "user_id": "bob",
                "topics": [{"topic_id": topic}],
            },
        },
    )
    outcome = await commits.commit(forged)
    assert outcome.status is CommitStatus.REJECTED
    assert outcome.reason == "experience_topic_not_owned_by_user"


@pytest.mark.asyncio
async def test_next_feedback_does_not_cancel_unread_previous_feedback(tmp_path):
    import asyncio

    gate = asyncio.Event()
    worker = Worker(lesson, gate=gate)
    c = companion(tmp_path, {"interaction": worker}, character_development=True)
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply("No nicknames", conversation_id="one", user_id="alice")
    await c.reply("I like that you listen", conversation_id="one", user_id="alice")
    gate.set()
    await c.settle()
    assert worker.abandoned == 0
    assert {r.metadata["user_quote"] for r in c.interaction_experiences("alice")} == {
        "No nicknames",
        "I like that you listen",
    }
    await c.close()


@pytest.mark.asyncio
async def test_export_import_carries_only_the_matching_partners_experience(tmp_path):
    c = companion(tmp_path / "first", {"interaction": Worker(lesson)}, character_development=True)
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply("No nicknames", conversation_id="one", user_id="alice")
    await c.settle()
    data = await c.export_state()
    await c.close()
    restored = companion(tmp_path / "restored", character_development=True)
    restored.import_state(data)
    assert len(restored.interaction_experiences("alice")) == 1
    assert restored.interaction_experiences("bob") == ()
    await restored.close()
