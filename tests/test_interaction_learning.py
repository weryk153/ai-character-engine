"""Mechanism checks; model quality is tested separately with real conversations."""

import json

import pytest

from ai_character_engine.memory.experience import (
    experience_topics,
    normalize_experience,
    experience_review_proof,
)
from tests.fakes import system_context
from tests.test_companion import Foreground, Worker, companion


def approving_review(source):
    return {
        "literal_user_meaning": "The user is stating an interaction preference.",
        "explicit_feedback": True,
        "correct_speakers": True,
        "supported_lesson": True,
        "supported_outcome": True,
        "correct_topic_update": True,
        "preserves_scope": True,
        "supporting_user_quote": source["user_feedback"],
        "reason": "Supported.",
    }


def lesson(messages):
    source = json.loads(messages[1].content)
    if "candidate" in source:
        return approving_review(source)
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
        if "experience" not in result:
            return result
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
            "interpretation_review": experience_review_proof(
                data["experience"],
                approving_review({"user_feedback": source["feedback"]}),
                source["feedback"],
            ),
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
    forged = replace(
        forged,
        provenance={
            **forged.provenance,
            "interpretation_review": experience_review_proof(
                forged.payload,
                approving_review({"user_feedback": source["feedback"]}),
                source["feedback"],
            ),
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_check",
    [
        "explicit_feedback",
        "correct_speakers",
        "supported_outcome",
        "correct_topic_update",
        "preserves_scope",
        "supporting_user_quote",
        "unavailable",
    ],
)
async def test_unconfirmed_interpretation_never_becomes_memory(tmp_path, failed_check):
    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "candidate" not in source:
            return lesson(messages)
        if failed_check == "unavailable":
            raise RuntimeError("review endpoint unavailable")
        answer = approving_review(source)
        answer[failed_check] = (
            "Not said by this user"
            if failed_check == "supporting_user_quote"
            else False
        )
        return answer

    c = companion(
        tmp_path, {"interaction": Worker(reviewing)}, character_development=True
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply("No nicknames", conversation_id="one", user_id="alice")
    await c.settle()
    assert c.interaction_experiences("alice") == ()
    await c.close()


@pytest.mark.asyncio
async def test_old_unreviewed_or_changed_interpretations_do_not_enter_context(tmp_path):
    from dataclasses import replace

    c = companion(tmp_path, {"interaction": Worker(lesson)}, character_development=True)
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply("No nicknames", conversation_id="one", user_id="alice")
    await c.settle()
    (record,) = c.interaction_experiences("alice")
    assert experience_topics([record])
    assert experience_topics([replace(record, summary="They allow all names")]) == []
    assert (
        experience_topics(
            [
                replace(
                    record,
                    metadata={
                        key: value
                        for key, value in record.metadata.items()
                        if key != "interpretation_review"
                    },
                )
            ]
        )
        == []
    )
    assert (
        experience_topics(
            [
                replace(
                    record,
                    metadata={
                        **record.metadata,
                        "review_feedback": "Actually, any nickname is fine",
                    },
                )
            ]
        )
        == []
    )
    await c.close()


def test_review_cannot_be_reused_for_different_candidate_or_feedback():
    from ai_character_engine.memory.experience import verified_experience

    source = {"previous_reply": "Hi", "feedback": "No nicknames", "topics": []}
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]
    review = approving_review({"user_feedback": source["feedback"]})
    proof = experience_review_proof(candidate, review, source["feedback"])
    assert verified_experience(candidate, proof, source["feedback"])
    assert not verified_experience(
        {**candidate, "lesson": "Any nickname is welcome"}, proof, source["feedback"]
    )
    assert not verified_experience(
        candidate, proof, "No nicknames is what I used to say; I changed my mind"
    )


@pytest.mark.asyncio
async def test_coordinator_refuses_unreviewed_and_changed_candidates():
    from dataclasses import replace
    from ai_character_engine.commit import CognitiveCommitCoordinator, CommitStatus
    from ai_character_engine.memory.experience import EXPERIENCE_TARGET
    from ai_character_engine.tasks import MultiTaskRuntime
    from tests.test_commit_coordinator import proposal, runtime

    source = {
        "user_id": "alice",
        "previous_reply": "Hello",
        "feedback": "No nicknames",
        "topics": [],
    }
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]
    p = proposal(EXPERIENCE_TARGET, candidate, worker_kind="interaction_learning")
    p = replace(
        p,
        provenance={
            **p.provenance,
            "user_id": "alice",
            "interaction": source,
            "foreground_event_content": source["feedback"],
        },
    )
    commits = CognitiveCommitCoordinator(MultiTaskRuntime(runtime()))
    result = await commits.commit(p)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "experience_requires_supported_interpretation"
    proof = experience_review_proof(
        candidate,
        approving_review({"user_feedback": source["feedback"]}),
        source["feedback"],
    )
    changed = replace(
        p,
        id="changed-after-review",
        payload={**candidate, "lesson": "All nicknames are fine"},
        provenance={**p.provenance, "interpretation_review": proof},
    )
    result = await commits.commit(changed)
    assert result.status is CommitStatus.REJECTED
    assert result.reason == "experience_requires_supported_interpretation"


@pytest.mark.asyncio
async def test_supported_feedback_survives_without_the_unsupported_lesson(tmp_path):
    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "candidate" in source:
            return {**approving_review(source), "supported_lesson": False}
        result = lesson(messages)
        result["experience"]["lesson"] = "They are deliberately trying to annoy me."
        result["experience"]["outcome"] = (
            "The user explicitly asks me not to give them nicknames."
        )
        return result

    llm = Foreground()
    c = companion(
        tmp_path,
        {"interaction": Worker(reviewing)},
        llm=llm,
        character_development=True,
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply("No nicknames", conversation_id="one", user_id="alice")
    await c.settle()
    (record,) = c.interaction_experiences("alice")
    assert record.summary == "The user explicitly asks me not to give them nicknames."
    assert record.metadata["interpretation_review"]["accepted_components"] == [
        "outcome"
    ]
    assert (
        record.metadata["lesson"] == "They are deliberately trying to annoy me."
    )  # audit only
    await c.reply("Hello again", conversation_id="fresh", user_id="alice")
    context = system_context(llm.calls[-1])
    assert "explicitly asks me not to give them nicknames" in context
    assert "deliberately trying to annoy" not in context
    assert "reviewed_observation_only" in context
    await c.close()
