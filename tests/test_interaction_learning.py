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


def approving_review(source, conditions=()):
    return {
        "literal_user_meaning": "The user is stating an interaction preference.",
        "user_conditions": list(conditions),
        "explicit_feedback": True,
        "topic_problems": [],
        "outcome_problems": [],
        "lesson_problems": [],
        "supporting_user_quote": source["user_feedback"],
    }


def condition_check(
    source, in_outcome="kept", in_lesson="kept", conflicting_outcome="", conflicting_lesson=""
):
    return {
        "conditions": [
            {
                "condition": condition,
                "in_outcome": in_outcome,
                "in_lesson": in_lesson,
                "conflicting_outcome_text": conflicting_outcome,
                "conflicting_lesson_text": conflicting_lesson,
                "reason": "Compared.",
            }
            for condition in source["user_conditions"]
        ]
    }


def lesson(messages):
    source = json.loads(messages[1].content)
    if "existing" in source:
        return {"action": "UPVOTE", "user_excerpt": source["feedback"], "reason": "Same again."}
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
    # Repeated support is a vote on the existing record, not a new episode.
    assert len(records) == 2
    before = experience_topics(records[:1])[0]
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
        "outcome_problems",
        "topic_problems",
        "supporting_user_quote",
        "literal_user_meaning",
        "user_conditions",
        "missing_outcome_problems",
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
        if failed_check == "missing_outcome_problems":
            del answer["outcome_problems"]
            return answer
        answer[failed_check] = {
            "explicit_feedback": False,
            "supporting_user_quote": "Not said by this user",
            "literal_user_meaning": "",
            "user_conditions": "only when we are alone",  # not a list
        }.get(failed_check, ["Adds a feeling the user never stated."])
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
    assert proof["version"] == 3
    assert verified_experience(candidate, proof, source["feedback"])
    assert not verified_experience(
        {**candidate, "lesson": "Any nickname is welcome"}, proof, source["feedback"]
    )
    assert not verified_experience(
        candidate, proof, "No nicknames is what I used to say; I changed my mind"
    )


def test_quote_may_join_exact_excerpts_in_order_with_an_ellipsis():
    feedback = "我改主意了：私下叫我菜攤老闆可以，這個是我們一起玩的。其他綽號還是先不要。"
    source = {"previous_reply": "Hi", "feedback": feedback, "topics": []}
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]

    def components(quote):
        review = {
            **approving_review({"user_feedback": feedback}),
            "supporting_user_quote": quote,
        }
        proof = experience_review_proof(candidate, review, feedback)
        return proof["accepted_components"] if proof else []

    assert components("私下叫我菜攤老闆可以...其他綽號還是先不要。") == [
        "outcome",
        "lesson",
    ]
    assert components("私下叫我菜攤老闆可以…其他綽號還是先不要") == ["outcome", "lesson"]
    # Reordered, altered or empty excerpts are not quotations of the feedback.
    assert components("其他綽號還是先不要...私下叫我菜攤老闆可以") == []
    assert components("私下叫我菜攤老闆可以...其他綽號都不要") == []
    assert components("...") == []
    assert components("") == []


def test_speaker_or_support_error_in_the_lesson_keeps_the_reviewed_outcome():
    feedback = "這樣陪我聊就很好。偶爾問也可以，只是不用每句都問。"
    source = {"previous_reply": "Hi", "feedback": feedback, "topics": []}
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]
    for problem in (
        "The lesson states the user's preference as the character's own.",
        "The lesson adds that the user felt ignored.",
    ):
        review = approving_review({"user_feedback": feedback})
        review["lesson_problems"] = [problem]
        proof = experience_review_proof(candidate, review, feedback)
        assert proof["accepted_components"] == ["outcome"]
        review["outcome_problems"] = [problem]
        assert experience_review_proof(candidate, review, feedback) is None


def test_changed_condition_drops_only_the_component_that_changed_it():
    from ai_character_engine.memory.experience import (
        condition_check_required,
        verified_experience,
    )

    feedback = "我喜歡你也自己下場玩。規則偏心一點也行，但被我抓到要認輸喔。"
    source = {"previous_reply": "Hi", "feedback": feedback, "topics": []}
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]
    review = approving_review({"user_feedback": feedback}, ["被我抓到要認輸"])
    assert condition_check_required(review, feedback)
    # A stated condition without the comparison is not a finished review.
    assert experience_review_proof(candidate, review, feedback) is None

    def proof_with(in_outcome, in_lesson, cite_outcome="", cite_lesson=""):
        return experience_review_proof(
            candidate,
            review,
            feedback,
            condition_check(
                review,
                in_outcome=in_outcome,
                in_lesson=in_lesson,
                conflicting_outcome=cite_outcome,
                conflicting_lesson=cite_lesson,
            ),
        )

    assert proof_with("kept", "kept")["accepted_components"] == ["outcome", "lesson"]
    cite = "I can use their name when we chat"
    assert proof_with("absent", "changed", cite_lesson=cite)["accepted_components"] == [
        "outcome"
    ]
    # A changed verdict has to point at the words that change the condition: an
    # exact excerpt of that component, or excerpts joined in order by an ellipsis.
    assert proof_with("absent", "changed") is None
    assert proof_with("absent", "changed", cite_lesson="when we play") is None
    assert (
        proof_with("absent", "changed", cite_lesson=cite.replace("their", "the"))
        is None
    )
    assert proof_with(
        "absent", "changed", cite_lesson="dislikes nicknames...when we chat"
    )["accepted_components"] == ["outcome"]
    assert proof_with("absent", "changed", cite_outcome=cite) is None
    assert proof_with("changed", "kept", cite_outcome="objected to a nickname") is None
    assert proof_with("absent", "unsure") is None
    full = proof_with("absent", "kept")
    assert verified_experience(candidate, full, feedback)
    # The comparison must cover exactly the conditions the review listed.
    tampered = {
        **full,
        "conditions": {
            "conditions": [
                {**full["conditions"]["conditions"][0], "condition": "規則偏心也行"}
            ]
        },
    }
    assert not verified_experience(candidate, tampered, feedback)
    assert not verified_experience(
        candidate, {**full, "conditions": {"conditions": []}}, feedback
    )
    assert not condition_check_required(
        {**review, "explicit_feedback": False}, feedback
    )
    assert not condition_check_required({**review, "user_conditions": []}, feedback)


def test_version_2_reviews_already_stored_still_verify():
    import hashlib

    from ai_character_engine.memory.experience import verified_experience

    source = {"previous_reply": "Hi", "feedback": "No nicknames", "topics": []}
    candidate = lesson([None, type("M", (), {"content": json.dumps(source)})()])[
        "experience"
    ]
    review = {
        "literal_user_meaning": "The user is stating an interaction preference.",
        "explicit_feedback": True,
        "correct_speakers": True,
        "supported_lesson": False,
        "supported_outcome": True,
        "correct_topic_update": True,
        "preserves_scope": True,
        "supporting_user_quote": "No nicknames",
        "reason": "Supported.",
    }
    proof = {
        "version": 2,
        "accepted_components": ["outcome"],
        "candidate_sha256": hashlib.sha256(
            json.dumps(dict(candidate), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "feedback_sha256": hashlib.sha256(b"No nicknames").hexdigest(),
        "review": review,
    }
    assert verified_experience(candidate, proof, "No nicknames")
    assert not verified_experience(
        candidate, {**proof, "accepted_components": ["outcome", "lesson"]}, "No nicknames"
    )
    assert not verified_experience(
        candidate,
        {**proof, "review": {**review, "correct_speakers": False}},
        "No nicknames",
    )


@pytest.mark.asyncio
async def test_condition_comparison_runs_as_its_own_call_and_guards_the_lesson(
    tmp_path,
):
    calls = []

    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "user_conditions" in source and "candidate" in source:
            calls.append("conditions")
            assert "character_previous_reply" not in source
            return condition_check(
                source,
                in_outcome="absent",
                in_lesson="changed",
                conflicting_lesson="as long as I admit a loss",
            )
        if "candidate" in source:
            calls.append("review")
            return approving_review(source, ["only if I catch them"])
        calls.append("extract")
        result = lesson(messages)
        result["experience"]["lesson"] = "Biased rules are fine as long as I admit a loss."
        result["experience"]["outcome"] = "The user wants me to join the game."
        return result

    llm = Foreground()
    c = companion(
        tmp_path,
        {"interaction": Worker(reviewing)},
        llm=llm,
        character_development=True,
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply(
        "Join the game; biased rules are fine, but admit a loss if I catch you.",
        conversation_id="one",
        user_id="alice",
    )
    await c.settle()
    assert calls == ["extract", "review", "conditions"]
    (record,) = c.interaction_experiences("alice")
    assert record.summary == "The user wants me to join the game."
    proof = record.metadata["interpretation_review"]
    assert proof["accepted_components"] == ["outcome"]
    assert proof["conditions"]["conditions"][0]["condition"] == "only if I catch them"
    await c.reply("Hello again", conversation_id="fresh", user_id="alice")
    context = system_context(llm.calls[-1])
    assert "wants me to join the game" in context
    assert "as long as I admit a loss" not in context
    await c.close()


@pytest.mark.asyncio
async def test_unavailable_condition_comparison_stores_nothing(tmp_path):
    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "user_conditions" in source and "candidate" in source:
            raise RuntimeError("comparison endpoint unavailable")
        if "candidate" in source:
            return approving_review(source, ["only when we are alone"])
        return lesson(messages)

    c = companion(
        tmp_path, {"interaction": Worker(reviewing)}, character_development=True
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply(
        "Nicknames only when we are alone", conversation_id="one", user_id="alice"
    )
    await c.settle()
    assert c.interaction_experiences("alice") == ()
    await c.close()


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
            answer = approving_review(source)
            answer["lesson_problems"] = ["Invents a motive the user never stated."]
            return answer
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


@pytest.mark.asyncio
async def test_users_own_conditions_ride_along_with_the_stored_summary(tmp_path):
    """A summary that leaves a condition out does not lose it: the condition is
    shown beside the interpretation, in the user's words only."""

    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "user_conditions" in source and "candidate" in source:
            return condition_check(source, in_outcome="absent", in_lesson="absent")
        if "candidate" in source:
            return approving_review(source, ["if I catch you", "only if I catch them"])
        result = lesson(messages)
        result["experience"]["lesson"] = "I should join the game instead of only hosting."
        result["experience"]["outcome"] = "The user wants me to join the game."
        return result

    llm = Foreground()
    c = companion(
        tmp_path,
        {"interaction": Worker(reviewing)},
        llm=llm,
        character_development=True,
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply(
        "Join the game; biased rules are fine, but admit a loss if I catch you.",
        conversation_id="one",
        user_id="alice",
    )
    await c.settle()
    (record,) = c.interaction_experiences("alice")
    (topic,) = experience_topics([record])
    # The paraphrased one is the reviewer's wording, not the user's.
    assert topic["conditions"] == ["if I catch you"]
    await c.reply("Hello again", conversation_id="fresh", user_id="alice")
    context = system_context(llm.calls[-1])
    assert '"conditions": ["if I catch you"]' in context
    assert "own words limiting that interpretation" in context
    await c.close()


def checklist_answers(**yes):
    """All six answers no, except the given keys: value = (text_excerpt, user_excerpt)."""
    from ai_character_engine.memory.experience import (
        CHECKLIST_COMPONENTS,
        CHECKLIST_QUESTIONS,
        checklist_key,
    )

    answers = {}
    for component in CHECKLIST_COMPONENTS:
        for question, _, _ in CHECKLIST_QUESTIONS:
            key = checklist_key(component, question)
            if key in yes:
                text_excerpt, user_excerpt = yes[key]
                answers[key] = {
                    "answer": "yes",
                    "text_excerpt": text_excerpt,
                    "user_excerpt": user_excerpt,
                    "reason": "Found.",
                }
            else:
                answers[key] = {
                    "answer": "no",
                    "text_excerpt": "",
                    "user_excerpt": "",
                    "reason": "Not found.",
                }
    return answers


def test_checklist_yes_counts_only_with_the_words_of_the_text():
    from ai_character_engine.memory.experience import verified_experience

    feedback = "You can call me Shiro, but only at home."
    candidate = {
        "topic": "nicknames",
        "topic_id": None,
        "lesson": "I don't like being called nicknames anywhere.",
        "outcome": "The user allows the nickname Shiro at home.",
        "character_quote": "Can I call you Shiro?",
        "user_quote": feedback,
        "basis": "boundary",
        "revises": False,
    }
    source = {"user_feedback": feedback}
    review = {**approving_review(source), "lesson_problems": ["invented problem"]}

    def accepted(answers):
        proof = experience_review_proof(candidate, review, feedback, None, answers)
        if proof is not None:
            assert proof["version"] == 4
            assert proof["checklist"] == answers
            assert verified_experience(candidate, proof, feedback)
        return proof["accepted_components"] if proof else []

    # The review's own problem list is not consulted; six noes accept both.
    assert accepted(checklist_answers()) == ["outcome", "lesson"]
    # A yes that copies the words of the lesson drops the lesson only.
    assert accepted(
        checklist_answers(**{"lesson:speaker": ("I don't like being called", "")})
    ) == ["outcome"]
    # A yes with paraphrased words is a problem nobody can point at: not found.
    assert accepted(
        checklist_answers(**{"lesson:speaker": ("the character dislikes nicknames", "")})
    ) == ["outcome", "lesson"]
    # Widening must also cite the user's narrower words.
    assert accepted(checklist_answers(**{"outcome:widened": ("at home", "")})) == [
        "outcome",
        "lesson",
    ]
    assert accepted(
        checklist_answers(**{"outcome:widened": ("at home", "only at home")})
    ) == []
    # A flagged outcome takes the lesson with it; an unusable answer stores nothing.
    assert accepted(
        checklist_answers(**{"outcome:unstated": ("allows the nickname", "")})
    ) == []
    incomplete = checklist_answers()
    del incomplete["lesson:widened"]
    assert accepted(incomplete) == []
    vague = checklist_answers()
    vague["outcome:speaker"]["answer"] = "maybe"
    assert accepted(vague) == []
    # Tampering with the stored answers breaks the proof.
    proof = experience_review_proof(candidate, review, feedback, None, checklist_answers())
    proof["checklist"]["lesson:speaker"] = {
        "answer": "yes",
        "text_excerpt": "I don't like",
        "user_excerpt": "",
        "reason": "",
    }
    assert not verified_experience(candidate, proof, feedback)


@pytest.mark.asyncio
async def test_checklist_review_asks_six_questions_and_its_proof_survives_restart(
    tmp_path,
):
    calls = []

    def reviewing(messages):
        source = json.loads(messages[1].content)
        if "question" in source:
            calls.append(f"{source['which']}:{source['question'][:4]}")
            assert source["user_feedback"] == "I don't like you asking a question in every sentence."
            if source["which"] == "lesson" and source["question"].startswith("Does the text put"):
                return {
                    "answer": "yes",
                    "text_excerpt": "I don't like being asked",
                    "user_excerpt": "",
                    "reason": "The user's dislike is written as the character's.",
                }
            return {"answer": "no", "text_excerpt": "", "user_excerpt": "", "reason": ""}
        if "candidate" in source:
            calls.append("review")
            return approving_review(source)
        calls.append("extract")
        result = lesson(messages)
        result["experience"]["lesson"] = "I don't like being asked questions in every sentence."
        result["experience"]["outcome"] = "The user dislikes a question in every sentence."
        return result

    llm = Foreground()
    c = companion(
        tmp_path,
        {"interaction": Worker(reviewing)},
        llm=llm,
        character_development=True,
        interaction_review_checklist=True,
    )
    await c.reply("Hello", conversation_id="one", user_id="alice")
    await c.reply(
        "I don't like you asking a question in every sentence.",
        conversation_id="one",
        user_id="alice",
    )
    await c.settle()
    assert calls == ["extract", "review"] + [
        f"{component}:Does" for component in ("outcome", "lesson") for _ in range(3)
    ]
    (record,) = c.interaction_experiences("alice")
    proof = record.metadata["interpretation_review"]
    assert proof["version"] == 4
    assert proof["accepted_components"] == ["outcome"]
    assert record.summary == "The user dislikes a question in every sentence."
    await c.close()

    foreground = Foreground()
    c = companion(
        tmp_path, llm=foreground, character_development=True, interaction_review_checklist=True
    )
    await c.reply("Hello again", conversation_id="fresh", user_id="alice")
    context = system_context(foreground.calls[-1])
    assert "dislikes a question in every sentence" in context
    assert "I don't like being asked" not in context
    await c.close()


def test_votes_move_support_only_with_the_users_words():
    from ai_character_engine.memory.experience import apply_vote

    feedback = "Yes, still no nicknames please."
    assert apply_vote(2, {"action": "UPVOTE", "user_excerpt": "still no nicknames"}, feedback) == 3
    assert apply_vote(2, {"action": "DOWNVOTE", "user_excerpt": "nicknames are fine now"}, feedback) is None
    assert apply_vote(2, {"action": "NOOP", "user_excerpt": ""}, feedback) == 2
    assert apply_vote(1, {"action": "DOWNVOTE", "user_excerpt": "still no nicknames"}, feedback) == 0
    assert apply_vote(2, {"action": "EDIT", "user_excerpt": "still no nicknames"}, feedback) is None


def voting(action):
    """A worker that extracts, approves the first review and answers votes with `action`."""
    calls = []

    def worker(messages):
        source = json.loads(messages[1].content)
        if "existing" in source:
            calls.append("vote")
            assert source["existing"]["topic_id"]
            return {"action": action, "user_excerpt": source["feedback"], "reason": "Read."}
        if "candidate" in source:
            calls.append("review")
            return approving_review(source)
        calls.append("extract")
        return lesson(messages)

    return worker, calls


@pytest.mark.asyncio
async def test_an_upvote_counts_the_exchange_without_a_new_review(tmp_path):
    worker, calls = voting("UPVOTE")
    c = companion(tmp_path, {"interaction": Worker(worker)}, character_development=True)
    for text in ["Hello", "Please don't use nicknames.", "I still prefer my name."]:
        await c.reply(text, conversation_id="one", user_id="alice")
        await c.settle()
    assert calls == ["extract", "review", "extract", "vote"]
    (record,) = c.interaction_experiences("alice")
    assert record.metadata["support"] == 3
    (topic,) = experience_topics([record])
    assert topic["distinct_exchanges"] == 2
    assert topic["interpretation_basis"] == "reviewed_lesson"
    await c.close()


@pytest.mark.asyncio
async def test_a_downvoted_lesson_leaves_the_prompt_but_keeps_its_outcome(tmp_path):
    from ai_character_engine.memory.experience import experience_scope

    worker, calls = voting("DOWNVOTE")
    c = companion(tmp_path, {"interaction": Worker(worker)}, character_development=True)
    for text in ["Hello", "Please don't use nicknames.", "Hm, that is not quite it."]:
        await c.reply(text, conversation_id="one", user_id="alice")
        await c.settle()
    (record,) = c.interaction_experiences("alice")
    assert record.metadata["support"] == 1
    (topic,) = experience_topics([record])
    assert topic["interpretation_basis"] == "reviewed_observation_only"
    assert topic["interpretation"] == "The user explicitly objected to a nickname."
    await c.reply("No, really, not that.", conversation_id="one", user_id="alice")
    await c.settle()
    assert experience_topics(c.interaction_experiences("alice")) == []
    stored = c.runtime.memory_manager.store.list_for_character(experience_scope("mei", "alice"))
    assert [r.status for r in stored] == ["forgotten"]
    assert stored[0].forgotten_at is not None
    assert [o["user_excerpt"] for o in stored[0].metadata["objections"]] == [
        "Hm, that is not quite it.",
        "No, really, not that.",
    ]
    await c.close()


@pytest.mark.asyncio
async def test_a_vote_in_paraphrased_words_changes_nothing(tmp_path):
    def worker(messages):
        source = json.loads(messages[1].content)
        if "existing" in source:
            return {"action": "DOWNVOTE", "user_excerpt": "the user objects", "reason": ""}
        return lesson(messages)

    c = companion(tmp_path, {"interaction": Worker(worker)}, character_development=True)
    for text in ["Hello", "Please don't use nicknames.", "Hm, that is not quite it."]:
        await c.reply(text, conversation_id="one", user_id="alice")
        await c.settle()
    (record,) = c.interaction_experiences("alice")
    assert record.metadata["support"] == 2
    assert record.metadata["objections"] == []
    await c.close()
