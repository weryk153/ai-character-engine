"""Relationship views: derived from reviewed experiences, held only while they are."""

import json
from dataclasses import replace

import pytest

from ai_character_engine.memory.relationship import (
    RELATIONSHIP_VIEW_KIND,
    current_topics,
    held_views,
    normalize_views,
    pending_importance,
    reflection_source,
    views_to_review,
)
from ai_character_engine.memory.models import MemoryRecord
from tests.test_companion import Worker, companion
from tests.test_interaction_learning import _held, keyed_lesson


def view(text, *records, minutes=100):
    from datetime import UTC, datetime, timedelta

    return MemoryRecord(
        character_id="mei#with:x", summary=text, kind=RELATIONSHIP_VIEW_KIND,
        importance=0.6, source_event_id="r",
        created_at=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=minutes),
        metadata={
            "derived_from": [
                {"record_id": r.id, "topic_id": r.metadata["topic_id"]} for r in records
            ],
            "reflected_at": (datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=minutes)).isoformat(),
        },
    )


def test_a_view_holds_while_every_source_is_its_topics_current_interpretation():
    a = _held("nick", "No nicknames.")
    b = _held("tea", "Green tea, please.", minutes=1)
    v = view("They like it plain and simple.", a, b)
    assert held_views([v], [a, b]) == [v]


def test_forgetting_voting_out_or_revising_a_source_withdraws_the_view():
    a = _held("nick", "No nicknames.")
    v = view("Plain is better with them.", a)
    forgotten = replace(a, status="forgotten")
    assert held_views([v], [forgotten]) == []
    voted_out = replace(a, metadata={**a.metadata, "support": 0, "retired_by_votes": True})
    assert held_views([v], [voted_out]) == []
    stored = replace(
        a, status="forgotten",
        metadata={**a.metadata, "support": 0, "retired_by_votes": True},
    )
    assert held_views([v], [stored]) == []
    newer = _held("nick", "Nicknames are fine at home.", minutes=5, topic_id=a.metadata["topic_id"])
    assert held_views([v], [a, newer]) == []
    assert views_to_review([v], [a, newer]) == [v]


def test_a_withdrawn_view_does_not_come_back_with_a_later_experience():
    a = _held("nick", "No nicknames.")
    v = view("Plain is better with them.", a)
    later = _held("nick", "No nicknames, really.", minutes=50, topic_id=a.metadata["topic_id"])
    assert held_views([v], [replace(a, status="forgotten"), later]) == []


def test_only_ids_that_were_offered_count_and_duplicates_collapse():
    a = _held("nick", "No nicknames.")
    source = reflection_source([a], [], limit=3)
    short = source["experiences"][0]["id"]
    data = {"views": [
        {"view": "They want their name.", "derived_from": [short, short]},
        {"view": "Made up.", "derived_from": ["zz"]},
        {"view": "", "derived_from": [short]},
        {"view": "No source.", "derived_from": []},
    ]}
    assert normalize_views(data, source, limit=3) == [
        {"view": "They want their name.",
         "derived_from": [{"record_id": a.id, "topic_id": a.metadata["topic_id"]}]}
    ]


def test_pending_importance_counts_what_came_after_the_last_reflection():
    a = _held("nick", "No nicknames.", minutes=0)
    b = _held("tea", "Green tea.", minutes=200)
    a = replace(a, metadata={**a.metadata, "importance_events": [
        {"event_id": "1", "importance": 0.9, "observed_at": a.metadata["observed_at"]}]})
    b = replace(b, metadata={**b.metadata, "importance_events": [
        {"event_id": "2", "importance": 0.4, "observed_at": b.metadata["observed_at"]}]})
    v = view("Earlier view.", a, minutes=100)
    assert pending_importance([a, b], [v]) == 0.4
    assert pending_importance([a, b], []) == 1.3


def test_records_from_before_importance_events_count_at_their_old_importance():
    a = _held("nick", "No nicknames.")
    assert "importance_events" not in a.metadata
    assert pending_importance([a], []) == 0.7


def test_nothing_to_reflect_on_without_held_experiences():
    assert reflection_source([], [], limit=3) is None


def test_an_unverified_later_record_does_not_become_the_current_interpretation():
    a = _held("nick", "No nicknames.")
    b = _held("nick", "Nicknames are fine.", minutes=5, topic_id=a.metadata["topic_id"])
    b = replace(b, summary="Something the review never accepted.")
    assert current_topics([a, b])[a.metadata["topic_id"]].id == a.id
    source = reflection_source([a, b], [], limit=3)
    assert source["ids"][source["experiences"][0]["id"]]["record_id"] == a.id


def test_a_counted_vote_after_the_last_reflection_adds_its_importance():
    a = _held("nick", "No nicknames.", minutes=0)
    a = replace(a, metadata={**a.metadata, "importance_events": [
        {"event_id": "1", "importance": 0.9, "observed_at": a.metadata["observed_at"]},
        {"event_id": "2", "importance": 0.3, "observed_at": "2026-10-01T05:00:00+00:00"},
    ]})
    v = view("Earlier view.", a, minutes=100)
    assert pending_importance([a], [v]) == 0.3


def reflect(messages):
    data = json.loads(messages[1].content)
    if "experiences" in data and not data["experiences"]:
        return {"views": []}
    if "experiences" in data:
        return {"views": [{"view": "With them I use their name.",
                           "derived_from": [data["experiences"][0]["id"]]}]}
    return keyed_lesson(messages)


async def taught(tmp_path, **settings):
    worker = Worker(reflect)
    c = companion(tmp_path, {"interaction": worker, "relationship": worker},
                  character_development=True, **settings)
    await c.reply("Hi", conversation_id="a", user_id="alice")
    await c.settle()
    await c.reply("Please don't use nicknames.", conversation_id="a", user_id="alice")
    await c.settle()
    await c.reply("Thanks.", conversation_id="a", user_id="alice")
    await c.settle()
    return c


@pytest.mark.asyncio
async def test_enough_importance_brings_a_view_for_that_person_only(tmp_path):
    c = await taught(tmp_path, relationship_reflection_threshold=0.8)
    (one,) = c.relationship_views("alice")
    assert one.summary == "With them I use their name."
    assert c.relationship_views("bob") == ()
    assert all(r.kind != "relationship_view"
               for r in c._memory_store.list_for_character(c._self_scope()))
    await c.close()


@pytest.mark.asyncio
async def test_below_the_threshold_nothing_is_reflected(tmp_path):
    c = await taught(tmp_path, relationship_reflection_threshold=5.0)
    assert c.relationship_views("alice") == ()
    await c.close()


def stored_views(c, user_id="alice"):
    scope = c.interaction_experiences(user_id)[0].character_id
    return [r for r in c._memory_store.list_for_character(scope)
            if r.kind == "relationship_view"]


@pytest.mark.asyncio
async def test_a_forgotten_source_withdraws_the_view_and_forces_a_reflection(tmp_path):
    worker = Worker(reflect)
    c = companion(tmp_path, {"interaction": worker, "relationship": worker},
                  character_development=True, relationship_reflection_threshold=0.8)
    for text in ("Hi", "Please don't use nicknames.", "Thanks."):
        await c.reply(text, conversation_id="a", user_id="alice")
        await c.settle()
    (view,) = c.relationship_views("alice")
    (record,) = c.interaction_experiences("alice")
    c.forget_interaction_experiences("alice", [record.id])
    assert c.relationship_views("alice") == ()
    # Only the review path can trigger now.
    c.settings = replace(c.settings, relationship_reflection_threshold=99.0)
    calls = worker.calls
    await c.reply("Anyway.", conversation_id="a", user_id="alice")
    await c.settle()
    assert worker.calls > calls
    assert next(r for r in stored_views(c) if r.id == view.id).status == "superseded"
    assert c.relationship_views("alice") == ()
    await c.close()


@pytest.mark.asyncio
async def test_a_new_batch_replaces_the_last(tmp_path):
    c = await taught(tmp_path, relationship_reflection_threshold=0.0)
    (first,) = c.relationship_views("alice")
    await c.reply("Still no nicknames.", conversation_id="a", user_id="alice")
    await c.settle()
    views = stored_views(c)
    assert sum(1 for r in views if r.is_active) == 1
    assert any(r.status == "superseded" for r in views)
    assert next(r for r in views if r.id == first.id).status == "superseded"
    await c.close()


@pytest.mark.asyncio
async def test_the_importance_of_the_reflecting_turn_still_counts(tmp_path):
    c = await taught(tmp_path, relationship_reflection_threshold=0.8)
    experiences, views = c._relationship_records("alice")
    assert views and pending_importance(experiences, views) > 0
    await c.close()


@pytest.mark.asyncio
async def test_an_empty_batch_leaves_a_superseded_marker_and_resets_pending(tmp_path):
    answers = []

    def empty_or_views(messages):
        data = json.loads(messages[1].content)
        if "experiences" in data and answers:
            return {"views": []}
        return reflect(messages)

    worker = Worker(empty_or_views)
    c = companion(tmp_path, {"interaction": worker, "relationship": worker},
                  character_development=True, relationship_reflection_threshold=0.8)
    for text in ("Hi", "Please don't use nicknames.", "Thanks."):
        await c.reply(text, conversation_id="a", user_id="alice")
        await c.settle()
    assert c.relationship_views("alice")
    answers.append(1)
    await c.reply("Still no nicknames.", conversation_id="a", user_id="alice")
    await c.settle()
    views = stored_views(c)
    assert c.relationship_views("alice") == ()
    assert not any(r.is_active for r in views)
    marker = next(r for r in views if r.summary == "(no views)")
    assert marker.status == "superseded" and marker.metadata["reflected_at"]
    experiences, all_views = c._relationship_records("alice")
    # Only what came in during the reflecting turn is still pending.
    assert pending_importance(experiences, all_views) < pending_importance(experiences, [])
    await c.close()


@pytest.mark.asyncio
async def test_no_user_id_no_reflection(tmp_path):
    worker = Worker(reflect)
    c = companion(tmp_path, {"interaction": worker, "relationship": worker},
                  character_development=True, relationship_reflection_threshold=0.0)
    await c.reply("Hi", conversation_id="a")
    await c.settle()
    await c.reply("Please don't use nicknames.", conversation_id="a")
    await c.settle()
    assert worker.calls == 0
    await c.close()


@pytest.mark.asyncio
async def test_two_conversations_reflecting_leave_one_batch_active(tmp_path):
    c = await taught(tmp_path, relationship_reflection_threshold=0.0)
    await c.reply("Hello", conversation_id="b", user_id="alice")
    await c.reply("Hello again", conversation_id="a", user_id="alice")
    await c.settle()
    scope = c.interaction_experiences("alice")[0].character_id
    active = [r for r in c._memory_store.list_for_character(scope)
              if r.kind == "relationship_view" and r.is_active]
    assert len(active) == 1
    await c.close()
