"""Relationship views: derived from reviewed experiences, held only while they are."""

import json
from dataclasses import replace

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
from tests.test_interaction_learning import _held


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
    newer = _held("nick", "Nicknames are fine at home.", minutes=5, topic_id=a.metadata["topic_id"])
    newer = replace(newer, metadata={**newer.metadata, "revises": True})
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
