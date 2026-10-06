"""What she is doing stays in the conversation it is done in.

Teaching the user Japanese in one conversation, she went on teaching in every
new one: what she said she was working on, her short-term goals and her
thoughts belonged to the character and were in every conversation. From
1.2.0 they are marked with the conversation they came from and are in her
mind only there; who she is (tastes, habits, history) and her long-term goals
are still hers everywhere.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

import ai_character_engine
from ai_character_engine import CharacterProfile
from ai_character_engine.companion import (
    CONVERSATION_SELF_MEMORY_KINDS,
    SELF_MEMORY_KINDS,
    CharacterCompanion,
    CompanionSettings,
)
from ai_character_engine.goals.models import (
    GoalEvidenceRef,
    GoalHorizon,
    GoalRecord,
    MotivationKind,
    MotivationSignal,
)
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
from ai_character_engine.long_term_cognition.models import ReflectionRecord
from ai_character_engine.memory.models import MemoryRecord
from ai_character_engine.memory.self_kinds import (
    about_the_user,
    kind_of_what_she_said,
    self_memory_kind,
)

SELF_LINE = "- you said about yourself: "
TEACHING = ("Mei is teaching the user Japanese", "I am teaching you Japanese", "working_on")
TEA = ("Mei loves jasmine tea", "I love jasmine tea", "taste")


class Her:
    """Her model: answers each call with the next line, then the last one again."""

    def __init__(self, *lines):
        self.lines = list(lines) or ["Mm."]
        self.calls = []

    def _next(self):
        return self.lines.pop(0) if len(self.lines) > 1 else self.lines[0]

    async def generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        return LLMResponse(text=self._next(), model="her")

    async def stream_generate(self, messages, *, tools=None):
        self.calls.append(list(messages))
        text = self._next()
        yield LLMStreamChunk(text=text)
        yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="her"))


class Worker:
    """A background model answering with fixed JSON or a function of the prompt."""

    def __init__(self, answer):
        self.answer = answer

    async def generate(self, messages, *, tools=None):
        payload = self.answer(messages) if callable(self.answer) else self.answer
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="worker")


def about_herself(*facts):
    """Reports each (summary, quote, kind) whose quote she said in the lines given."""

    def answer(messages):
        listed = messages[1].content.split("Character lines to extract from:")[1]
        items = [
            {"summary": summary, "kind": kind, "importance": 0.6, "confidence": 0.9, "evidence": quote}
            for summary, quote, kind in facts
            if quote in listed
        ]
        return {"items": items, "confidence": 0.9, "evidence": []}

    return answer


def make(tmp_path, *, her=None, workers=None, **settings):
    every = {
        "emotion_every": 0,
        "mood_every": 0,
        "memory_every": 0,
        "self_memory_every": 0,
        "goal_every": 0,
        "reflection_every": 0,
        "summary_every": 0,
    }
    every.update(settings)
    return CharacterCompanion(
        character=CharacterProfile(id="mei", name="Mei", description="A researcher."),
        llm=her or Her(),
        background_llm=workers or {},
        storage_dir=tmp_path / "engine",
        settings=CompanionSettings(**every),
    )


def self_lines(messages):
    return [
        line.removeprefix(SELF_LINE)
        for message in messages
        for line in message.content.splitlines()
        if line.startswith(SELF_LINE)
    ]


def goal(objective, horizon, *, conversation="a", updated_at=None, before_1_2=False):
    """A goal of ``conversation``; ``before_1_2``: one that names none."""
    metadata = {} if before_1_2 else {"conversation_id": conversation}
    record = GoalRecord(
        "mei",
        objective,
        horizon,
        0.8,
        0.9,
        (
            MotivationSignal(
                kind=MotivationKind.EXPLICIT_REQUEST,
                strength=0.9,
                evidence=GoalEvidenceRef(source_type="event", source_id="e1", excerpt="please"),
                rationale="The user asked",
            ),
        ),
        metadata=metadata,
    )
    return record if updated_at is None else replace(record, updated_at=updated_at)


def said(summary, kind, *, conversation="a", minutes_ago=0, before_1_2=False):
    """Something she said about herself in ``conversation``; ``before_1_2``:
    kept before conversations were named."""
    metadata = {"evidence_type": "character_statement"}
    if not before_1_2:
        metadata["conversation_id"] = conversation
    return MemoryRecord(
        character_id="mei#self",
        summary=summary,
        kind=kind,
        metadata=metadata,
        created_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


def goals_citing_the_event(messages):
    event_id = re.search(r'"event": \{.*?"id": "([^"]+)"', messages[1].content).group(1)

    def one(objective, horizon):
        return {
            "objective": objective,
            "horizon": horizon,
            "urgency": 0.8,
            "conflict_key": None,
            "motivation_signals": [
                {
                    "kind": "explicit_request",
                    "strength": 0.9,
                    "source_type": "event",
                    "source_id": event_id,
                    "rationale": "The user asked",
                }
            ],
            "confidence": 0.9,
        }

    return {
        "goals": [
            one("Give the user the next Japanese exercise", "short_term"),
            one("Become a good teacher", "long_term"),
        ],
        "confidence": 0.9,
        "evidence": [],
    }


THOUGHT = {
    "insight": "I think they really want to learn",
    "belief_candidate": None,
    "confidence": 0.9,
    "evidence": ["teach me Japanese"],
}


# --- the kinds ---------------------------------------------------------------------


def test_the_kinds_she_says_things_about_herself_in():
    assert SELF_MEMORY_KINDS == (
        "identity", "trait", "taste", "habit", "history", "relationship", "opinion",
        "working_on", "plan", "view_of_user",
    )
    assert CONVERSATION_SELF_MEMORY_KINDS == ("working_on", "plan", "view_of_user")
    assert "SELF_MEMORY_KINDS" not in ai_character_engine.__all__


@pytest.mark.parametrize(
    ("named", "kind"),
    [
        ("habit", "habit"),
        ("habits", "habit"),
        ("Plans", "plan"),
        ("working on", "working_on"),
        ("physical_trait", "trait"),
        ("feeling", "view_of_user"),
        ("feelings", "view_of_user"),
        ("preference", "preference"),
        ("", "fact"),
        (None, "fact"),
    ],
)
def test_a_kind_named_freely_is_one_on_the_list(named, kind):
    assert self_memory_kind(named) == kind


@pytest.mark.parametrize(
    "summary",
    [
        "佩克拉認為用戶聽不懂日語發音",
        "佩克拉覺得对方亂打一通",
        "Mei thinks the user is a fool",
        "Mei is annoyed with you",
        "ペコラはあなたの発音が下手だと思っている",
        "메이는 사용자가 바보라고 생각한다",
    ],
)
def test_a_summary_that_names_the_user_is_about_the_user(summary):
    assert about_the_user(summary)


@pytest.mark.parametrize(
    "summary",
    ["佩克拉威脅要把紅蘿蔔丟出去", "Mei grows carrots", "Mei is a youtuber", ""],
)
def test_a_summary_that_does_not_name_the_user_is_not(summary):
    # "youtuber" holds "user" inside a word: not the user.
    assert not about_the_user(summary)


@pytest.mark.parametrize(
    ("kind", "summary", "kept_as"),
    [
        # The comparison pages: a 9B model files its judgements of the user
        # under opinion/trait/habit whatever the prompt says.
        ("opinion", "佩克拉認為用戶聽不懂日語發音", "view_of_user"),
        ("trait", "佩克拉會用中文諧音梗嘲笑用戶的日語發音", "view_of_user"),
        ("habit", "Mei keeps correcting the user's typing", "view_of_user"),
        ("history", "Mei taught the user kana today", "view_of_user"),
        # Not about the user: kept as filed.
        ("trait", "佩克拉威脅要把紅蘿蔔丟出去", "trait"),
        ("opinion", "Mei thinks horror films are boring", "opinion"),
        # "I like you" is hers to keep; a taste is hers too.
        ("relationship", "Mei likes the user", "relationship"),
        ("taste", "Mei likes the same tea as the user", "taste"),
        ("identity", "Mei is the user's tutor", "identity"),
        # Names a model uses still come onto the list first.
        ("Opinions", "Mei thinks the user is a fool", "view_of_user"),
        ("feeling", "Mei feels tired", "view_of_user"),
    ],
)
def test_what_she_thinks_of_the_user_is_a_view_of_the_user_whatever_the_model_filed(
    kind, summary, kept_as
):
    assert kind_of_what_she_said(kind, summary) == kept_as


# --- marking -----------------------------------------------------------------------


def test_what_she_said_is_marked_with_its_conversation_and_kind(tmp_path):
    async def scenario():
        her = Her("I am teaching you Japanese.")
        worker = Worker(about_herself((*TEACHING[:2], "Working On")))
        current = make(tmp_path, her=her, workers={"self_memory": worker}, self_memory_every=1)
        await current.reply("teach me Japanese", conversation_id="a")
        await current.settle()
        records = current.runtime.memory_manager.store.list_for_character("mei#self")
        await current.close()
        return [(r.kind, r.metadata.get("conversation_id")) for r in records]

    assert asyncio.run(scenario()) == [("working_on", "a")]


def test_her_judgement_of_the_user_filed_as_an_opinion_stays_in_its_conversation(tmp_path):
    async def scenario():
        her = Her("You still cannot hear the difference, can you.")
        judgement = (
            "Mei thinks the user cannot hear the difference",
            "You still cannot hear the difference, can you.",
            "opinion",
        )
        worker = Worker(about_herself(judgement))
        current = make(tmp_path, her=her, workers={"self_memory": worker}, self_memory_every=1)
        await current.reply("is it ka or ga?", conversation_id="a")
        await current.settle()
        records = current.runtime.memory_manager.store.list_for_character("mei#self")
        in_a = current.self_memories(in_conversation="a")
        in_b = current.self_memories(in_conversation="b")
        await current.close()
        return [(r.kind, r.metadata.get("conversation_id")) for r in records], in_a, in_b

    kinds, in_a, in_b = asyncio.run(scenario())
    assert kinds == [("view_of_user", "a")]
    assert in_a == ["Mei thinks the user cannot hear the difference"]
    assert in_b == []


def test_goals_and_thoughts_are_marked_with_their_conversation(tmp_path):
    async def scenario():
        workers = {"goal": Worker(goals_citing_the_event), "reflection": Worker(THOUGHT)}
        current = make(tmp_path, workers=workers, goal_every=1, reflection_every=1)
        await current.reply("teach me Japanese", conversation_id="a")
        await current.settle()
        goals = current.runtime.goal_manager.goals(character_id="mei")
        thoughts = current.runtime.long_term_cognition.reflections(character_id="mei")
        await current.close()
        return {g.metadata.get("conversation_id") for g in goals}, [
            t.metadata.get("conversation_id") for t in thoughts
        ]

    assert asyncio.run(scenario()) == ({"a"}, ["a"])


# --- in mind -----------------------------------------------------------------------


def test_what_she_is_doing_stays_in_its_conversation(tmp_path):
    async def scenario():
        her = Her("I love jasmine tea. I am teaching you Japanese.", "Hello.", "Where were we?")
        worker = Worker(about_herself(TEACHING, TEA))
        current = make(tmp_path, her=her, workers={"self_memory": worker}, self_memory_every=1)
        await current.reply("teach me Japanese", conversation_id="a")
        await current.settle()
        await current.reply("hi", conversation_id="b")
        in_b = self_lines(her.calls[-1])
        await current.reply("let's go on", conversation_id="a")
        in_a = self_lines(her.calls[-1])
        everything = current.self_memories()
        await current.close()
        return in_b, in_a, everything

    in_b, in_a, everything = asyncio.run(scenario())
    assert in_b == ["Mei loves jasmine tea"]
    assert set(in_a) == {"Mei loves jasmine tea", "Mei is teaching the user Japanese"}
    # The host's memory page still shows all of it.
    assert set(everything) == {"Mei loves jasmine tea", "Mei is teaching the user Japanese"}


def test_after_a_restart_she_still_knows_what_she_was_doing_where(tmp_path):
    async def scenario():
        worker = Worker(about_herself(TEACHING))
        first = make(
            tmp_path, her=Her("I am teaching you Japanese."), workers={"self_memory": worker},
            self_memory_every=1,
        )
        await first.reply("teach me Japanese", conversation_id="a")
        await first.settle()
        await first.close()
        her = Her("Hello.")
        again = make(tmp_path, her=her)
        await again.reply("hi", conversation_id="b")
        in_b = self_lines(her.calls[-1])
        await again.reply("go on", conversation_id="a")
        in_a = self_lines(her.calls[-1])
        await again.close()
        return in_b, in_a

    assert asyncio.run(scenario()) == ([], ["Mei is teaching the user Japanese"])


def test_self_memories_in_a_conversation(tmp_path):
    async def scenario():
        current = make(tmp_path)
        store = current.runtime.memory_manager.store
        store.add(said("Mei loves jasmine tea", "taste", minutes_ago=3))
        store.add(said("Mei is teaching the user Japanese", "working_on", minutes_ago=2))
        store.add(said("Mei plans a quiz", "plan", conversation="b", minutes_ago=1))
        found = (
            current.self_memories(in_conversation="a"),
            current.self_memories(in_conversation="b"),
            current.self_memories(in_conversation=None),
            current.self_memories(),
        )
        await current.close()
        return found

    in_a, in_b, in_none, everything = asyncio.run(scenario())
    assert in_a == ["Mei loves jasmine tea", "Mei is teaching the user Japanese"]
    assert in_b == ["Mei loves jasmine tea", "Mei plans a quiz"]
    assert in_none == ["Mei loves jasmine tea"]
    assert everything == [
        "Mei loves jasmine tea", "Mei is teaching the user Japanese", "Mei plans a quiz",
    ]


def test_a_host_that_names_no_conversation_keeps_what_she_is_doing_in_mind(tmp_path):
    async def scenario():
        her = Her("I am teaching you Japanese.", "Where were we?")
        worker = Worker(about_herself(TEACHING))
        current = make(tmp_path, her=her, workers={"self_memory": worker}, self_memory_every=1)
        await current.reply("teach me Japanese")
        await current.settle()
        await current.reply("go on")
        lines = self_lines(her.calls[-1])
        marks = [
            record.metadata.get("conversation_id", "none")
            for record in current.runtime.memory_manager.store.list_for_character("mei#self")
        ]
        await current.close()
        return lines, marks

    assert asyncio.run(scenario()) == (["Mei is teaching the user Japanese"], [None])


def test_goals_and_thoughts_of_a_conversation_stay_in_it(tmp_path):
    async def scenario():
        workers = {"goal": Worker(goals_citing_the_event), "reflection": Worker(THOUGHT)}
        her = Her("Let's begin.")
        current = make(tmp_path, her=her, workers=workers, goal_every=1, reflection_every=1)
        await current.reply("teach me Japanese", conversation_id="a")
        await current.settle()
        await current.reply("hi", conversation_id="b")
        in_b = current.snapshot()
        context_b = "\n".join(m.content for m in her.calls[-1])
        await current.reply("go on", conversation_id="a")
        in_a = current.snapshot()
        await current.close()
        return in_b, context_b, in_a

    in_b, context_b, in_a = asyncio.run(scenario())
    assert list(in_b.goals) == ["Become a good teacher"]
    assert in_b.thoughts == ()
    assert "Give the user the next Japanese exercise" not in context_b
    assert "I think they really want to learn" not in context_b
    assert set(in_a.goals) == {"Give the user the next Japanese exercise", "Become a good teacher"}
    assert list(in_a.thoughts) == ["I think they really want to learn"]


def test_a_short_term_goal_untouched_for_a_day_leaves_her_mind(tmp_path):
    async def scenario(**settings):
        current = make(tmp_path / str(len(settings)), **settings)
        store = current.runtime.goal_manager.store
        day_ago = datetime.now(UTC) - timedelta(hours=25)
        store.add_goal(goal("Give the next exercise", GoalHorizon.SHORT_TERM, updated_at=day_ago))
        store.add_goal(goal("Ask about the quiz", GoalHorizon.SHORT_TERM))
        store.add_goal(goal("Become a good teacher", GoalHorizon.LONG_TERM, updated_at=day_ago))
        await current.reply("hi", conversation_id="a")
        goals = set(current.snapshot().goals)
        await current.close()
        return goals

    assert asyncio.run(scenario()) == {"Ask about the quiz", "Become a good teacher"}
    assert asyncio.run(scenario(short_term_goal_max_age_hours=0)) == {
        "Give the next exercise", "Ask about the quiz", "Become a good teacher",
    }


def test_self_memories_kept_are_counted_apart(tmp_path):
    async def scenario():
        current = make(tmp_path, self_memories_kept=2)
        store = current.runtime.memory_manager.store
        for number in range(3):
            store.add(said(f"Mei habit {number}", "habit", minutes_ago=30 - number))
            store.add(said(f"Mei plan a{number}", "plan", minutes_ago=20 - number))
            store.add(said(f"Mei plan b{number}", "plan", conversation="b", minutes_ago=10 - number))
        current.rewrite_self_memories([], edited_from=[])
        kept = current.self_memories()
        await current.close()
        return kept

    assert asyncio.run(scenario()) == [
        "Mei habit 1", "Mei habit 2", "Mei plan a1", "Mei plan a2", "Mei plan b1", "Mei plan b2",
    ]


def edited_on_the_memory_page(tmp_path, records, edit):
    """What she holds after the user's memory page changed line ``edit[0]``
    into ``edit[1]``: (summary, kind, conversation) of each active line."""

    async def scenario():
        current = make(tmp_path)
        store = current.runtime.memory_manager.store
        for record in records:
            store.add(record)
        shown = current.self_memories()
        current.rewrite_self_memories(
            [edit[1] if line == edit[0] else line for line in shown], edited_from=shown
        )
        held = [
            (record.summary, record.kind, record.metadata.get("conversation_id", "none"))
            for record in store.list_for_character("mei#self")
            if record.is_active
        ]
        in_b = current.self_memories(in_conversation="b")
        await current.close()
        return held, in_b

    return asyncio.run(scenario())


def test_an_edited_plan_stays_in_its_conversation(tmp_path):
    """The memory page wrote an edited line back as a fact of no conversation:
    her plan for one conversation was then in every one."""
    held, in_b = edited_on_the_memory_page(
        tmp_path,
        [said("Mei loves jasmine tea", "taste", minutes_ago=2), said("Mei plans a quiz", "plans", minutes_ago=1)],
        ("Mei plans a quiz", "Mei plans a quiz on verbs"),
    )
    assert ("Mei plans a quiz on verbs", "plan", "a") in held
    assert in_b == ["Mei loves jasmine tea"]


def test_an_edited_taste_is_still_hers_everywhere(tmp_path):
    held, in_b = edited_on_the_memory_page(
        tmp_path,
        [said("Mei loves jasmine tea", "taste", minutes_ago=2), said("Mei plans a quiz", "plan", minutes_ago=1)],
        ("Mei loves jasmine tea", "Mei loves green tea"),
    )
    assert ("Mei loves green tea", "taste", "a") in held
    assert in_b == ["Mei loves green tea"]


def test_a_line_added_on_the_memory_page_is_a_fact_of_no_conversation(tmp_path):
    async def scenario():
        current = make(tmp_path)
        store = current.runtime.memory_manager.store
        store.add(said("Mei plans a quiz", "plan"))
        shown = current.self_memories()
        current.rewrite_self_memories([*shown, "Mei is left-handed"], edited_from=shown)
        held = [
            (record.summary, record.kind, record.metadata.get("conversation_id", "none"))
            for record in store.list_for_character("mei#self")
            if record.is_active
        ]
        await current.close()
        return held

    assert asyncio.run(scenario()) == [
        ("Mei plans a quiz", "plan", "a"),
        ("Mei is left-handed", "fact", "none"),
    ]


# --- as before ---------------------------------------------------------------------


def test_with_plans_not_staying_put_everything_is_in_every_conversation(tmp_path):
    async def scenario():
        her = Her("Hello.")
        current = make(tmp_path, her=her, plans_stay_in_conversation=False)
        current.runtime.memory_manager.store.add(
            said("Mei is teaching the user Japanese", "working_on")
        )
        current.runtime.goal_manager.store.add_goal(
            goal("Give the next exercise", GoalHorizon.SHORT_TERM)
        )
        current.runtime.long_term_cognition.store.add_reflection(
            ReflectionRecord("mei", "I think they want to learn", 0.9, metadata={"conversation_id": "a"})
        )
        await current.reply("hi", conversation_id="b")
        snapshot = current.snapshot()
        lines = self_lines(her.calls[-1])
        await current.close()
        return lines, snapshot.goals, snapshot.thoughts

    assert asyncio.run(scenario()) == (
        ["Mei is teaching the user Japanese"],
        ("Give the next exercise",),
        ("I think they want to learn",),
    )


def test_what_was_kept_before_1_2_is_from_another_conversation(tmp_path):
    async def scenario(conversation):
        her = Her("Hello.")
        current = make(tmp_path / str(conversation), her=her)
        current.runtime.memory_manager.store.add(
            said("Mei is teaching the user Japanese", "habit", before_1_2=True, minutes_ago=2)
        )
        current.runtime.memory_manager.store.add(
            said("Mei plans a quiz", "plan", before_1_2=True, minutes_ago=1)
        )
        current.runtime.goal_manager.store.add_goal(
            goal("Give the next exercise", GoalHorizon.SHORT_TERM, before_1_2=True)
        )
        current.runtime.goal_manager.store.add_goal(
            goal("Become a good teacher", GoalHorizon.LONG_TERM, before_1_2=True)
        )
        current.runtime.long_term_cognition.store.add_reflection(
            ReflectionRecord("mei", "I think they want to learn", 0.9)
        )
        await current.reply("hi", conversation_id=conversation)
        snapshot = current.snapshot()
        lines = self_lines(her.calls[-1])
        everything = current.self_memories()
        await current.close()
        return lines, snapshot.goals, snapshot.thoughts, everything

    for conversation in ("a", None):
        lines, goals, thoughts, everything = asyncio.run(scenario(conversation))
        # A habit is hers everywhere, whatever it says; the user clears it on
        # the memory page. A plan of before is from another conversation.
        assert lines == ["Mei is teaching the user Japanese"]
        assert goals == ("Become a good teacher",)
        assert thoughts == ()
        assert everything == ["Mei is teaching the user Japanese", "Mei plans a quiz"]
