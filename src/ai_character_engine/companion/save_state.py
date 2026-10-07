"""Her state as bytes, for a game to keep in its own save.

A save is a zip: ``manifest.json`` (format, engine version, character id, her
clock when it was made, the sha256 of every other file) and one file for each
thing she keeps: her state, her memories, her goals, her thoughts and beliefs,
her diary and what her days were made of, and the conversations she holds.
The stores are read and written here as the companion builds them (the
in-memory and JSONL ones of the engine), in the lines their JSONL files hold.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai_character_engine._version import VERSION
from ai_character_engine.goals.store import JsonlGoalStore, _goal_from_dict, _goal_to_dict
from ai_character_engine.llm.models import Message
from ai_character_engine.long_term_cognition.store import (
    JsonlLongTermCognitionStore,
    _belief_from_dict,
    _belief_to_dict,
    _reflection_from_dict,
    _reflection_to_dict,
)
from ai_character_engine.memory.models import MemoryRecord
from ai_character_engine.memory.store import JsonlMemoryStore
from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload
from ai_character_engine.session.serialization import message_from_dict, message_to_dict

FORMAT_VERSION = 1
MANIFEST = "manifest.json"
STATE = "state.json"
MEMORY = "memory.jsonl"
GOALS = "goals.jsonl"
COGNITION = "cognition.jsonl"
DIARY = "diary.jsonl"
DAY_LOG = "diary_log.jsonl"
CONVERSATIONS = "conversations.json"
FILES = (STATE, MEMORY, GOALS, COGNITION, DIARY, DAY_LOG, CONVERSATIONS)


class StateBusy(RuntimeError):
    """Her background work did not settle in time; nothing was saved."""


class StateFormatError(ValueError):
    """Not a save this engine can load: damaged, or made by a newer one."""


# --- packing -------------------------------------------------------------------


def pack(files: Mapping[str, bytes], *, character_id: str, clock: float) -> bytes:
    manifest = {
        "format_version": FORMAT_VERSION,
        "engine_version": VERSION,
        "character_id": character_id,
        "clock": clock,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def unpack(data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The manifest and every file it lists, each checked against its sha256."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            manifest = json.loads(archive.read(MANIFEST))
            listed = manifest["files"]
            files = {name: archive.read(name) for name in listed}
    except (zipfile.BadZipFile, KeyError, ValueError, TypeError) as exc:
        raise StateFormatError(f"not a save: {exc}") from exc
    version = manifest.get("format_version") if isinstance(manifest, dict) else None
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise StateFormatError("not a save: no format version")
    if not isinstance(listed, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) for name, digest in listed.items()
    ):
        raise StateFormatError("not a save: its manifest lists no files")
    if version > FORMAT_VERSION:
        raise StateFormatError(
            f"made by a newer engine ({manifest.get('engine_version')}, format {version})"
        )
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != listed[name]:
            raise StateFormatError(f"damaged save: {name} does not match its checksum")
    missing = [name for name in FILES if name not in files]
    if missing:
        raise StateFormatError(f"not a save: {', '.join(missing)} missing")
    return manifest, files


# --- the stores ----------------------------------------------------------------


def _lines(payloads, *, sort_keys: bool = False) -> bytes:
    return "".join(
        json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys) + "\n" for payload in payloads
    ).encode("utf-8")


def _read_lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def memory_lines(store) -> bytes:
    return _lines(JsonlMemoryStore._to_dict(record) for record in store._records)


def goal_lines(store) -> bytes:
    return _lines((_goal_to_dict(record) for record in store._goals), sort_keys=True)


def cognition_lines(store) -> bytes:
    return _lines(
        [
            *(_reflection_to_dict(record) for record in store._reflections),
            *(_belief_to_dict(record) for record in store._beliefs),
        ],
        sort_keys=True,
    )


def read_memory(data: bytes) -> list[MemoryRecord]:
    return [
        JsonlMemoryStore._from_dict(
            dict(migrate_persistence_payload(PersistenceSurface.MEMORY, payload).payload)
        )
        for payload in _read_lines(data)
    ]


def read_goals(data: bytes) -> list:
    goals = []
    for payload in _read_lines(data):
        if payload.get("record_type") != "goal":
            raise ValueError(f"unknown goal record_type: {payload.get('record_type')!r}")
        goals.append(
            _goal_from_dict(dict(migrate_persistence_payload(PersistenceSurface.GOAL, payload).payload))
        )
    return goals


def read_cognition(data: bytes) -> tuple[list, list]:
    reflections, beliefs = [], []
    for payload in _read_lines(data):
        kind = payload.get("record_type")
        if kind == "reflection":
            surface, read, kept = PersistenceSurface.REFLECTION, _reflection_from_dict, reflections
        elif kind == "belief":
            surface, read, kept = PersistenceSurface.BELIEF, _belief_from_dict, beliefs
        else:
            raise ValueError(f"unknown cognition record_type: {kind!r}")
        kept.append(read(dict(migrate_persistence_payload(surface, payload).payload)))
    return reflections, beliefs


def put_memory(store, records: list[MemoryRecord]) -> None:
    store._records = list(records)
    if isinstance(store, JsonlMemoryStore):
        store._rewrite()


def put_goals(store, goals: list) -> None:
    store._goals = list(goals)
    if isinstance(store, JsonlGoalStore):
        store._rewrite()


def put_cognition(store, reflections: list, beliefs: list) -> None:
    store._reflections, store._beliefs = list(reflections), list(beliefs)
    if isinstance(store, JsonlLongTermCognitionStore):
        store._rewrite()


def as_another_character(old: str, new: str) -> Callable[[str], str]:
    """Her scopes ("mei", "mei:<conversation>", "mei#self") under another id."""

    def scope(character_id: str) -> str:
        if character_id == old:
            return new
        for mark in (":", "#"):
            if character_id.startswith(old + mark):
                return new + character_id[len(old) :]
        return character_id

    return scope


def rescoped(records: list, scope: Callable[[str], str]) -> list:
    return [replace(record, character_id=scope(record.character_id)) for record in records]


# --- conversations -------------------------------------------------------------


def conversations_to_bytes(
    *,
    active: str | None,
    started: bool,
    kept: list[tuple[str | None, list[Message], list]],
    told: Mapping[str | None, frozenset[str]],
    emotions: Mapping[str | None, Any] = {},
    reply_note: tuple[str | None, str, tuple[str, ...]] | None = None,
) -> bytes:
    return json.dumps(
        {
            "active": active,
            "started": started,
            # A slip of her last reply, pointed out to her on her next one.
            "reply_note": None if reply_note is None else [reply_note[0], reply_note[1], list(reply_note[2])],
            # Oldest first, the one at hand last: the order she lets them go in.
            "conversations": [
                {
                    "id": conversation_id,
                    "history": [message_to_dict(message) for message in history],
                    "notes": [
                        [message_to_dict(anchor), message_to_dict(note)] for anchor, note in notes
                    ],
                    # What the host last passed as what it knows: a line it no
                    # longer passes is taken out of the notes.
                    "told": sorted(told.get(conversation_id, ())),
                    # How the user seemed since the user state was last read.
                    "emotions": list(emotions.get(conversation_id, ())),
                }
                for conversation_id, history, notes in kept
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def conversations_from_bytes(data: bytes) -> dict[str, Any]:
    """``active``, ``started``, ``kept`` (id, history, notes), ``told``,
    ``emotions`` and ``reply_note``."""
    raw = json.loads(data)
    kept = [
        (
            item["id"],
            [message_from_dict(message) for message in item["history"]],
            [(message_from_dict(anchor), message_from_dict(note)) for anchor, note in item["notes"]],
        )
        for item in raw["conversations"]
    ]
    told = {
        item["id"]: frozenset(str(line) for line in item["told"])
        for item in raw["conversations"]
        if item.get("told")
    }
    emotions = {
        item["id"]: [dict(reading) for reading in item["emotions"]]
        for item in raw["conversations"]
        if item.get("emotions")
    }
    note = raw.get("reply_note")
    return {
        "active": raw["active"],
        "started": bool(raw["started"]),
        "kept": kept,
        "told": told,
        "emotions": emotions,
        "reply_note": None if not note else (note[0], str(note[1]), tuple(str(fix) for fix in note[2])),
    }


# --- files ---------------------------------------------------------------------


async def save_state_file(companion, path: str | Path, **options) -> None:
    """``export_state()`` written to ``path``: beside it first, then moved into place."""
    data = await companion.export_state(**options)
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)


def load_state_file(companion, path: str | Path, **options) -> None:
    """``import_state()`` of what ``save_state_file`` wrote."""
    companion.import_state(Path(path).read_bytes(), **options)
