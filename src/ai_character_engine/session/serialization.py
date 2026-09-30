from __future__ import annotations

from datetime import datetime
from typing import Any

from ai_character_engine.llm.models import Message
from ai_character_engine.persistence import PersistenceSurface, migrate_persistence_payload, stamp_current_schema
from ai_character_engine.state.models import CharacterStateSnapshot
from ai_character_engine.tools.models import ToolCall, ToolResult

from .models import SessionRecord, SessionRuntimeSnapshot


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ],
        "tool_result": (
            {
                "call_id": message.tool_result.call_id,
                "name": message.tool_result.name,
                "output": message.tool_result.output,
                "is_error": message.tool_result.is_error,
            }
            if message.tool_result is not None
            else None
        ),
    }


def message_from_dict(payload: dict[str, Any]) -> Message:
    result_payload = payload.get("tool_result")
    return Message(
        role=payload["role"],
        content=payload.get("content", ""),
        tool_calls=tuple(
            ToolCall(
                call_id=item["call_id"],
                name=item["name"],
                arguments=dict(item.get("arguments", {})),
            )
            for item in payload.get("tool_calls", ())
        ),
        tool_result=(
            ToolResult(
                call_id=result_payload["call_id"],
                name=result_payload["name"],
                output=result_payload.get("output", ""),
                is_error=bool(result_payload.get("is_error", False)),
            )
            if result_payload is not None
            else None
        ),
    )


def state_to_dict(state: CharacterStateSnapshot) -> dict[str, Any]:
    return {
        "emotion": state.emotion,
        "energy": state.energy,
        "trust": state.trust,
        "favorability": state.favorability,
        "relationship_stage": state.relationship_stage,
        "custom": state.custom,
    }


def state_from_dict(payload: dict[str, Any]) -> CharacterStateSnapshot:
    return CharacterStateSnapshot(
        emotion=payload.get("emotion", "neutral"),
        energy=float(payload.get("energy", 100.0)),
        trust=float(payload.get("trust", 50.0)),
        favorability=float(payload.get("favorability", 50.0)),
        relationship_stage=payload.get("relationship_stage", "stranger"),
        custom=dict(payload.get("custom", {})),
    )


def snapshot_to_dict(snapshot: SessionRuntimeSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "state": state_to_dict(snapshot.state),
        "history": [message_to_dict(message) for message in snapshot.history],
    }


def snapshot_from_dict(payload: dict[str, Any] | None) -> SessionRuntimeSnapshot | None:
    if payload is None:
        return None
    return SessionRuntimeSnapshot(
        state=state_from_dict(dict(payload.get("state", {}))),
        history=tuple(message_from_dict(item) for item in payload.get("history", ())),
    )


def session_to_dict(record: SessionRecord) -> dict[str, Any]:
    return stamp_current_schema(PersistenceSurface.SESSION, {
        "id": record.id,
        "user_id": record.user_id,
        "character_id": record.character_id,
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "last_activity": record.last_activity.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "closed_at": record.closed_at.isoformat() if record.closed_at else None,
        "metadata": record.metadata,
        "runtime_snapshot": snapshot_to_dict(record.runtime_snapshot),
        "version": record.version,
    })


def session_from_dict(payload: dict[str, Any]) -> SessionRecord:
    payload = dict(migrate_persistence_payload(PersistenceSurface.SESSION, payload).payload)
    return SessionRecord(
        id=payload["id"],
        user_id=payload["user_id"],
        character_id=payload["character_id"],
        status=payload.get("status", "active"),
        created_at=datetime.fromisoformat(payload["created_at"]),
        last_activity=datetime.fromisoformat(payload["last_activity"]),
        expires_at=(
            datetime.fromisoformat(payload["expires_at"])
            if payload.get("expires_at")
            else None
        ),
        closed_at=(
            datetime.fromisoformat(payload["closed_at"])
            if payload.get("closed_at")
            else None
        ),
        metadata=dict(payload.get("metadata", {})),
        runtime_snapshot=snapshot_from_dict(payload.get("runtime_snapshot")),
        version=int(payload.get("version", 1)),
    )
