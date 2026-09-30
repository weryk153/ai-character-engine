from __future__ import annotations

import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .models import SpanRecord, TraceContext
from .sink import NullObservabilitySink, ObservabilitySink


class Span(AbstractContextManager["Span"]):
    def __init__(
        self,
        *,
        sink: ObservabilitySink,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("span name must not be empty")
        self.sink = sink
        self.context = context
        self.name = name
        self.attributes = dict(attributes or {})
        self.span_id = uuid4().hex
        self.started_at = datetime.now(UTC)
        self._started = time.perf_counter()
        self._finished = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._finished:
            return False
        self._finished = True
        ended_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - self._started) * 1000
        self.sink.record_span(
            SpanRecord(
                trace_id=self.context.trace_id,
                span_id=self.span_id,
                parent_span_id=self.context.parent_span_id,
                name=self.name,
                started_at=self.started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                status="error" if exc is not None else "ok",
                request_id=self.context.request_id,
                session_id=self.context.session_id,
                user_id=self.context.user_id,
                character_id=self.context.character_id,
                attributes=dict(self.attributes),
                error_type=type(exc).__name__ if exc is not None else None,
                error_message=str(exc) if exc is not None else None,
            )
        )
        return False


class Tracer:
    def __init__(self, sink: ObservabilitySink | None = None) -> None:
        self.sink = sink or NullObservabilitySink()

    def span(
        self,
        name: str,
        *,
        context: TraceContext,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return Span(sink=self.sink, context=context, name=name, attributes=attributes)
