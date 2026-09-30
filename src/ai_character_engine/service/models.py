from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    version: str
    details: dict[str, Any] = Field(default_factory=dict)


class SessionCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    character_id: str = Field(min_length=1)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: float | None = Field(default=None, gt=0)

    @field_validator("user_id", "character_id", "session_id")
    @classmethod
    def strip_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("identifier must not be blank")
        return stripped


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    character_id: str
    status: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime | None = None
    closed_at: datetime | None = None
    metadata: dict[str, Any]
    version: int


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        return stripped


class MessageResponse(BaseModel):
    session_id: str
    text: str
    request_id: str
    trace_id: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    rounds: int = 1
    tool_count: int = 0


class ImageMessageRequest(BaseModel):
    prompt: str | None = Field(default=None, max_length=100_000)
    image_base64: str | None = Field(default=None, max_length=12_000_000)
    image_url: str | None = Field(default=None, max_length=4096)
    mime_type: str = Field(min_length=1, max_length=100)
    source_type: Literal["upload", "screenshot", "camera", "game", "other"] = "upload"
    remember: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def model_post_init(self, __context: Any) -> None:
        provided = int(self.image_base64 is not None) + int(self.image_url is not None)
        if provided != 1:
            raise ValueError("exactly one of image_base64 or image_url must be provided")


class VisionMessageResponse(MessageResponse):
    vision_provider: str
    vision_model: str | None = None
    vision_observation: str
    vision_memory: Literal["ephemeral", "persistent"]
    source_type: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class WebSocketClientMessage(BaseModel):
    type: Literal["message", "ping"] = "message"
    content: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


class WebSocketServerEvent(BaseModel):
    type: Literal["ready", "delta", "trace", "final", "error", "pong"]
    request_id: str
    trace_id: str
    data: dict[str, Any] = Field(default_factory=dict)
