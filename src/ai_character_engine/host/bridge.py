"""Renderer- and product-neutral host bridge for one character session.

The host owns capture, ASR, TTS, playback, UI, and rendering. The core engine
only owns a bounded character turn and neutral multimodal context.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_character_engine.context.budget import ContextBudgetExceededError
from ai_character_engine.context.builder import (
    PICTURE_DESCRIPTION_KEY,
    is_picture_description,
)
from ai_character_engine.events.models import CharacterEvent
from ai_character_engine.llm.models import Message
from ai_character_engine.observability import TraceContext
from ai_character_engine.runtime import CharacterRuntime
from ai_character_engine.runtime.coordination import RuntimeBusyError
from ai_character_engine.runtime.models import CharacterRunResult
from ai_character_engine.vision import ImageInput, VisionFrame, VisionPipeline


class HostBridgeError(RuntimeError):
    """Public, credential-free error suitable for host display."""

    def __init__(self, message, *, requires_review=False):
        self.requires_review = requires_review
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HostBridgeConfig:
    turn_timeout_seconds: float = 90.0
    max_images: int = 4
    max_image_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if (
            not math.isfinite(self.turn_timeout_seconds)
            or self.turn_timeout_seconds <= 0
        ):
            raise ValueError("turn_timeout_seconds must be finite and positive")
        if self.max_images < 1 or self.max_image_bytes < 1:
            raise ValueError("image limits must be positive")


def image_from_host(data: str, mime_type: str, *, max_bytes: int) -> ImageInput:
    """Decode bounded inline host images. Do not fetch arbitrary host URLs or paths."""
    if data.startswith("data:"):
        header, separator, data = data.partition(",")
        if not separator or header != f"data:{mime_type};base64":
            raise HostBridgeError("Invalid image data URL or MIME type.")
    if len(data) > 4 * ((max_bytes + 2) // 3):
        raise HostBridgeError("Image exceeds the configured size limit.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HostBridgeError("Image must be inline base64 data.") from exc
    if not raw or len(raw) > max_bytes:
        raise HostBridgeError(
            "Image is empty or exceeds the configured size limit."
        )
    return ImageInput.from_bytes(raw, mime_type=mime_type)


class CharacterHostBridge:
    def __init__(
        self,
        runtime: CharacterRuntime,
        *,
        vision: VisionPipeline | None = None,
        config: HostBridgeConfig | None = None,
    ):
        self.runtime = runtime
        self.vision = vision
        self.config = config or HostBridgeConfig()
        self._task: asyncio.Task | None = None
        self._closed = False
        self._last_reply: Message | None = None

    @property
    def busy(self) -> bool:
        return self._task is not None or self.runtime.turns.busy

    def replace_reply(self, text: str):
        """Normalize the completed reply using the host's display-language rules."""
        if self.busy:
            raise HostBridgeError("Cannot replace a reply during a character turn.")
        if (
            self._last_reply is not None
            and self.runtime.history
            and self.runtime.history[-1] is self._last_reply
        ):
            self._last_reply = Message("assistant", text)
            self.runtime.history[-1] = self._last_reply

    async def process(
        self,
        text: str,
        *,
        frames: tuple[VisionFrame, ...] = (),
        skip_memory: bool = False,
        proactive: bool = False,
        on_text_delta: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CharacterRunResult:
        if self._closed:
            raise HostBridgeError("Character session is closed.")
        if self.busy:
            raise HostBridgeError("A character turn is already running.")
        if not text.strip() and not frames:
            raise HostBridgeError("A turn needs text or an image.")
        if len(frames) > self.config.max_images:
            raise HostBridgeError("Too many images in one turn.")
        if frames and self.vision is None:
            raise HostBridgeError("Vision is disabled for this character.")
        try:
            with self.runtime.turns.reserve() as lease:
                return await self._process_reserved(
                    text, frames, skip_memory, proactive, lease, on_text_delta
                )
        except RuntimeBusyError as exc:
            raise HostBridgeError("A character turn is already running.") from exc

    async def _process_reserved(
        self, text, frames, skip_memory, proactive, lease, on_text_delta=None
    ):
        history = list(self.runtime.history)
        notes = list(self.runtime.context_notes)
        state = copy.deepcopy(self.runtime.state)
        self._last_reply = None
        memory_manager = self.runtime.memory_manager
        if skip_memory:
            self.runtime.memory_manager = None
        self._task = asyncio.create_task(
            self._process(text, frames, proactive, lease, on_text_delta)
        )
        try:
            result = await asyncio.wait_for(
                self._task, self.config.turn_timeout_seconds
            )
            if not result.text.strip():
                raise HostBridgeError(
                    "The character provider reached its output limit before a "
                    "visible reply. Raise the output limit or disable hidden reasoning."
                    if result.response.metadata.get("finish_reason") == "length"
                    else "The character provider returned an empty reply."
                )
            if skip_memory:
                self.runtime.history = history
                self.runtime.state = state
                self.runtime.context_notes = notes
            else:
                # Visual facts are transient: the description of the picture is
                # not in the conversation (the runtime keeps the turn from the
                # user's words on). Everything else stays exactly as it was sent,
                # the same message objects, so the turn's note stays with them.
                if frames:
                    self.runtime.history = [
                        message
                        for message in self.runtime.history
                        if not is_picture_description(message)
                    ]
                    self._trim_history()
                if (
                    self.runtime.history
                    and self.runtime.history[-1].role == "assistant"
                ):
                    self._last_reply = self.runtime.history[-1]
            return result
        except asyncio.CancelledError as exc:
            self.runtime.history, self.runtime.state = history, state
            self.runtime.context_notes = notes
            exc.requires_review = lease.requires_review
            raise
        except TimeoutError as exc:
            self.runtime.history, self.runtime.state = history, state
            self.runtime.context_notes = notes
            raise HostBridgeError(
                "Character turn timed out. Review possible effects before retry."
                if lease.requires_review else "Character turn timed out. Please try again.",
                requires_review=lease.requires_review,
            ) from exc
        except Exception as exc:
            self.runtime.history, self.runtime.state = history, state
            # The note written for a turn that is taken back goes with it.
            self.runtime.context_notes = notes
            if isinstance(exc, HostBridgeError):
                exc.requires_review |= lease.requires_review
                raise
            if isinstance(exc, ContextBudgetExceededError):
                raise HostBridgeError(
                    "Character context exceeds the configured context budget. "
                    "Shorten the profile or raise the budget.",
                    requires_review=lease.requires_review,
                ) from exc
            raise HostBridgeError(
                "Character processing failed. Review possible effects before retry."
                if lease.requires_review else "Character processing failed. Check the configured providers.",
                requires_review=lease.requires_review,
            ) from exc
        finally:
            self.runtime.memory_manager = memory_manager
            self._task = None

    async def _process(self, text, frames, proactive, lease, on_text_delta=None):
        trace = TraceContext.create().child(character_id=self.runtime.character.id)
        parts = [text] if text.strip() else []
        seen = []
        observations = []
        for frame in frames:
            result = await self.vision.analyze_frame(frame, trace_context=trace)
            if result is not None:
                seen.append(result.event_content)
                observations.append(result.event_payload["vision"])
        if frames and not observations:
            raise HostBridgeError("No visual frame was accepted; try again.")
        payload = (
            {
                "vision_memory": "ephemeral",
                "memory_importance": 0.0,
                "vision": observations,
            }
            if frames
            else {}
        )
        if frames and not proactive:
            # What the picture shows, in a message of its own before the user's
            # words (ContextBuilder); it is not kept in the conversation.
            payload[PICTURE_DESCRIPTION_KEY] = "\n\n".join(seen)
        else:
            # A remark of her own is not kept in the conversation (skip_memory),
            # so what she saw can stay with the prompt that asked for it.
            parts.extend(seen)
        event = CharacterEvent(
            type="proactive_observation"
            if proactive
            else ("multimodal_user_message" if frames else "user_message"),
            source="host",
            content="\n\n".join(parts),
            payload=payload,
        )
        return await self.runtime.process_event(
            event, trace_context=trace, turn_lease=lease, on_text_delta=on_text_delta
        )

    def _trim_history(self):
        # Host-side edits keep the same complete-exchange guarantee as turns. A
        # raw slice can start on a reply or split a tool cycle, which strict
        # provider chat templates reject on every later turn.
        self.runtime._trim_history()

    def restore_history(self, messages: list[Message]):
        """Replace history, dropping leading messages that answer no retained input."""
        if self.busy:
            raise HostBridgeError(
                "Cannot replace history during a character turn."
            )
        history = list(messages)
        start = next(
            (
                index
                for index, message in enumerate(history)
                if message.role in {"user", "event"}
            ),
            len(history),
        )
        self.runtime.history = history[start:]
        # Another conversation, or this one as the host recorded it: the notes
        # written into the previous one do not belong in it.
        self.runtime.context_notes = []
        self._trim_history()
        self._last_reply = None

    def record_interrupted_turn(self, user_text: str, heard_response: str) -> None:
        """Persist conversational continuity after a streamed turn is cancelled.

        Runtime state/memory stay rolled back. This method only records what the
        user said and what audio was conservatively confirmed as fully played.
        It must run after the active runtime turn has released its lease.
        """
        if self.busy:
            raise HostBridgeError(
                "Cannot record an interrupted turn while generation is active."
            )
        heard = heard_response.strip()
        if not heard:
            return
        user = user_text.strip() or "[Audio input]"
        self.runtime.history.extend(
            [
                Message("user", user),
                Message("assistant", heard + " [Interrupted by user]"),
            ]
        )
        self._trim_history()
        self._last_reply = None

    def cancel(self):
        if self._task is not None:
            self._task.cancel()

    def interrupt(self, heard_response: str):
        self.cancel()
        # Never overwrite an older turn when cancellation happens during inference.
        if (
            not self.runtime.turns.busy
            and
            self._last_reply is not None
            and self.runtime.history
            and self.runtime.history[-1] is self._last_reply
        ):
            self.runtime.history[-1] = Message(
                "assistant", (heard_response.strip() + " [Interrupted by user]").strip()
            )
        self._last_reply = None

    async def close(self):
        self._closed = True
        task = self._task
        self.cancel()
        if task is not None:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Providers are owned by the host; borrowed providers are never closed here.
