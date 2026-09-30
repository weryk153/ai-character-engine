from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field

from .errors import VisionRateLimitError
from .models import VisionFrame


@dataclass(slots=True)
class FrameGate:
    """Rate-limit and deduplicate high-frequency screen/camera/game frames."""

    min_interval_seconds: float = 0.25
    max_frames_per_minute: int = 120
    deduplicate: bool = True
    _last_accepted_at: float | None = field(default=None, init=False)
    _recent: deque[float] = field(default_factory=deque, init=False)
    _last_fingerprint: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be >= 0")
        if self.max_frames_per_minute < 1:
            raise ValueError("max_frames_per_minute must be >= 1")

    def accept(self, frame: VisionFrame) -> bool:
        now = time.monotonic()
        while self._recent and now - self._recent[0] >= 60:
            self._recent.popleft()
        if len(self._recent) >= self.max_frames_per_minute:
            raise VisionRateLimitError("vision frame rate limit exceeded")
        if self._last_accepted_at is not None and now - self._last_accepted_at < self.min_interval_seconds:
            return False

        fingerprint = self._fingerprint(frame)
        if self.deduplicate and fingerprint is not None and fingerprint == self._last_fingerprint:
            return False

        self._last_accepted_at = now
        self._recent.append(now)
        self._last_fingerprint = fingerprint
        return True

    @staticmethod
    def _fingerprint(frame: VisionFrame) -> str | None:
        image = frame.image
        if image.data is not None:
            return hashlib.sha256(image.data).hexdigest()
        if image.path is not None:
            try:
                return hashlib.sha256(image.read_bytes()).hexdigest()
            except OSError:
                return None
        if image.url is not None:
            return hashlib.sha256(image.url.encode("utf-8")).hexdigest()
        return None
