from __future__ import annotations


class SentenceSegmenter:
    """Accumulates model text deltas and releases TTS-friendly segments."""

    def __init__(self, *, min_chars: int = 4, max_chars: int = 120) -> None:
        if min_chars < 1:
            raise ValueError("min_chars must be >= 1")
        if max_chars < min_chars:
            raise ValueError("max_chars must be >= min_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._boundaries = set("。！？!?\n")

    def push(self, delta: str) -> tuple[str, ...]:
        if not delta:
            return ()
        self._buffer += delta
        emitted: list[str] = []
        while True:
            boundary = self._find_boundary()
            if boundary is None:
                if len(self._buffer) >= self.max_chars:
                    emitted.append(self._take(self.max_chars))
                    continue
                break
            if boundary + 1 < self.min_chars and len(self._buffer) < self.max_chars:
                break
            emitted.append(self._take(boundary + 1))
        return tuple(text for text in emitted if text)

    def flush(self) -> str | None:
        text = self._buffer.strip()
        self._buffer = ""
        return text or None

    def _find_boundary(self) -> int | None:
        if len(self._buffer) < self.min_chars:
            return None
        for index, char in enumerate(self._buffer):
            if char in self._boundaries and index + 1 >= self.min_chars:
                return index
        return None

    def _take(self, count: int) -> str:
        text = self._buffer[:count].strip()
        self._buffer = self._buffer[count:]
        return text
