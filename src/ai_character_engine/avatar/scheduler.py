from __future__ import annotations

from dataclasses import dataclass

from .models import ExpressionCue, ExpressionRequest


_DEFAULT_EMOTION_MAP = {
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "pleased": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "mad": "angry",
    "surprised": "surprised",
    "surprise": "surprised",
    "relaxed": "relaxed",
    "calm": "relaxed",
}


@dataclass(slots=True)
class _Scheduled:
    request: ExpressionRequest
    start_ms: float
    end_ms: float
    order: int


class EmotionExpressionPolicy:
    def __init__(self, mapping: dict[str, str] | None = None, *, priority: int = 10, weight: float = 0.65) -> None:
        self.mapping = {key.lower(): value for key, value in (mapping or _DEFAULT_EMOTION_MAP).items()}
        self.priority = priority
        self.weight = weight

    def cue(self, emotion: str | None, *, start_ms: float, duration_ms: float) -> ExpressionCue | None:
        if not emotion:
            return None
        expression = self.mapping.get(emotion.strip().lower())
        if expression is None:
            return None
        return ExpressionCue(
            expression,
            start_ms,
            duration_ms,
            weight=self.weight,
            priority=self.priority,
            source="character_state",
            group="face",
        )


class ExpressionScheduler:
    """Resolve host requests and low-priority state emotion per exclusive group."""

    def __init__(self, policy: EmotionExpressionPolicy | None = None) -> None:
        self.policy = policy or EmotionExpressionPolicy()
        self._scheduled: list[_Scheduled] = []
        self._order = 0

    def reset(self) -> None:
        self._scheduled.clear()
        self._order = 0

    def schedule(self, request: ExpressionRequest, *, base_offset_ms: float = 0.0) -> str:
        start = base_offset_ms + request.delay_ms
        self._scheduled.append(_Scheduled(request, start, start + request.duration_ms, self._order))
        self._order += 1
        return request.id

    def resolve(self, *, start_ms: float, duration_ms: float, emotion: str | None = None) -> tuple[ExpressionCue, ...]:
        end_ms = start_ms + duration_ms
        candidates: list[tuple[ExpressionCue, int]] = []
        state_cue = self.policy.cue(emotion, start_ms=start_ms, duration_ms=duration_ms)
        if state_cue is not None:
            candidates.append((state_cue, -1))

        kept: list[_Scheduled] = []
        for item in self._scheduled:
            if item.end_ms > start_ms:
                kept.append(item)
            overlap_start = max(start_ms, item.start_ms)
            overlap_end = min(end_ms, item.end_ms)
            if overlap_end <= overlap_start:
                continue
            req = item.request
            candidates.append(
                (
                    ExpressionCue(
                        req.expression,
                        overlap_start,
                        overlap_end - overlap_start,
                        weight=req.weight,
                        priority=req.priority,
                        fade_in_ms=req.fade_in_ms,
                        fade_out_ms=req.fade_out_ms,
                        source=req.source,
                        group=req.group,
                    ),
                    item.order,
                )
            )
        self._scheduled = kept

        winners: dict[str, tuple[ExpressionCue, int]] = {}
        for cue, order in candidates:
            current = winners.get(cue.group)
            if current is None or (cue.priority, order) >= (current[0].priority, current[1]):
                winners[cue.group] = (cue, order)
        return tuple(winners[group][0] for group in sorted(winners))
