from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .models import Viseme, VisemeCue


class VisemeAdapter(Protocol):
    def adapt(
        self,
        text: str,
        *,
        duration_ms: float | None,
        metadata: Mapping[str, Any] | None = None,
        allow_text_fallback: bool = True,
    ) -> tuple[VisemeCue, ...]: ...


_ALIASES = {
    "aa": Viseme.A,
    "a": Viseme.A,
    "ih": Viseme.I,
    "i": Viseme.I,
    "ou": Viseme.U,
    "u": Viseme.U,
    "ee": Viseme.E,
    "e": Viseme.E,
    "oh": Viseme.O,
    "o": Viseme.O,
    "sil": Viseme.CLOSED,
    "silence": Viseme.CLOSED,
    "closed": Viseme.CLOSED,
    "neutral": Viseme.NEUTRAL,
}


class MetadataVisemeAdapter:
    """Read provider-supplied timing without depending on one TTS vendor.

    Expected metadata shape::

        {"visemes": [{"viseme": "a", "start_ms": 0, "duration_ms": 80, "weight": 1.0}]}

    Unknown/malformed entries are ignored. The engine never executes metadata as
    code and never assumes a vendor-specific phoneme alphabet.
    """

    def adapt(
        self,
        text: str,
        *,
        duration_ms: float | None,
        metadata: Mapping[str, Any] | None = None,
        allow_text_fallback: bool = True,
    ) -> tuple[VisemeCue, ...]:
        raw = (metadata or {}).get("visemes")
        if not isinstance(raw, (list, tuple)):
            return ()
        result: list[VisemeCue] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("viseme", item.get("label", ""))).strip().lower()
            mapped = _ALIASES.get(label)
            if mapped is None:
                continue
            try:
                start = float(item.get("start_ms", item.get("start_offset_ms", 0.0)))
                item_duration = float(item.get("duration_ms", 0.0))
                weight = float(item.get("weight", 1.0))
                cue = VisemeCue(mapped, start, item_duration, weight, source="tts_metadata")
            except (TypeError, ValueError):
                continue
            if duration_ms is not None and cue.start_offset_ms > duration_ms:
                continue
            result.append(cue)
        return tuple(sorted(result, key=lambda cue: cue.start_offset_ms))


_A = set("aAあぁかがさざただなはばぱまゃやらゎわアァカガサザタダナハバパマャヤラヮワ")
_I = set("iIいぃきぎしじちぢにひびぴみりゐイィキギシジチヂニヒビピミリヰ")
_U = set("uUうぅくぐすずつづぬふぶぷむゅゆるウゥクグスズツヅヌフブプムュユル")
_E = set("eEえぇけげせぜてでねへべぺめれゑエェケゲセゼテデネヘベペメレヱ")
_O = set("oOおぉこごそぞとどのほぼぽもょよろをオォコゴソゾトドノホボポモョヨロヲ")


def _text_viseme(ch: str, previous: Viseme | None) -> Viseme | None:
    if ch == "ー":
        return previous
    if ch in _A:
        return Viseme.A
    if ch in _I:
        return Viseme.I
    if ch in _U:
        return Viseme.U
    if ch in _E:
        return Viseme.E
    if ch in _O:
        return Viseme.O
    return None


class TextVowelVisemeAdapter:
    """Low-confidence fallback for Latin/Japanese vowel-bearing characters.

    This is not phoneme recognition. It is deliberately labelled
    ``text_heuristic`` so a host can prefer real TTS timing when available.
    """

    def adapt(
        self,
        text: str,
        *,
        duration_ms: float | None,
        metadata: Mapping[str, Any] | None = None,
        allow_text_fallback: bool = True,
    ) -> tuple[VisemeCue, ...]:
        if not allow_text_fallback or duration_ms is None or duration_ms <= 0:
            return ()
        labels: list[Viseme] = []
        previous: Viseme | None = None
        for ch in text:
            current = _text_viseme(ch, previous)
            if current is not None:
                labels.append(current)
                previous = current
        if not labels:
            return ()
        step = duration_ms / len(labels)
        return tuple(
            VisemeCue(label, index * step, step, 0.75, source="text_heuristic")
            for index, label in enumerate(labels)
        )


class CompositeVisemeAdapter:
    def __init__(
        self,
        metadata_adapter: VisemeAdapter | None = None,
        fallback_adapter: VisemeAdapter | None = None,
    ) -> None:
        self.metadata_adapter = metadata_adapter or MetadataVisemeAdapter()
        self.fallback_adapter = fallback_adapter or TextVowelVisemeAdapter()

    def adapt(
        self,
        text: str,
        *,
        duration_ms: float | None,
        metadata: Mapping[str, Any] | None = None,
        allow_text_fallback: bool = True,
    ) -> tuple[VisemeCue, ...]:
        precise = self.metadata_adapter.adapt(
            text,
            duration_ms=duration_ms,
            metadata=metadata,
            allow_text_fallback=False,
        )
        if precise:
            return precise
        return self.fallback_adapter.adapt(
            text,
            duration_ms=duration_ms,
            metadata=metadata,
            allow_text_fallback=allow_text_fallback,
        )
