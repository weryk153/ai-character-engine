from __future__ import annotations

from dataclasses import dataclass, field

from .models import VisionAnalysis, VisionFrame


@dataclass(slots=True, frozen=True)
class VisionMemoryPolicy:
    """Decides whether a visual observation becomes long-term memory.

    Default is ephemeral. Hosts must opt in through frame metadata or configured
    source types, preventing continuous screen/camera feeds from polluting memory.
    """

    persistent_sources: frozenset[str] = frozenset()
    importance: float = 0.55
    remember_metadata_key: str = "remember"

    def should_persist(self, frame: VisionFrame, analysis: VisionAnalysis) -> bool:
        explicit = frame.metadata.get(self.remember_metadata_key)
        if explicit is not None:
            return bool(explicit)
        return frame.source_type in self.persistent_sources

    def event_payload(self, frame: VisionFrame, analysis: VisionAnalysis) -> dict:
        persist = self.should_persist(frame, analysis)
        return {
            "memory_importance": self.importance if persist else 0.0,
            "vision_memory": "persistent" if persist else "ephemeral",
        }
