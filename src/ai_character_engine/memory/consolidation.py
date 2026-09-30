from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from .models import MemoryRecord
from .retriever import _terms
from .store import MemoryStore


@dataclass(slots=True, frozen=True)
class MemoryConsolidationResult:
    character_id: str
    before_count: int
    after_count: int
    created: tuple[MemoryRecord, ...] = field(default_factory=tuple)
    removed_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(self.created or self.removed_ids)


class MemoryConsolidationSummarizer(Protocol):
    def consolidate(self, records: list[MemoryRecord]) -> str:
        ...


class HeuristicConsolidationSummarizer:
    """Deterministic baseline summarizer for repeated memories.

    Later versions may replace this with an LLM summarizer. The reference
    implementation deliberately remains cheap and testable.
    """

    def __init__(self, *, max_chars: int = 700) -> None:
        self.max_chars = max_chars

    def consolidate(self, records: list[MemoryRecord]) -> str:
        unique: list[str] = []
        for record in sorted(records, key=lambda item: item.created_at):
            summary = record.summary.strip()
            if not summary:
                continue
            if any(summary == existing or summary in existing for existing in unique):
                continue
            # Prefer the more informative version when one contains another.
            replaced = False
            for index, existing in enumerate(unique):
                if existing in summary:
                    unique[index] = summary
                    replaced = True
                    break
            if not replaced:
                unique.append(summary)

        if not unique:
            return "Repeated related memory."
        if len(unique) == 1:
            text = unique[0]
        else:
            text = "Repeated related memories: " + " | ".join(unique[:4])
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 1].rstrip() + "…"


@dataclass(slots=True)
class MemoryConsolidator:
    store: MemoryStore
    similarity_threshold: float = 0.58
    min_cluster_size: int = 2
    summarizer: MemoryConsolidationSummarizer = field(
        default_factory=HeuristicConsolidationSummarizer
    )

    def consolidate(self, *, character_id: str) -> MemoryConsolidationResult:
        records = self.store.list_for_character(character_id)
        before_count = len(records)
        if before_count < self.min_cluster_size:
            return MemoryConsolidationResult(
                character_id=character_id,
                before_count=before_count,
                after_count=before_count,
            )

        # Event memories can merge with an existing consolidated memory. This
        # lets repeated evidence reinforce a stable fact instead of creating a
        # fresh duplicate next to the old consolidated record.
        candidates = [
            record
            for record in records
            if record.is_active and record.kind in {"event", "consolidated"}
        ]
        untouched = [
            record
            for record in records
            if not record.is_active or record.kind not in {"event", "consolidated"}
        ]
        clusters = self._clusters(candidates)

        created: list[MemoryRecord] = []
        removed_ids: list[str] = []
        survivors: list[MemoryRecord] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                survivors.extend(cluster)
                continue
            created.append(self._merge_cluster(character_id, cluster))
            removed_ids.extend(record.id for record in cluster)

        replacement = [*untouched, *survivors, *created]
        if removed_ids:
            self.store.replace_for_character(character_id, replacement)

        return MemoryConsolidationResult(
            character_id=character_id,
            before_count=before_count,
            after_count=len(replacement) if removed_ids else before_count,
            created=tuple(created),
            removed_ids=tuple(removed_ids),
        )

    def _clusters(self, records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        clusters: list[list[MemoryRecord]] = []
        for record in sorted(records, key=lambda item: item.created_at):
            record_terms = _terms(record.summary + " " + " ".join(record.tags))
            best_cluster: list[MemoryRecord] | None = None
            best_score = 0.0
            for cluster in clusters:
                representative_terms: set[str] = set()
                for item in cluster:
                    representative_terms.update(
                        _terms(item.summary + " " + " ".join(item.tags))
                    )
                score = self._jaccard(record_terms, representative_terms)
                # Same event type is a useful weak hint, but not enough alone.
                if (
                    record.source_event_type
                    and any(
                        item.source_event_type == record.source_event_type
                        for item in cluster
                    )
                ):
                    score += 0.08
                if score > best_score:
                    best_cluster = cluster
                    best_score = score
            if best_cluster is not None and best_score >= self.similarity_threshold:
                best_cluster.append(record)
            else:
                clusters.append([record])
        return clusters

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _merge_cluster(
        self,
        character_id: str,
        cluster: list[MemoryRecord],
    ) -> MemoryRecord:
        summary = self.summarizer.consolidate(cluster)
        tags = tuple(
            dict.fromkeys(tag for record in cluster for tag in record.tags)
        )
        source_ids: list[str] = []
        occurrences = 0
        for record in cluster:
            occurrences += int(record.metadata.get("occurrences", 1))
            existing_sources = record.metadata.get("source_memory_ids")
            if isinstance(existing_sources, list):
                source_ids.extend(str(item) for item in existing_sources)
            else:
                source_ids.append(record.id)
        source_ids = list(dict.fromkeys(source_ids))
        latest = max(
            datetime.fromisoformat(str(record.metadata["last_reinforced_at"]))
            if record.metadata.get("last_reinforced_at")
            else record.created_at
            for record in cluster
        )
        importance = min(
            1.0,
            max(record.importance for record in cluster)
            + min(0.20, 0.03 * max(1, occurrences - 1)),
        )
        return MemoryRecord(
            character_id=character_id,
            summary=summary,
            importance=importance,
            kind="consolidated",
            tags=tags,
            metadata={
                "occurrences": occurrences,
                "source_memory_ids": source_ids,
                "last_reinforced_at": latest.isoformat(),
            },
            created_at=min(record.created_at for record in cluster),
        )
