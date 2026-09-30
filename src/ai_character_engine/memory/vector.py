from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .embedding import unit_vector


@dataclass(frozen=True, slots=True)
class VectorHit:
    memory_id: str
    score: float


class InMemoryVectorIndex:
    """Exact cosine search, partitioned by character. Rebuildable, not a database.

    One index contains exactly one embedding space. Mutations copy vectors.
    The index stores no memory text and does not decide memory eligibility.
    """

    def __init__(self, *, dimensions: int, model_id: str) -> None:
        if not isinstance(dimensions, int) or dimensions < 1 or not model_id.strip():
            raise ValueError("a positive dimension and non-empty model_id are required")
        self.dimensions = dimensions
        self.model_id = model_id
        self._vectors: dict[tuple[str, str], tuple[float, ...]] = {}
        self._fingerprints: dict[tuple[str, str], str] = {}

    def upsert(
        self,
        *,
        character_id: str,
        memory_id: str,
        vector: Sequence[float],
        content_hash: str = "",
    ) -> None:
        if not character_id or not memory_id:
            raise ValueError("character_id and memory_id must not be empty")
        normalized = unit_vector(vector, self.dimensions)
        key = (character_id, memory_id)
        self._vectors[key] = normalized
        self._fingerprints[key] = content_hash

    def fingerprint(self, *, character_id: str, memory_id: str) -> str | None:
        return self._fingerprints.get((character_id, memory_id))

    def delete(self, *, character_id: str, memory_id: str) -> None:
        key = (character_id, memory_id)
        self._vectors.pop(key, None)
        self._fingerprints.pop(key, None)

    def retain(self, *, character_id: str, memory_ids: Iterable[str]) -> int:
        retained = set(memory_ids)
        removed = 0
        for owner, memory_id in tuple(self._vectors):
            if owner == character_id and memory_id not in retained:
                self.delete(character_id=owner, memory_id=memory_id)
                removed += 1
        return removed

    def search(
        self,
        *,
        character_id: str,
        vector: Sequence[float],
        limit: int = 5,
        min_score: float = math.nextafter(0.0, 1.0),
        allowed_ids: Iterable[str] | None = None,
    ) -> list[VectorHit]:
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be finite and between -1 and 1")
        query = unit_vector(vector, self.dimensions)
        if limit <= 0 or not any(query):
            return []
        allowed = set(allowed_ids) if allowed_ids is not None else None
        hits = []
        for (owner, memory_id), values in self._vectors.items():
            if owner != character_id or (
                allowed is not None and memory_id not in allowed
            ):
                continue
            if not any(values):
                continue
            score = max(-1.0, min(1.0, sum(a * b for a, b in zip(query, values))))
            if score >= min_score:
                hits.append(VectorHit(memory_id, score))
        hits.sort(key=lambda hit: (-hit.score, hit.memory_id))
        return hits[:limit]
