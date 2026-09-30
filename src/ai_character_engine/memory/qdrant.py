from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from .vector import VectorHit


@dataclass(frozen=True, slots=True)
class QdrantPoint:
    memory_id: str
    vector: tuple[float, ...]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QdrantSyncStats:
    upserted: int = 0
    unchanged: int = 0
    removed: int = 0


class QdrantBackend(Protocol):
    async def ensure_collection(
        self, *, collection_name: str, dimensions: int
    ) -> None: ...

    async def fingerprints(
        self, *, collection_name: str, character_id: str, model_id: str
    ) -> dict[str, str]: ...

    async def upsert_points(
        self, *, collection_name: str, points: Sequence[QdrantPoint]
    ) -> None: ...

    async def delete_memory_ids(
        self,
        *,
        collection_name: str,
        character_id: str,
        model_id: str,
        memory_ids: Sequence[str],
    ) -> None: ...

    async def search(
        self,
        *,
        collection_name: str,
        character_id: str,
        model_id: str,
        vector: Sequence[float],
        limit: int,
        min_score: float,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]: ...


def qdrant_point_id(*, character_id: str, memory_id: str, model_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"ai-character-engine:{character_id}:{memory_id}:{model_id}"))


class QdrantVectorIndex:
    """Provider-neutral Qdrant index facade used by the memory retriever."""

    def __init__(
        self,
        *,
        backend: QdrantBackend,
        collection_name: str,
        dimensions: int,
        model_id: str,
    ) -> None:
        if not collection_name.strip() or dimensions < 1 or not model_id.strip():
            raise ValueError("collection_name, model_id and positive dimensions are required")
        self.backend = backend
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.model_id = model_id
        self._ready = False

    async def ensure_ready(self) -> None:
        if not self._ready:
            await self.backend.ensure_collection(
                collection_name=self.collection_name,
                dimensions=self.dimensions,
            )
            self._ready = True

    async def fingerprints(self, *, character_id: str) -> dict[str, str]:
        await self.ensure_ready()
        return await self.backend.fingerprints(
            collection_name=self.collection_name,
            character_id=character_id,
            model_id=self.model_id,
        )

    async def sync_points(
        self,
        *,
        character_id: str,
        points: Sequence[QdrantPoint],
        active_memory_ids: Sequence[str] | None = None,
        existing_fingerprints: Mapping[str, str] | None = None,
    ) -> QdrantSyncStats:
        await self.ensure_ready()
        existing = (
            dict(existing_fingerprints)
            if existing_fingerprints is not None
            else await self.fingerprints(character_id=character_id)
        )
        incoming = {point.memory_id: point for point in points}
        active = set(active_memory_ids) if active_memory_ids is not None else set(incoming)
        changed = [
            point
            for point in points
            if existing.get(point.memory_id) != point.payload.get("content_hash", "")
        ]
        stale = [memory_id for memory_id in existing if memory_id not in active]
        if changed:
            await self.backend.upsert_points(
                collection_name=self.collection_name,
                points=changed,
            )
        if stale:
            await self.backend.delete_memory_ids(
                collection_name=self.collection_name,
                character_id=character_id,
                model_id=self.model_id,
                memory_ids=stale,
            )
        return QdrantSyncStats(
            upserted=len(changed),
            unchanged=max(0, len(active) - len(changed)),
            removed=len(stale),
        )

    async def search(
        self,
        *,
        character_id: str,
        vector: Sequence[float],
        limit: int,
        min_score: float,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        await self.ensure_ready()
        return await self.backend.search(
            collection_name=self.collection_name,
            character_id=character_id,
            model_id=self.model_id,
            vector=vector,
            limit=limit,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )


class QdrantClientBackend:
    """Thin adapter over qdrant-client. Import is optional until instantiated."""

    def __init__(self, client: Any) -> None:
        try:
            from qdrant_client import models
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "install ai-character-engine[qdrant] to use QdrantClientBackend"
            ) from exc
        self.client = client
        self.models = models

    @classmethod
    def from_url(
        cls,
        *,
        url: str,
        api_key: str | None = None,
        prefer_grpc: bool = False,
    ) -> "QdrantClientBackend":
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "install ai-character-engine[qdrant] to use Qdrant"
            ) from exc
        return cls(
            AsyncQdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
            )
        )

    async def ensure_collection(self, *, collection_name: str, dimensions: int) -> None:
        exists = False
        collection_exists = getattr(self.client, "collection_exists", None)
        if callable(collection_exists):
            exists = await collection_exists(collection_name)
        else:  # pragma: no cover - compatibility fallback
            try:
                await self.client.get_collection(collection_name)
                exists = True
            except Exception:
                exists = False
        if not exists:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.models.VectorParams(
                    size=dimensions,
                    distance=self.models.Distance.COSINE,
                ),
            )

    def _base_filter(self, character_id: str, model_id: str):
        m = self.models
        return [
            m.FieldCondition(key="character_id", match=m.MatchValue(value=character_id)),
            m.FieldCondition(key="model_id", match=m.MatchValue(value=model_id)),
        ]

    async def fingerprints(
        self, *, collection_name: str, character_id: str, model_id: str
    ) -> dict[str, str]:
        m = self.models
        result: dict[str, str] = {}
        offset = None
        while True:
            points, next_offset = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=m.Filter(must=self._base_filter(character_id, model_id)),
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                memory_id = payload.get("memory_id")
                if memory_id:
                    result[str(memory_id)] = str(payload.get("content_hash", ""))
            if next_offset is None:
                break
            offset = next_offset
        return result

    async def upsert_points(
        self, *, collection_name: str, points: Sequence[QdrantPoint]
    ) -> None:
        m = self.models
        structs = [
            m.PointStruct(
                id=qdrant_point_id(
                    character_id=str(point.payload["character_id"]),
                    memory_id=point.memory_id,
                    model_id=str(point.payload["model_id"]),
                ),
                vector=list(point.vector),
                payload=point.payload,
            )
            for point in points
        ]
        await self.client.upsert(
            collection_name=collection_name,
            points=structs,
            wait=True,
        )

    async def delete_memory_ids(
        self,
        *,
        collection_name: str,
        character_id: str,
        model_id: str,
        memory_ids: Sequence[str],
    ) -> None:
        if not memory_ids:
            return
        m = self.models
        ids = [
            qdrant_point_id(
                character_id=character_id,
                memory_id=memory_id,
                model_id=model_id,
            )
            for memory_id in memory_ids
        ]
        await self.client.delete(
            collection_name=collection_name,
            points_selector=m.PointIdsList(points=ids),
            wait=True,
        )

    async def search(
        self,
        *,
        collection_name: str,
        character_id: str,
        model_id: str,
        vector: Sequence[float],
        limit: int,
        min_score: float,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        m = self.models
        must = [
            *self._base_filter(character_id, model_id),
            m.FieldCondition(key="status", match=m.MatchValue(value="active")),
        ]
        for key, value in (metadata_filter or {}).items():
            must.append(
                m.FieldCondition(
                    key=f"metadata.{key}",
                    match=m.MatchValue(value=value),
                )
            )
        result = await self.client.query_points(
            collection_name=collection_name,
            query=list(vector),
            query_filter=m.Filter(must=must),
            with_payload=True,
            limit=limit,
            score_threshold=min_score,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            memory_id = payload.get("memory_id")
            if memory_id is not None and float(point.score) >= min_score:
                hits.append(VectorHit(str(memory_id), float(point.score)))
        return hits
