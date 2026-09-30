from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter

from .embedding import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    content_hash,
    memory_text,
    stored_embedding,
)
from .models import RetrievedMemory
from .ranking import MemoryReranker, ReciprocalRankFusion
from .retriever import LexicalMemoryRetriever, MemoryRetriever
from .store import MemoryStore
from .trace import RetrievalCandidateTrace, RetrievalResult, RetrievalTrace
from .vector import InMemoryVectorIndex


class VectorMemoryRetriever(MemoryRetriever):
    def __init__(
        self,
        store: MemoryStore,
        *,
        embedding: EmbeddingProvider | None = None,
        index: InMemoryVectorIndex | None = None,
        min_score: float = math.nextafter(0.0, 1.0),
    ) -> None:
        super().__init__(store)
        self.embedding = embedding if embedding is not None else HashEmbeddingProvider()
        self.index = (
            index
            if index is not None
            else InMemoryVectorIndex(
                dimensions=self.embedding.dimensions,
                model_id=self.embedding.model_id,
            )
        )
        self._check_space()
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be finite and between -1 and 1")
        self.min_score = min_score

    def _check_space(self) -> None:
        if (
            self.index.dimensions != self.embedding.dimensions
            or self.index.model_id != self.embedding.model_id
        ):
            raise ValueError(
                "index and embedding provider must use the same vector space"
            )

    def retrieve_with_trace(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> RetrievalResult:
        start = perf_counter()
        self._check_space()
        trace = RetrievalTrace(
            strategy="vector",
            character_id=character_id,
            query=query,
            requested_limit=limit,
            embedding_model_id=self.embedding.model_id,
            embedding_dimensions=self.embedding.dimensions,
            min_vector_score=self.min_score,
            vector_executed=True,
            index_backend="in-memory",
        )
        if limit <= 0:
            return RetrievalResult(
                (), replace(trace, elapsed_ms=(perf_counter() - start) * 1000)
            )
        records = self.store.list_for_character(character_id)
        active = {
            r.id: r for r in records if r.character_id == character_id and r.is_active
        }
        removed = self.index.retain(character_id=character_id, memory_ids=active)
        embedded = reused = cached = 0
        # The memory store is authoritative: synchronize changed/new/removed records
        # before each search, including revision, forgetting, and consolidation.
        for record in active.values():
            fingerprint = content_hash(record)
            if (
                self.index.fingerprint(character_id=character_id, memory_id=record.id)
                == fingerprint
            ):
                cached += 1
                continue
            vector = stored_embedding(record, self.embedding)
            if vector is None:
                vector = self.embedding.embed(memory_text(record))
                embedded += 1
            else:
                reused += 1
            self.index.upsert(
                character_id=character_id,
                memory_id=record.id,
                vector=vector,
                content_hash=fingerprint,
            )
        hits = (
            self.index.search(
                character_id=character_id,
                vector=self.embedding.embed(query),
                limit=len(active),
                min_score=self.min_score,
                allowed_ids=active,
            )
            if active and query.strip()
            else []
        )
        selected = tuple(
            RetrievedMemory(active[hit.memory_id], hit.score) for hit in hits[:limit]
        )
        selected_ids = tuple(item.record.id for item in selected)
        return RetrievalResult(
            selected,
            replace(
                trace,
                total_records=len(records),
                active_records=len(active),
                filtered_inactive=sum(not r.is_active for r in records),
                embedded_records=embedded,
                reused_embeddings=reused,
                cached_vectors=cached,
                removed_vectors=removed,
                selected_memory_ids=selected_ids,
                candidates=tuple(
                    RetrievalCandidateTrace(
                        memory_id=hit.memory_id,
                        vector_score=hit.score,
                        vector_rank=rank,
                        final_score=hit.score,
                        selected=hit.memory_id in selected_ids,
                    )
                    for rank, hit in enumerate(hits, 1)
                ),
                elapsed_ms=(perf_counter() - start) * 1000,
            ),
        )


class HybridMemoryRetriever(MemoryRetriever):
    def __init__(
        self,
        store: MemoryStore,
        *,
        embedding: EmbeddingProvider | None = None,
        index: InMemoryVectorIndex | None = None,
        candidate_limit: int = 20,
        fusion: ReciprocalRankFusion | None = None,
        reranker: MemoryReranker | None = None,
        min_vector_score: float = math.nextafter(0.0, 1.0),
    ) -> None:
        super().__init__(store)
        if not isinstance(candidate_limit, int) or candidate_limit < 1:
            raise ValueError("candidate_limit must be a positive integer")
        self.lexical = LexicalMemoryRetriever(store)
        self.vector = VectorMemoryRetriever(
            store, embedding=embedding, index=index, min_score=min_vector_score
        )
        self.candidate_limit = candidate_limit
        self.fusion = fusion if fusion is not None else ReciprocalRankFusion()
        self.reranker = reranker

    def retrieve_with_trace(
        self,
        *,
        character_id: str,
        query: str,
        limit: int = 5,
        now: datetime | None = None,
    ) -> RetrievalResult:
        start = perf_counter()
        if limit <= 0:
            return RetrievalResult(
                (),
                RetrievalTrace(
                    strategy="hybrid",
                    character_id=character_id,
                    query=query,
                    requested_limit=limit,
                    elapsed_ms=(perf_counter() - start) * 1000,
                ),
            )
        now = now or datetime.now(UTC)
        pool_limit = max(limit, self.candidate_limit)
        args = dict(character_id=character_id, query=query, limit=pool_limit, now=now)
        lexical = (
            self.lexical.retrieve_with_trace(**args)
            if self.fusion.lexical_weight > 0
            else None
        )
        vector = (
            self.vector.retrieve_with_trace(**args)
            if self.fusion.vector_weight > 0
            else None
        )
        lexical_memories = lexical.memories if lexical else ()
        vector_memories = vector.memories if vector else ()
        fused = self.fusion.fuse(lexical_memories, vector_memories)
        # Normalize by the theoretical best RRF score, not by each query's winner.
        normalized = [
            RetrievedMemory(item.record, item.score / self.fusion.max_score)
            for item in fused
        ]
        ranked = (
            self.reranker.rerank(normalized, query=query, now=now)
            if self.reranker
            else normalized
        )
        # A reranker may reorder/drop candidates, but may not inject a different
        # memory or bypass the lifecycle/character filter.
        allowed = {item.record.id: item.record for item in fused}
        if len({item.record.id for item in ranked}) != len(ranked) or any(
            item.record.id not in allowed
            or item.record != allowed[item.record.id]
            or not math.isfinite(item.score)
            for item in ranked
        ):
            raise ValueError(
                "reranker must return unique, finite-scored input candidates"
            )
        selected = tuple(ranked[:limit])
        selected_ids = tuple(item.record.id for item in selected)
        lexical_scores = {
            item.record.id: (rank, item.score)
            for rank, item in enumerate(lexical_memories, 1)
        }
        vector_scores = {
            item.record.id: (rank, item.score)
            for rank, item in enumerate(vector_memories, 1)
        }
        fused_scores = {item.record.id: item.score for item in fused}
        candidates = []
        for item in ranked:
            memory_id = item.record.id
            lr, ls = lexical_scores.get(memory_id, (None, None))
            vr, vs = vector_scores.get(memory_id, (None, None))
            candidates.append(
                RetrievalCandidateTrace(
                    memory_id=memory_id,
                    lexical_score=ls,
                    vector_score=vs,
                    lexical_rank=lr,
                    vector_rank=vr,
                    fusion_score=fused_scores[memory_id],
                    final_score=item.score,
                    selected=memory_id in selected_ids,
                )
            )
        base_trace = vector.trace if vector else lexical.trace
        return RetrievalResult(
            selected,
            replace(
                base_trace,
                strategy="hybrid",
                requested_limit=limit,
                candidates=tuple(candidates),
                selected_memory_ids=selected_ids,
                candidate_limit=pool_limit,
                fusion_k=self.fusion.k,
                lexical_weight=self.fusion.lexical_weight,
                vector_weight=self.fusion.vector_weight,
                reranker=type(self.reranker).__name__ if self.reranker else None,
                lexical_executed=lexical is not None,
                vector_executed=vector is not None,
                lexical_elapsed_ms=lexical.trace.elapsed_ms if lexical else 0.0,
                vector_elapsed_ms=vector.trace.elapsed_ms if vector else 0.0,
                index_backend=vector.trace.index_backend if vector else None,
                elapsed_ms=(perf_counter() - start) * 1000,
            ),
        )
