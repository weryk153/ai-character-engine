from __future__ import annotations

from dataclasses import dataclass, field

from .models import RetrievedMemory


@dataclass(frozen=True, slots=True)
class RetrievalCandidateTrace:
    memory_id: str
    lexical_score: float | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    fusion_score: float | None = None
    final_score: float = 0.0
    selected: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    strategy: str
    character_id: str
    query: str
    requested_limit: int
    query_original: str | None = None
    query_rewritten: str | None = None
    rewrite_used: bool = False
    total_records: int = 0
    active_records: int = 0
    filtered_inactive: int = 0
    candidates: tuple[RetrievalCandidateTrace, ...] = field(default_factory=tuple)
    selected_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    embedding_model_id: str | None = None
    embedding_dimensions: int | None = None
    embedded_records: int = 0
    reused_embeddings: int = 0
    cached_vectors: int = 0
    removed_vectors: int = 0
    candidate_limit: int | None = None
    fusion_k: float | None = None
    lexical_weight: float | None = None
    vector_weight: float | None = None
    min_vector_score: float | None = None
    reranker: str | None = None
    reranker_used: bool = False
    candidate_count: int = 0
    final_count: int = 0
    rewrite_elapsed_ms: float = 0.0
    reranker_elapsed_ms: float = 0.0
    lexical_executed: bool = False
    vector_executed: bool = False
    lexical_elapsed_ms: float = 0.0
    vector_elapsed_ms: float = 0.0
    embedding_elapsed_ms: float = 0.0
    vector_search_elapsed_ms: float = 0.0
    index_backend: str | None = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    memories: tuple[RetrievedMemory, ...]
    trace: RetrievalTrace
