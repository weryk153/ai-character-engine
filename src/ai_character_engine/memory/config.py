from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_limit: int = 20
    result_limit: int = 5
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    min_vector_score: float = 0.0


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    collection_name: str = "ai_character_memories"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    prefer_grpc: bool = False
