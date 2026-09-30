"""Production Qdrant example.

Setup:
    pip install -e '.[qdrant]'
    docker run -p 6333:6333 qdrant/qdrant

Set OPENAI_API_KEY before using OpenAIEmbeddingProvider, or replace it with any
provider implementing aembed_many/model_id/dimensions.
"""

import asyncio
import os

from ai_character_engine.memory import (
    AsyncHybridMemoryRetriever,
    ImportanceRecencyReranker,
    InMemoryMemoryStore,
    MemoryRecord,
    OpenAIEmbeddingProvider,
    QdrantClientBackend,
    QdrantVectorIndex,
    QdrantVectorMemoryRetriever,
)


async def main() -> None:
    store = InMemoryMemoryStore(
        [
            MemoryRecord(
                character_id="demo",
                summary="使用者喜歡喝 espresso 與手沖咖啡",
                importance=0.8,
                metadata={"locale": "zh-TW"},
            ),
            MemoryRecord(
                character_id="demo",
                summary="使用者喜歡黑澤明電影",
                importance=0.8,
                metadata={"locale": "zh-TW"},
            ),
        ]
    )
    embedding = OpenAIEmbeddingProvider(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
    )
    backend = QdrantClientBackend.from_url(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    index = QdrantVectorIndex(
        backend=backend,
        collection_name=os.getenv("QDRANT_COLLECTION", "character_memories"),
        dimensions=embedding.dimensions,
        model_id=embedding.model_id,
    )
    vector = QdrantVectorMemoryRetriever(
        store,
        embedding=embedding,
        index=index,
        metadata_filter={"locale": "zh-TW"},
    )
    retriever = AsyncHybridMemoryRetriever(
        store,
        vector_retriever=vector,
        reranker=ImportanceRecencyReranker(),
    )
    result = await retriever.retrieve_with_trace_async(
        character_id="demo",
        query="我以前有說過喜歡什麼咖啡嗎？",
    )
    for memory in result.memories:
        print(f"{memory.score:.3f}  {memory.record.summary}")
    print(result.trace)


if __name__ == "__main__":
    asyncio.run(main())
