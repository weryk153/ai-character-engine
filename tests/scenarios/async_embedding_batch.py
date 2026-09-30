import asyncio

from ai_character_engine.memory import HashEmbeddingProvider, embed_many_async


async def main() -> None:
    provider = HashEmbeddingProvider(dimensions=32)
    vectors = await embed_many_async(
        provider,
        ["使用者喜歡咖啡", "使用者喜歡黑澤明電影"],
    )
    print("model:", provider.model_id)
    print("batch:", len(vectors), "dimensions:", len(vectors[0]))


if __name__ == "__main__":
    asyncio.run(main())
