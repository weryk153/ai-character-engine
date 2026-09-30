import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_qdrant_client_optional_integration():
    if os.getenv("QDRANT_INTEGRATION") != "1":
        pytest.skip("set QDRANT_INTEGRATION=1 to run the real Qdrant integration test")
    pytest.importorskip("qdrant_client")
    from ai_character_engine.memory import QdrantClientBackend

    backend = QdrantClientBackend.from_url(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    await backend.ensure_collection(
        collection_name=os.getenv("QDRANT_COLLECTION", "ai_character_engine_test"),
        dimensions=3,
    )
