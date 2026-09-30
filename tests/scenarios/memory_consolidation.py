from ai_character_engine.memory import (
    InMemoryEventLedger,
    InMemoryMemoryStore,
    MemoryConsolidator,
    MemoryManager,
    MemoryRecord,
)


store = InMemoryMemoryStore(
    [
        MemoryRecord(
            character_id="demo",
            summary="User likes Kurosawa and Seven Samurai.",
            importance=0.7,
        ),
        MemoryRecord(
            character_id="demo",
            summary="User really likes Seven Samurai by Kurosawa.",
            importance=0.75,
        ),
        MemoryRecord(
            character_id="demo",
            summary="User bought coffee.",
            importance=0.4,
        ),
    ]
)

manager = MemoryManager(
    store=store,
    ledger=InMemoryEventLedger(),
    consolidator=MemoryConsolidator(
        store,
        similarity_threshold=0.25,
    ),
    auto_consolidate_threshold=None,
)

print("Before:")
for record in store.list_for_character("demo"):
    print("-", record.kind, record.summary)

result = manager.consolidate(character_id="demo")

print("\nConsolidation:", result)
print("\nAfter:")
for record in store.list_for_character("demo"):
    print("-", record.kind, record.summary, record.metadata)
