"""Offline retrieval evaluation example for v0.11.

Run with:
    PYTHONPATH=src python tests/scenarios/rag_evaluation.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ai_character_engine.memory import (
    InMemoryMemoryStore,
    LexicalMemoryRetriever,
    MemoryRecord,
    RetrievalEvaluator,
    load_eval_jsonl,
)


async def main() -> None:
    store = InMemoryMemoryStore()
    store.add(MemoryRecord(character_id="demo", id="coffee", summary="The user likes coffee."))
    store.add(MemoryRecord(character_id="demo", id="seven-samurai", summary="The user's favorite Kurosawa movie is Seven Samurai."))
    store.add(MemoryRecord(character_id="demo", id="cat", summary="The user owns a cat."))

    cases = load_eval_jsonl(Path(__file__).resolve().parents[2] / "examples" / "data" / "retrieval_eval.jsonl")
    report = await RetrievalEvaluator(LexicalMemoryRetriever(store), k=2).evaluate(cases)
    print(report.metrics)
    for case in report.cases:
        print(case.case.case_id, "->", case.retrieved_memory_ids)


if __name__ == "__main__":
    asyncio.run(main())
