from ai_character_engine.autonomy import ProactiveCandidate

raw = {"scene": {"objects": ["cup"]}}
item = ProactiveCandidate(content="A cup is visible", source="vision", payload=raw)
raw["scene"]["objects"].append("phone")
assert item.payload["scene"]["objects"] == ("cup",)
copy_for_event = item.event_payload()
copy_for_event["scene"]["objects"].append("book")
assert item.event_payload()["scene"]["objects"] == ["cup"]
