from datetime import UTC, datetime
from ai_character_engine.autonomy import AutonomyScheduler, ProactiveCandidate

now = datetime(2026, 9, 21, 12, tzinfo=UTC)
scheduler = AutonomyScheduler(clock=lambda: now)
item = ProactiveCandidate(content="scene changed", source="host", created_at=now)
scheduler.submit(item)
claim = scheduler.claim_next()
assert claim.candidate is item
assert scheduler.in_flight is item
assert scheduler.pending == ()
assert scheduler.claim_next() is None
scheduler.ack(claim)
assert not scheduler.busy
assert scheduler.pending == ()
