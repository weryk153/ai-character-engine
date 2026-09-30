from datetime import UTC, datetime, timedelta
from ai_character_engine.autonomy import AutonomyScheduler, ProactiveCandidate

clock = [datetime(2026, 9, 21, 12, tzinfo=UTC)]
scheduler = AutonomyScheduler(clock=lambda: clock[0])
item = ProactiveCandidate(
    content="idle check", source="test_host", created_at=clock[0],
    not_before=clock[0] + timedelta(minutes=1),
)
scheduler.submit(item)
assert scheduler.next_ready() is None
clock[0] += timedelta(minutes=1)
assert scheduler.next_ready() is item
