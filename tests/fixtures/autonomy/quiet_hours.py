from datetime import UTC, datetime, time, timedelta, timezone
from ai_character_engine.autonomy import QuietHours

window = QuietHours(time(22), time(7), timezone(timedelta(hours=8)))
for hour, minute, expected in [
    (13, 59, False), (14, 0, True),
    (22, 59, True), (23, 0, False),
]:
    moment = datetime(2026, 9, 21, hour, minute, tzinfo=UTC)
    assert window.contains(moment) is expected
