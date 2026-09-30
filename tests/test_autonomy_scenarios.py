"""Executable autonomy regression scenarios and host lifecycle checks."""
import os
from pathlib import Path
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "tests/fixtures/autonomy").glob("*.py"))


def execute(*args, source=None):
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, *args], input=source, text=True, cwd=ROOT,
        env=environment, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_all_autonomy_regression_scenarios_are_present():
    assert len(SCENARIOS) == 7


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.stem)
def test_autonomy_regression(scenario):
    execute(str(scenario))


@pytest.mark.parametrize("example,expected", [
    ("autonomous_character.py", "delivered"),
    ("autonomy_retry.py", "held"),
    ("autonomy_host.py", "history messages=2"),
])
def test_offline_examples(example, expected):
    directory = "examples" if example == "autonomy_host.py" else "tests/scenarios"
    assert expected in execute(str(ROOT / directory / example))


def test_idle_host_boundaries():
    # Importing an example must not start a host task or make a model call.
    import importlib.util

    spec = importlib.util.spec_from_file_location("autonomy_host_example", ROOT / "examples/autonomy_host.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        now = datetime(2026, 9, 21, 12, tzinfo=UTC)
        detector = module.InactivityDetector(now)
        assert detector.poll(now + timedelta(minutes=9)) is None
        assert detector.poll(now + timedelta(minutes=10)) is not None
        assert detector.poll(now + timedelta(minutes=11)) is None
        detector.touch(now + timedelta(minutes=12))
        assert detector.poll(now + timedelta(minutes=21)) is None
        assert detector.poll(now + timedelta(minutes=22)) is not None
    finally:
        sys.modules.pop(spec.name, None)
