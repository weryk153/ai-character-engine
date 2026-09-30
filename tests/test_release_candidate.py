import json
from pathlib import Path

import pytest

from ai_character_engine.release_candidate import evaluate_release_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_sealed_v048_to_v1_is_exact_and_stable():
    gates = {gate.name: gate for gate in evaluate_release_candidate(ROOT).gates}
    assert gates["api.v048_upgrade"].status.value == "pass"
    assert gates["api.stable_manifest"].status.value == "pass"


@pytest.mark.parametrize("mutation", ["remove_symbol", "candidate_status"])
def test_v1_rejects_changed_api_or_candidate_status(tmp_path, mutation):
    data = json.loads((ROOT / "docs/public_api_v1_stable.json").read_text(encoding="utf-8"))
    if mutation == "remove_symbol":
        data["symbols"].pop()
    else:
        data["metadata"]["status"] = "v1-stable-freeze-candidate"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    gates = {gate.name: gate for gate in evaluate_release_candidate(ROOT, stable_api_path=path).gates}
    assert gates["api.stable_manifest"].status.value == "fail"
    if mutation == "remove_symbol":
        assert gates["api.v048_upgrade"].status.value == "fail"
