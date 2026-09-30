from pathlib import Path
import shutil

import pytest


@pytest.fixture
def unlicensed_project(tmp_path):
    """Exercise missing-license guards on an actual unlicensed candidate copy."""
    from ai_character_engine.release_candidate.provenance import IGNORED

    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "unlicensed"
    shutil.copytree(root, target, ignore=shutil.ignore_patterns(
        *IGNORED, "*.egg-info", "*.pyc", ".DS_Store", "LICENSE", "LICENSE.md", "LICENSE.txt"))
    return target
