"""Inspect a real VRM 0.x or VRM 1.x file and emit a model-bound calibration profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_character_engine_vrm import VRMCalibrationProfile, inspect_vrm_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vrm", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("vrm-calibration"))
    parser.add_argument("--external-animation", action="append", default=[])
    args = parser.parse_args()

    manifest = inspect_vrm_path(args.vrm)
    profile = VRMCalibrationProfile.for_manifest(manifest)
    if args.external_animation:
        profile = VRMCalibrationProfile.from_dict(
            {**profile.to_dict(), "external_animations": args.external_animation}
        )
    report = profile.validate(manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(args.out_dir / "manifest.json")
    profile.save(args.out_dir / "calibration.json")
    (args.out_dir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
