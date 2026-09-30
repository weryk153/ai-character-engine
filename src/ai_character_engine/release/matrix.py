from __future__ import annotations

import platform
import sys

from .models import ReleaseMatrix, ReleaseTarget

SUPPORTED_PYTHON_VERSIONS = ("3.11", "3.12", "3.13")
SUPPORTED_PLATFORMS = ("linux", "macos", "windows")


def default_release_matrix() -> ReleaseMatrix:
    return ReleaseMatrix(
        tuple(
            ReleaseTarget(platform_name, python_version)
            for platform_name in SUPPORTED_PLATFORMS
            for python_version in SUPPORTED_PYTHON_VERSIONS
        )
    )


def current_release_target() -> ReleaseTarget:
    name = platform.system().lower()
    mapping = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    platform_name = mapping.get(name)
    if platform_name is None:
        raise RuntimeError(f"current platform is outside the supported release matrix: {name!r}")
    return ReleaseTarget(platform_name, f"{sys.version_info.major}.{sys.version_info.minor}")
