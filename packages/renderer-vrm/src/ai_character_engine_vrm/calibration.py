from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


_JSON_CHUNK = 0x4E4F534A
_GLB_MAGIC = b"glTF"


class VRMInspectionError(ValueError):
    """Raised when a file is not a supported GLB/VRM container."""


class CalibrationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VRMSpecFamily(StrEnum):
    VRM0 = "vrm0"
    VRM1 = "vrm1"


@dataclass(frozen=True, slots=True)
class VRMModelManifest:
    model_name: str
    model_version: str | None
    spec_version: str
    generator: str | None
    sha256: str
    expressions: tuple[str, ...]
    custom_expressions: tuple[str, ...]
    humanoid_bones: tuple[str, ...]
    animation_names: tuple[str, ...]
    extensions: tuple[str, ...]
    look_at_type: str | None = None
    spec_family: VRMSpecFamily = VRMSpecFamily.VRM1
    expression_presets: dict[str, str] = field(default_factory=dict)
    secondary_animation_groups: int = 0
    collider_groups: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "spec_version": self.spec_version,
            "generator": self.generator,
            "sha256": self.sha256,
            "expressions": list(self.expressions),
            "custom_expressions": list(self.custom_expressions),
            "humanoid_bones": list(self.humanoid_bones),
            "animation_names": list(self.animation_names),
            "extensions": list(self.extensions),
            "look_at_type": self.look_at_type,
            "spec_family": self.spec_family.value,
            "expression_presets": dict(self.expression_presets),
            "secondary_animation_groups": self.secondary_animation_groups,
            "collider_groups": self.collider_groups,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VRMModelManifest":
        return cls(
            model_name=str(data["model_name"]),
            model_version=str(data["model_version"]) if data.get("model_version") is not None else None,
            spec_version=str(data["spec_version"]),
            generator=str(data["generator"]) if data.get("generator") is not None else None,
            sha256=str(data["sha256"]),
            expressions=tuple(data.get("expressions", ())),
            custom_expressions=tuple(data.get("custom_expressions", ())),
            humanoid_bones=tuple(data.get("humanoid_bones", ())),
            animation_names=tuple(data.get("animation_names", ())),
            extensions=tuple(data.get("extensions", ())),
            look_at_type=str(data["look_at_type"]) if data.get("look_at_type") is not None else None,
            spec_family=VRMSpecFamily(data.get("spec_family") or ("vrm1" if str(data.get("spec_version", "")).startswith("1") else "vrm0")),
            expression_presets={str(k): str(v) for k, v in (data.get("expression_presets") or {}).items()},
            secondary_animation_groups=int(data.get("secondary_animation_groups", 0)),
            collider_groups=int(data.get("collider_groups", 0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VRMModelManifest":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")


@dataclass(frozen=True, slots=True)
class CalibrationIssue:
    severity: CalibrationSeverity
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    model_name: str
    ready: bool
    issues: tuple[CalibrationIssue, ...]

    @property
    def errors(self) -> tuple[CalibrationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is CalibrationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[CalibrationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is CalibrationSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ready": self.ready,
            "issues": [item.to_dict() for item in self.issues],
        }


def _finite(name: str, value: float, *, minimum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class VRMCalibrationProfile:
    """Model-bound renderer calibration without modifying the VRM file itself."""

    model_sha256: str | None = None
    expression_map: dict[str, str] = field(default_factory=dict)
    animation_map: dict[str, str] = field(default_factory=dict)
    external_animations: tuple[str, ...] = ()
    required_expressions: tuple[str, ...] = ("aa", "blink")
    required_bones: tuple[str, ...] = ("head",)
    head_bone: str = "head"
    mouth_gain: float = 1.0
    expression_gain: float = 1.0
    blink_gain: float = 1.0
    gaze_yaw_scale: float = 1.0
    gaze_pitch_scale: float = 1.0
    max_gaze_yaw_deg: float = 35.0
    max_gaze_pitch_deg: float = 25.0
    head_motion_scale: float = 1.0
    max_head_yaw_delta_deg: float = 10.0
    max_head_pitch_delta_deg: float = 8.0
    max_head_roll_delta_deg: float = 5.0

    def __post_init__(self) -> None:
        if self.model_sha256 is not None:
            digest = self.model_sha256.lower().strip()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("model_sha256 must be a 64-character hex digest")
            object.__setattr__(self, "model_sha256", digest)
        for name in ("mouth_gain", "expression_gain", "blink_gain", "gaze_yaw_scale", "gaze_pitch_scale", "head_motion_scale"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        for name in ("max_gaze_yaw_deg", "max_gaze_pitch_deg", "max_head_yaw_delta_deg", "max_head_pitch_delta_deg", "max_head_roll_delta_deg"):
            object.__setattr__(self, name, _finite(name, getattr(self, name), minimum=0.0))
        if not self.head_bone.strip():
            raise ValueError("head_bone must not be empty")
        object.__setattr__(self, "expression_map", {str(k): str(v) for k, v in self.expression_map.items()})
        object.__setattr__(self, "animation_map", {str(k): str(v) for k, v in self.animation_map.items()})
        object.__setattr__(self, "external_animations", tuple(str(x) for x in self.external_animations))
        object.__setattr__(self, "required_expressions", tuple(str(x) for x in self.required_expressions))
        object.__setattr__(self, "required_bones", tuple(str(x) for x in self.required_bones))

    @classmethod
    def for_manifest(cls, manifest: VRMModelManifest) -> "VRMCalibrationProfile":
        expressions = set(manifest.expressions) | set(manifest.custom_expressions)
        standard = ("happy", "angry", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight", "neutral")
        if manifest.spec_family is VRMSpecFamily.VRM1:
            expression_map = {name: name for name in standard if name in expressions}
        else:
            # VRM 0.x uses BlendShapeGroup names/presets. Normalize those legacy
            # names into the engine's VRM 1-style logical vocabulary while still
            # sending the renderer the model's actual group name.
            by_name = {name.lower(): name for name in expressions}
            by_preset: dict[str, str] = {}
            for group_name, preset_name in manifest.expression_presets.items():
                if preset_name:
                    by_preset.setdefault(preset_name.lower(), group_name)

            aliases: dict[str, tuple[str, ...]] = {
                "aa": ("a", "aa"),
                "ih": ("i", "ih"),
                "ou": ("u", "ou"),
                "ee": ("e", "ee"),
                "oh": ("o", "oh"),
                "blink": ("blink",),
                "blinkLeft": ("blink_l", "blinkleft"),
                "blinkRight": ("blink_r", "blinkright"),
                "angry": ("angry",),
                "happy": ("joy", "happy", "fun"),
                "relaxed": ("fun", "relaxed", "joy"),
                "sad": ("sorrow", "sad"),
                "surprised": ("surprised",),
                "neutral": ("neutral",),
            }
            expression_map: dict[str, str] = {}
            for logical, candidates in aliases.items():
                target = next((by_preset[c] for c in candidates if c in by_preset), None)
                if target is None:
                    target = next((by_name[c] for c in candidates if c in by_name), None)
                if target is not None:
                    expression_map[logical] = target

        return cls(
            model_sha256=manifest.sha256,
            expression_map=expression_map,
            animation_map={name: name for name in manifest.animation_names},
        )

    def mapped_expression(self, name: str) -> str:
        return self.expression_map.get(name, name)

    def mapped_animation(self, name: str) -> str:
        return self.animation_map.get(name, name)

    def validate(self, manifest: VRMModelManifest) -> CalibrationReport:
        issues: list[CalibrationIssue] = []
        if manifest.spec_family is VRMSpecFamily.VRM0:
            if not manifest.spec_version.startswith("0"):
                issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "unsupported_vrm_version", f"Unsupported VRM 0.x spec version: {manifest.spec_version}."))
            else:
                issues.append(CalibrationIssue(
                    CalibrationSeverity.INFO,
                    "legacy_vrm0_compatibility",
                    f"VRM {manifest.spec_version} is using the VRM 0.x compatibility adapter.",
                ))
        elif manifest.spec_family is VRMSpecFamily.VRM1:
            if not manifest.spec_version.startswith("1"):
                issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "unsupported_vrm_version", f"Unsupported VRM 1.x spec version: {manifest.spec_version}."))
        else:
            issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "unsupported_vrm_version", f"Unsupported VRM family: {manifest.spec_family}."))
        if self.model_sha256 is not None and self.model_sha256 != manifest.sha256:
            issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "model_hash_mismatch", "Calibration profile belongs to a different VRM file."))

        expressions = set(manifest.expressions) | set(manifest.custom_expressions)
        for logical in self.required_expressions:
            target = self.mapped_expression(logical)
            if target not in expressions:
                issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "missing_required_expression", f"Required expression '{logical}' maps to missing target '{target}'."))
        for logical, target in self.expression_map.items():
            if target not in expressions:
                issues.append(CalibrationIssue(CalibrationSeverity.WARNING, "missing_expression_target", f"Expression '{logical}' maps to unavailable target '{target}'."))

        bones = set(manifest.humanoid_bones)
        for bone in self.required_bones:
            if bone not in bones:
                issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "missing_required_bone", f"Required humanoid bone '{bone}' is missing."))
        if self.head_bone not in bones:
            issues.append(CalibrationIssue(CalibrationSeverity.ERROR, "missing_head_bone", f"Configured head bone '{self.head_bone}' is missing."))
        if manifest.look_at_type is None:
            issues.append(CalibrationIssue(CalibrationSeverity.WARNING, "look_at_unavailable", "VRM lookAt is not declared; gaze may need a renderer-side fallback."))

        embedded = set(manifest.animation_names)
        external = set(self.external_animations)
        for logical, target in self.animation_map.items():
            if target not in embedded and target not in external:
                issues.append(CalibrationIssue(CalibrationSeverity.WARNING, "animation_target_unavailable", f"Animation '{logical}' maps to '{target}', but it is neither embedded nor declared as a host animation."))
        if not embedded and not external:
            issues.append(CalibrationIssue(CalibrationSeverity.INFO, "no_animation_library", "The VRM contains no embedded animations; gestures/idle motion require host-provided clips or VRMA."))

        return CalibrationReport(manifest.model_name, not any(i.severity is CalibrationSeverity.ERROR for i in issues), tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_sha256": self.model_sha256,
            "expression_map": dict(self.expression_map),
            "animation_map": dict(self.animation_map),
            "external_animations": list(self.external_animations),
            "required_expressions": list(self.required_expressions),
            "required_bones": list(self.required_bones),
            "head_bone": self.head_bone,
            "mouth_gain": self.mouth_gain,
            "expression_gain": self.expression_gain,
            "blink_gain": self.blink_gain,
            "gaze_yaw_scale": self.gaze_yaw_scale,
            "gaze_pitch_scale": self.gaze_pitch_scale,
            "max_gaze_yaw_deg": self.max_gaze_yaw_deg,
            "max_gaze_pitch_deg": self.max_gaze_pitch_deg,
            "head_motion_scale": self.head_motion_scale,
            "max_head_yaw_delta_deg": self.max_head_yaw_delta_deg,
            "max_head_pitch_delta_deg": self.max_head_pitch_delta_deg,
            "max_head_roll_delta_deg": self.max_head_roll_delta_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VRMCalibrationProfile":
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "VRMCalibrationProfile":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")


def _manifest_from_gltf(gltf: dict[str, Any], sha256: str) -> VRMModelManifest:
    root_extensions = gltf.get("extensions") or {}
    vrm1 = root_extensions.get("VRMC_vrm")
    if isinstance(vrm1, dict):
        meta = vrm1.get("meta") or {}
        expressions = vrm1.get("expressions") or {}
        preset = tuple(sorted((expressions.get("preset") or {}).keys()))
        custom = tuple(sorted((expressions.get("custom") or {}).keys()))
        bones = tuple(sorted(((vrm1.get("humanoid") or {}).get("humanBones") or {}).keys()))
        animations = tuple(a.get("name") for a in gltf.get("animations", []) if isinstance(a, dict) and a.get("name"))
        look_at = vrm1.get("lookAt") or {}
        spring = root_extensions.get("VRMC_springBone") or {}
        return VRMModelManifest(
            model_name=str(meta.get("name") or "Unnamed VRM"),
            model_version=str(meta.get("version")) if meta.get("version") is not None else None,
            spec_version=str(vrm1.get("specVersion") or "1.0"),
            generator=(gltf.get("asset") or {}).get("generator"),
            sha256=sha256,
            expressions=preset,
            custom_expressions=custom,
            humanoid_bones=bones,
            animation_names=animations,
            extensions=tuple(gltf.get("extensionsUsed") or ()),
            look_at_type=str(look_at.get("type")).lower() if look_at.get("type") is not None else None,
            spec_family=VRMSpecFamily.VRM1,
            expression_presets={name: name for name in preset},
            secondary_animation_groups=len(spring.get("springs") or ()),
            collider_groups=len(spring.get("colliderGroups") or ()),
        )

    legacy = root_extensions.get("VRM")
    if not isinstance(legacy, dict):
        raise VRMInspectionError("GLB does not contain VRMC_vrm or legacy VRM extension")

    meta = legacy.get("meta") or {}
    humanoid = legacy.get("humanoid") or {}
    raw_bones = humanoid.get("humanBones") or ()
    bones = tuple(sorted({
        str(item.get("bone"))
        for item in raw_bones
        if isinstance(item, dict) and item.get("bone")
    }))

    blend_shape_master = legacy.get("blendShapeMaster") or {}
    raw_groups = blend_shape_master.get("blendShapeGroups") or ()
    expression_names: list[str] = []
    expression_presets: dict[str, str] = {}
    for item in raw_groups:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("presetName") or "").strip()
        if not name:
            continue
        expression_names.append(name)
        preset_name = str(item.get("presetName") or "").strip()
        if preset_name:
            expression_presets[name] = preset_name

    first_person = legacy.get("firstPerson") or {}
    look_at_type = first_person.get("lookAtTypeName")
    secondary = legacy.get("secondaryAnimation") or {}
    animations = tuple(a.get("name") for a in gltf.get("animations", []) if isinstance(a, dict) and a.get("name"))

    return VRMModelManifest(
        model_name=str(meta.get("title") or "Unnamed VRM"),
        model_version=str(meta.get("version")) if meta.get("version") is not None else None,
        spec_version=str(legacy.get("specVersion") or "0.x"),
        generator=(gltf.get("asset") or {}).get("generator"),
        sha256=sha256,
        expressions=tuple(sorted(set(expression_names))),
        custom_expressions=(),
        humanoid_bones=bones,
        animation_names=animations,
        extensions=tuple(gltf.get("extensionsUsed") or ()),
        look_at_type=str(look_at_type).lower() if look_at_type is not None else None,
        spec_family=VRMSpecFamily.VRM0,
        expression_presets=expression_presets,
        secondary_animation_groups=len(secondary.get("boneGroups") or ()),
        collider_groups=len(secondary.get("colliderGroups") or ()),
    )


def _parse_glb_json(data: bytes) -> dict[str, Any]:
    if len(data) < 20 or data[:4] != _GLB_MAGIC:
        raise VRMInspectionError("not a GLB/VRM file")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise VRMInspectionError(f"unsupported GLB version {version}")
    if total_length > len(data):
        raise VRMInspectionError("truncated GLB")
    offset = 12
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > total_length:
            raise VRMInspectionError("truncated GLB chunk")
        if chunk_type == _JSON_CHUNK:
            try:
                return json.loads(data[offset:end].rstrip(b"\x00 \t\r\n"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VRMInspectionError("invalid GLB JSON chunk") from exc
        offset = end
    raise VRMInspectionError("GLB JSON chunk not found")


def inspect_vrm_bytes(data: bytes) -> VRMModelManifest:
    if not isinstance(data, bytes) or not data:
        raise VRMInspectionError("VRM bytes must not be empty")
    gltf = _parse_glb_json(data)
    return _manifest_from_gltf(gltf, hashlib.sha256(data).hexdigest())


def inspect_vrm_path(path: str | Path, *, max_json_bytes: int = 8 * 1024 * 1024) -> VRMModelManifest:
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        header = fh.read(12)
        if len(header) != 12 or header[:4] != _GLB_MAGIC:
            raise VRMInspectionError("not a GLB/VRM file")
        version, total_length = struct.unpack_from("<II", header, 4)
        if version != 2:
            raise VRMInspectionError(f"unsupported GLB version {version}")
        sha.update(header)
        json_obj: dict[str, Any] | None = None
        consumed = 12
        while consumed < total_length:
            chunk_header = fh.read(8)
            if len(chunk_header) != 8:
                raise VRMInspectionError("truncated GLB chunk header")
            sha.update(chunk_header)
            consumed += 8
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_length < 0 or consumed + chunk_length > total_length:
                raise VRMInspectionError("invalid GLB chunk length")
            remaining = chunk_length
            if chunk_type == _JSON_CHUNK:
                if chunk_length > max_json_bytes:
                    raise VRMInspectionError("GLB JSON chunk exceeds configured limit")
                raw = fh.read(chunk_length)
                if len(raw) != chunk_length:
                    raise VRMInspectionError("truncated GLB JSON chunk")
                sha.update(raw)
                try:
                    json_obj = json.loads(raw.rstrip(b"\x00 \t\r\n"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise VRMInspectionError("invalid GLB JSON chunk") from exc
                remaining = 0
            while remaining:
                block = fh.read(min(1024 * 1024, remaining))
                if not block:
                    raise VRMInspectionError("truncated GLB")
                sha.update(block)
                remaining -= len(block)
            consumed += chunk_length
        if consumed != total_length:
            raise VRMInspectionError("GLB length mismatch")
    if json_obj is None:
        raise VRMInspectionError("GLB JSON chunk not found")
    return _manifest_from_gltf(json_obj, sha.hexdigest())
