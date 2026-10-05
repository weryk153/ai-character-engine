from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class CognitiveRole(str, Enum):
    """Stable semantic roles used by character cognition.

    Roles describe *why* inference is being requested, not which provider/model
    should serve it. Hosts may bind several roles to one model or spread them
    across many models without changing task code.
    """

    DIALOGUE = "dialogue"
    MEMORY = "memory"
    VISION = "vision"
    EMOTION = "emotion"
    SUMMARY = "summary"
    REFLECTION = "reflection"
    GOAL = "goal"
    PLANNER = "planner"
    TOOL = "tool"
    VERIFIER = "verifier"
    # What the character said about herself; a role of its own so that a host
    # can give it a model apart from memory of the user.
    SELF_MEMORY = "self_memory"
    # The character's own mood, read from both sides of the conversation; a
    # role of its own so that a host can give it a model apart from the
    # emotion of the user.
    MOOD = "mood"


class CognitiveOptimization(str, Enum):
    BALANCED = "balanced"
    LOW_LATENCY = "low_latency"
    LOW_COST = "low_cost"
    THROUGHPUT = "throughput"
    HIGH_QUALITY = "high_quality"


@dataclass(frozen=True, slots=True)
class CognitiveRouteRequirements:
    """Per-request constraints layered on top of one role policy."""

    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    preferred_tags: frozenset[str] = field(default_factory=frozenset)
    optimization: CognitiveOptimization | None = None
    require_native_streaming: bool = False
    max_cost_tier: int | None = None
    max_latency_tier: int | None = None
    min_quality_tier: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cost_tier", self.max_cost_tier),
            ("max_latency_tier", self.max_latency_tier),
            ("min_quality_tier", self.min_quality_tier),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when provided")


@dataclass(frozen=True, slots=True)
class CognitiveRolePolicy:
    """Routing policy for one cognitive role.

    Explicit primary/fallback ids form a deterministic chain. If both are empty,
    all configured endpoints are ranked using ``optimization`` and endpoint hints.
    Capability filters still apply to explicit chains, so a configured model can be
    safely skipped when a request needs tools/vision/streaming it cannot provide.
    """

    primary_endpoint_ids: tuple[str, ...] = ()
    fallback_endpoint_ids: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"chat"}))
    preferred_tags: frozenset[str] = field(default_factory=frozenset)
    optimization: CognitiveOptimization = CognitiveOptimization.BALANCED
    max_cost_tier: int | None = None
    max_latency_tier: int | None = None
    min_quality_tier: int | None = None

    def __post_init__(self) -> None:
        combined = self.primary_endpoint_ids + self.fallback_endpoint_ids
        if len(set(combined)) != len(combined):
            raise ValueError("cognitive role endpoint ids must be unique across primary/fallback chains")
        for endpoint_id in combined:
            if not endpoint_id.strip():
                raise ValueError("cognitive role endpoint ids must not be empty")
        for name, value in (
            ("max_cost_tier", self.max_cost_tier),
            ("max_latency_tier", self.max_latency_tier),
            ("min_quality_tier", self.min_quality_tier),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when provided")

    @property
    def explicit_chain(self) -> tuple[str, ...]:
        return self.primary_endpoint_ids + self.fallback_endpoint_ids


@dataclass(frozen=True, slots=True)
class CognitiveRouteDecision:
    role: CognitiveRole
    route_name: str
    endpoint_ids: tuple[str, ...]
    optimization: CognitiveOptimization
    required_capabilities: frozenset[str]
    preferred_tags: frozenset[str]
    rejected: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint_ids:
            raise ValueError("cognitive route decision requires at least one endpoint")
        object.__setattr__(self, "rejected", MappingProxyType(dict(self.rejected)))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "route_name": self.route_name,
            "endpoint_ids": list(self.endpoint_ids),
            "optimization": self.optimization.value,
            "required_capabilities": sorted(self.required_capabilities),
            "preferred_tags": sorted(self.preferred_tags),
            "rejected": dict(self.rejected),
        }
