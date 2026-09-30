from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ai_character_engine.llm.gateway import GatewayRequest, ModelEndpoint

from .errors import MissingCognitiveRolePolicyError, NoEligibleCognitiveModelError
from .models import (
    CognitiveOptimization,
    CognitiveRole,
    CognitiveRolePolicy,
    CognitiveRouteDecision,
    CognitiveRouteRequirements,
)


@dataclass(frozen=True, slots=True)
class CognitiveModelRouter:
    """Deterministic task-role -> endpoint planner.

    It only chooses an ordered endpoint chain. Provider calls, circuit breaking and
    fallback execution remain the responsibility of ``ModelGatewayClient``.
    """

    policies: Mapping[CognitiveRole, CognitiveRolePolicy]
    default_policy: CognitiveRolePolicy | None = None

    def policy_for(self, role: CognitiveRole) -> CognitiveRolePolicy:
        policy = self.policies.get(role)
        if policy is not None:
            return policy
        if self.default_policy is not None:
            return self.default_policy
        raise MissingCognitiveRolePolicyError(f"no cognitive routing policy for role {role.value!r}")

    def route(
        self,
        role: CognitiveRole,
        request: GatewayRequest,
        endpoints: Mapping[str, ModelEndpoint],
        *,
        requirements: CognitiveRouteRequirements | None = None,
    ) -> CognitiveRouteDecision:
        requirements = requirements or CognitiveRouteRequirements()
        policy = self.policy_for(role)
        optimization = requirements.optimization or policy.optimization
        required = set(policy.required_capabilities) | set(requirements.required_capabilities)
        if request.requires_tools:
            required.add("tools")
        if requirements.require_native_streaming:
            required.add("streaming")
        preferred_tags = set(policy.preferred_tags) | set(requirements.preferred_tags)

        max_cost = _tightest_max(policy.max_cost_tier, requirements.max_cost_tier)
        max_latency = _tightest_max(policy.max_latency_tier, requirements.max_latency_tier)
        min_quality = _tightest_min(policy.min_quality_tier, requirements.min_quality_tier)

        rejected: dict[str, str] = {}
        explicit = policy.explicit_chain
        candidate_ids = explicit or tuple(sorted(endpoints))
        eligible: list[ModelEndpoint] = []
        for endpoint_id in candidate_ids:
            endpoint = endpoints.get(endpoint_id)
            if endpoint is None:
                rejected[endpoint_id] = "unknown_endpoint"
                continue
            reason = _rejection_reason(
                endpoint,
                required_capabilities=required,
                max_cost_tier=max_cost,
                max_latency_tier=max_latency,
                min_quality_tier=min_quality,
            )
            if reason is not None:
                rejected[endpoint_id] = reason
                continue
            eligible.append(endpoint)

        if not eligible:
            raise NoEligibleCognitiveModelError(
                f"no eligible endpoint for cognitive role {role.value!r}; rejected={rejected}"
            )

        if explicit:
            ordered = tuple(endpoint.endpoint_id for endpoint in eligible)
        else:
            ordered = tuple(
                endpoint.endpoint_id
                for endpoint in sorted(
                    eligible,
                    key=lambda endpoint: _score(endpoint, optimization, preferred_tags),
                )
            )

        return CognitiveRouteDecision(
            role=role,
            route_name=f"cognitive:{role.value}:{optimization.value}",
            endpoint_ids=ordered,
            optimization=optimization,
            required_capabilities=frozenset(required),
            preferred_tags=frozenset(preferred_tags),
            rejected=rejected,
        )


def _tightest_max(a: int | None, b: int | None) -> int | None:
    values = [value for value in (a, b) if value is not None]
    return min(values) if values else None


def _tightest_min(a: int | None, b: int | None) -> int | None:
    values = [value for value in (a, b) if value is not None]
    return max(values) if values else None


def _rejection_reason(
    endpoint: ModelEndpoint,
    *,
    required_capabilities: set[str],
    max_cost_tier: int | None,
    max_latency_tier: int | None,
    min_quality_tier: int | None,
) -> str | None:
    if not endpoint.enabled:
        return "disabled"
    missing = sorted(required_capabilities.difference(endpoint.capabilities))
    if missing:
        return "missing_capabilities:" + ",".join(missing)
    if max_cost_tier is not None and endpoint.cost_tier > max_cost_tier:
        return "cost_tier_exceeded"
    if max_latency_tier is not None and endpoint.latency_tier > max_latency_tier:
        return "latency_tier_exceeded"
    if min_quality_tier is not None and endpoint.quality_tier < min_quality_tier:
        return "quality_tier_too_low"
    return None


def _score(
    endpoint: ModelEndpoint,
    optimization: CognitiveOptimization,
    preferred_tags: set[str],
) -> tuple[float, int, str]:
    tag_penalty = 0.0 if preferred_tags.issubset(endpoint.tags) else 100.0
    if optimization is CognitiveOptimization.LOW_LATENCY:
        primary = float(endpoint.latency_tier)
    elif optimization is CognitiveOptimization.LOW_COST:
        primary = float(endpoint.cost_tier)
    elif optimization is CognitiveOptimization.THROUGHPUT:
        primary = float(-endpoint.throughput_tier)
    elif optimization is CognitiveOptimization.HIGH_QUALITY:
        primary = float(-endpoint.quality_tier)
    else:
        primary = (
            float(endpoint.latency_tier)
            + float(endpoint.cost_tier)
            - float(endpoint.throughput_tier)
            - float(endpoint.quality_tier)
        ) / 4.0
    return (tag_penalty + primary, endpoint.priority, endpoint.endpoint_id)
