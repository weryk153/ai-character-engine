from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from ai_character_engine.llm.base import LLMClient
from ai_character_engine.llm.gateway import ModelEndpoint
from ai_character_engine.llm.local import (
    InferenceRuntimeMetadata,
    OllamaClient,
    OpenAICompatibleChatClient,
    VLLMClient,
    safe_base_url,
)

DeploymentProfile = Literal[
    "local_ollama",
    "local_vllm",
    "remote_vllm",
    "cloud_openai_compatible",
]


@dataclass(frozen=True, slots=True)
class LocalDeploymentConfig:
    profile: DeploymentProfile
    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"chat", "tools"}))
    timeout_seconds: float | None = 60.0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.25
    max_concurrency: int = 4
    health_timeout_seconds: float = 3.0
    require_model_available: bool = True
    device: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0 when set")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.health_timeout_seconds <= 0:
            raise ValueError("health_timeout_seconds must be > 0")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be > 0 when set")

    @classmethod
    def local_ollama(cls, *, model: str, **kwargs: Any) -> "LocalDeploymentConfig":
        return cls(
            profile="local_ollama",
            model=model,
            base_url=kwargs.pop("base_url", "http://127.0.0.1:11434/v1"),
            api_key=kwargs.pop("api_key", "ollama"),
            **kwargs,
        )

    @classmethod
    def local_vllm(cls, *, model: str, **kwargs: Any) -> "LocalDeploymentConfig":
        return cls(
            profile="local_vllm",
            model=model,
            base_url=kwargs.pop("base_url", "http://127.0.0.1:8000/v1"),
            **kwargs,
        )

    @classmethod
    def remote_vllm(cls, *, model: str, base_url: str, **kwargs: Any) -> "LocalDeploymentConfig":
        return cls(profile="remote_vllm", model=model, base_url=base_url, **kwargs)

    @classmethod
    def cloud_openai_compatible(
        cls, *, model: str, base_url: str, **kwargs: Any
    ) -> "LocalDeploymentConfig":
        return cls(profile="cloud_openai_compatible", model=model, base_url=base_url, **kwargs)

    @property
    def backend(self) -> str:
        if self.profile == "local_ollama":
            return "ollama"
        if self.profile in {"local_vllm", "remote_vllm"}:
            return "vllm"
        return "openai_compatible"

    @property
    def is_local(self) -> bool:
        return self.profile in {"local_ollama", "local_vllm"}

    def public_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "backend": self.backend,
            "model": self.model,
            "base_url": safe_base_url(self.base_url),
            "device": self.device,
            "quantization": self.quantization,
            "context_length": self.context_length,
            **self.runtime_metadata,
        }


class ProbeTransport(Protocol):
    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class HttpxProbeTransport:
    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # optional deployment dependency
            raise RuntimeError("health probes require the optional 'httpx' package") from exc
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("health endpoint returned a non-object JSON value")
            return value


@dataclass(frozen=True, slots=True)
class DeploymentStatus:
    profile: DeploymentProfile
    backend: str
    model: str
    base_url: str
    reachable: bool
    ready: bool
    model_available: bool | None
    available_models: tuple[str, ...] = ()
    latency_ms: float | None = None
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "reachable": self.reachable,
            "ready": self.ready,
            "model_available": self.model_available,
            "available_models": list(self.available_models),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class DeploymentHealthChecker:
    def __init__(self, transport: ProbeTransport | None = None) -> None:
        self.transport = transport or HttpxProbeTransport()

    async def check(self, config: LocalDeploymentConfig) -> DeploymentStatus:
        started = time.perf_counter()
        try:
            if config.profile == "local_ollama":
                url = self._ollama_tags_url(config.base_url)
                payload = await self.transport.get_json(
                    url, timeout_seconds=config.health_timeout_seconds
                )
                models = tuple(
                    str(item.get("name") or item.get("model"))
                    for item in payload.get("models", [])
                    if isinstance(item, dict) and (item.get("name") or item.get("model"))
                )
            else:
                url = config.base_url.rstrip("/") + "/models"
                headers = (
                    {"Authorization": f"Bearer {config.api_key}"}
                    if config.api_key
                    else None
                )
                payload = await self.transport.get_json(
                    url,
                    headers=headers,
                    timeout_seconds=config.health_timeout_seconds,
                )
                models = tuple(
                    str(item.get("id"))
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                )
        except Exception as exc:
            return DeploymentStatus(
                profile=config.profile,
                backend=config.backend,
                model=config.model,
                base_url=safe_base_url(config.base_url),
                reachable=False,
                ready=False,
                model_available=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        model_available = config.model in models if models else None
        ready = True
        if config.require_model_available:
            ready = model_available is True
        return DeploymentStatus(
            profile=config.profile,
            backend=config.backend,
            model=config.model,
            base_url=safe_base_url(config.base_url),
            reachable=True,
            ready=ready,
            model_available=model_available,
            available_models=models,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _ollama_tags_url(base_url: str) -> str:
        parts = urlsplit(base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, path + "/api/tags", "", ""))


def build_deployment_client(
    config: LocalDeploymentConfig,
    *,
    client: Any | None = None,
) -> LLMClient:
    runtime = InferenceRuntimeMetadata(
        backend=config.backend,
        device=config.device,
        quantization=config.quantization,
        context_length=config.context_length,
        extra={"profile": config.profile, **config.runtime_metadata},
    )
    kwargs = {
        "model": config.model,
        "base_url": config.base_url,
        "api_key": config.api_key,
        "client": client,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "max_concurrency": config.max_concurrency,
        "runtime_metadata": runtime,
    }
    if config.profile == "local_ollama":
        return OllamaClient(**kwargs)
    if config.profile in {"local_vllm", "remote_vllm"}:
        return VLLMClient(**kwargs)
    return OpenAICompatibleChatClient(
        backend="openai_compatible",
        **kwargs,
    )


async def build_validated_model_endpoint(
    *,
    endpoint_id: str,
    config: LocalDeploymentConfig,
    checker: DeploymentHealthChecker | None = None,
    client: LLMClient | None = None,
    tags: frozenset[str] = frozenset(),
    cost_tier: int = 0,
    latency_tier: int = 1,
    throughput_tier: int = 1,
    quality_tier: int = 1,
    priority: int = 100,
) -> tuple[ModelEndpoint, DeploymentStatus]:
    """Build an endpoint that becomes disabled when startup validation fails.

    Place the returned endpoint before a cloud endpoint in ModelGatewayClient to
    get graceful local->remote fallback without special runtime logic.
    """

    checker = checker or DeploymentHealthChecker()
    status = await checker.check(config)
    endpoint = ModelEndpoint(
        endpoint_id=endpoint_id,
        client=client or build_deployment_client(config),
        capabilities=config.capabilities,
        tags=frozenset({config.profile, config.backend, *tags}),
        timeout_seconds=config.timeout_seconds,
        enabled=status.ready,
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        throughput_tier=throughput_tier,
        quality_tier=quality_tier,
        priority=priority,
    )
    return endpoint, status


def deployment_llm_factory(config: LocalDeploymentConfig):
    """Return a SessionRuntimeFactory-compatible LLM factory."""

    def factory(_record: Any) -> LLMClient:
        return build_deployment_client(config)

    return factory


def deployment_readiness_probe(
    config: LocalDeploymentConfig,
    *,
    checker: DeploymentHealthChecker | None = None,
):
    checker = checker or DeploymentHealthChecker()

    async def probe() -> dict[str, Any]:
        status = await checker.check(config)
        payload = status.to_metadata()
        payload["ready"] = status.ready
        return payload

    return probe
