from __future__ import annotations

import asyncio
import math
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ai_character_engine.llm.base import LLMClient
from ai_character_engine.llm.models import LLMResponse, Message
from ai_character_engine.tools.models import ToolDefinition

InferenceProfileName = Literal["low_latency", "balanced", "throughput", "low_cost"]


class InferenceBackpressureError(RuntimeError):
    """Raised when an inference queue has reached its configured capacity."""


@dataclass(frozen=True, slots=True)
class InferenceCapabilities:
    """Provider-neutral capability declaration for an inference backend.

    These flags describe what the backend *can* support. They do not enable the
    feature by themselves; the deployment/runtime still needs to be configured.
    """

    kv_cache: bool = True
    prefix_caching: bool = False
    continuous_batching: bool = False
    speculative_decoding: bool = False
    quantization: bool = False

    def unsupported_hints(self, hints: "InferenceTuningHints") -> tuple[str, ...]:
        unsupported: list[str] = []
        if hints.prefer_prefix_cache and not self.prefix_caching:
            unsupported.append("prefix_caching")
        if hints.prefer_continuous_batching and not self.continuous_batching:
            unsupported.append("continuous_batching")
        if hints.prefer_speculative_decoding and not self.speculative_decoding:
            unsupported.append("speculative_decoding")
        if hints.quantization is not None and not self.quantization:
            unsupported.append("quantization")
        return tuple(unsupported)

    def to_dict(self) -> dict[str, bool]:
        return {
            "kv_cache": self.kv_cache,
            "prefix_caching": self.prefix_caching,
            "continuous_batching": self.continuous_batching,
            "speculative_decoding": self.speculative_decoding,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class InferenceTuningHints:
    """Hints that a provider adapter may translate into backend-specific options."""

    prefer_prefix_cache: bool = False
    prefer_continuous_batching: bool = False
    prefer_speculative_decoding: bool = False
    quantization: str | None = None
    max_batch_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefer_prefix_cache": self.prefer_prefix_cache,
            "prefer_continuous_batching": self.prefer_continuous_batching,
            "prefer_speculative_decoding": self.prefer_speculative_decoding,
            "quantization": self.quantization,
            "max_batch_tokens": self.max_batch_tokens,
        }


@dataclass(frozen=True, slots=True)
class InferencePolicy:
    profile: InferenceProfileName = "balanced"
    max_concurrency: int = 4
    max_queue_size: int = 32
    request_timeout_seconds: float | None = 60.0
    max_output_tokens: int | None = None
    context_budget_tokens: int | None = None
    hints: InferenceTuningHints = field(default_factory=InferenceTuningHints)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        if self.request_timeout_seconds is not None and self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0 when set")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1 when set")
        if self.context_budget_tokens is not None and self.context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be >= 1 when set")

    @classmethod
    def low_latency(cls, **kwargs: Any) -> "InferencePolicy":
        return cls(profile="low_latency", max_concurrency=kwargs.pop("max_concurrency", 8), **kwargs)

    @classmethod
    def balanced(cls, **kwargs: Any) -> "InferencePolicy":
        return cls(profile="balanced", **kwargs)

    @classmethod
    def throughput(cls, **kwargs: Any) -> "InferencePolicy":
        return cls(
            profile="throughput",
            max_concurrency=kwargs.pop("max_concurrency", 16),
            max_queue_size=kwargs.pop("max_queue_size", 128),
            hints=kwargs.pop(
                "hints",
                InferenceTuningHints(prefer_continuous_batching=True, max_batch_tokens=8192),
            ),
            **kwargs,
        )

    @classmethod
    def low_cost(cls, **kwargs: Any) -> "InferencePolicy":
        return cls(profile="low_cost", max_concurrency=kwargs.pop("max_concurrency", 4), **kwargs)


@dataclass(frozen=True, slots=True)
class TokenCostModel:
    """Simple token/request cost model.

    Prices are in arbitrary currency units; callers typically use USD.
    Local deployments may use only flat_request_cost to represent an amortized
    per-request estimate without pretending local inference has token billing.
    """

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    flat_request_cost: float = 0.0
    label: str = "unpriced"

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("flat_request_cost", self.flat_request_cost),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def cloud(
        cls,
        *,
        input_per_million: float,
        output_per_million: float,
        label: str = "cloud",
    ) -> "TokenCostModel":
        return cls(
            input_per_million=input_per_million,
            output_per_million=output_per_million,
            label=label,
        )

    @classmethod
    def local_amortized(
        cls,
        *,
        flat_request_cost: float,
        label: str = "local_amortized",
    ) -> "TokenCostModel":
        return cls(flat_request_cost=flat_request_cost, label=label)

    def estimate(self, *, input_tokens: int | None, output_tokens: int | None) -> float:
        return (
            self.flat_request_cost
            + ((input_tokens or 0) / 1_000_000) * self.input_per_million
            + ((output_tokens or 0) / 1_000_000) * self.output_per_million
        )


@dataclass(frozen=True, slots=True)
class InferenceMeasurement:
    profile: InferenceProfileName
    queue_wait_ms: float
    request_elapsed_ms: float
    ttft_ms: float | None
    decode_tokens_per_second: float | None
    estimated_cost: float
    cost_label: str
    capabilities: InferenceCapabilities
    hints: InferenceTuningHints
    unsupported_hints: tuple[str, ...] = ()
    max_output_tokens: int | None = None
    context_budget_tokens: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "queue_wait_ms": self.queue_wait_ms,
            "request_elapsed_ms": self.request_elapsed_ms,
            "ttft_ms": self.ttft_ms,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "estimated_cost": self.estimated_cost,
            "cost_label": self.cost_label,
            "capabilities": self.capabilities.to_dict(),
            "tuning_hints": self.hints.to_dict(),
            "unsupported_hints": list(self.unsupported_hints),
            "max_output_tokens": self.max_output_tokens,
            "context_budget_tokens": self.context_budget_tokens,
        }


class ConcurrencyLimiter:
    """Async concurrency limiter with bounded waiting queue and queue telemetry."""

    def __init__(self, *, max_concurrency: int, max_queue_size: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._state_lock = asyncio.Lock()
        self._active = 0
        self._waiting = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def slot(self):
        queued_at = time.perf_counter()
        acquired_immediately = False
        acquired = False
        waiting_registered = False
        async with self._state_lock:
            if self._active < self.max_concurrency:
                acquired_immediately = True
            elif self._waiting >= self.max_queue_size:
                raise InferenceBackpressureError(
                    f"inference queue full: active={self._active}, waiting={self._waiting}, "
                    f"max_queue_size={self.max_queue_size}"
                )
            else:
                self._waiting += 1
                waiting_registered = True

        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._state_lock:
                if waiting_registered:
                    self._waiting -= 1
                    waiting_registered = False
                self._active += 1
            queue_wait_ms = (time.perf_counter() - queued_at) * 1000
            try:
                yield queue_wait_ms
            finally:
                async with self._state_lock:
                    self._active -= 1
                self._semaphore.release()
                acquired = False
        finally:
            if waiting_registered:
                async with self._state_lock:
                    self._waiting = max(0, self._waiting - 1)
            if acquired:
                # Defensive cleanup for cancellation between acquire and yield.
                async with self._state_lock:
                    self._active = max(0, self._active - 1)
                self._semaphore.release()



class InferenceOptimizedClient:
    """Provider-neutral wrapper adding queueing, timeout and inference telemetry."""

    def __init__(
        self,
        client: LLMClient,
        *,
        policy: InferencePolicy | None = None,
        capabilities: InferenceCapabilities | None = None,
        cost_model: TokenCostModel | None = None,
        limiter: ConcurrencyLimiter | None = None,
    ) -> None:
        self.client = client
        self.policy = policy or InferencePolicy()
        self.capabilities = capabilities or InferenceCapabilities()
        self.cost_model = cost_model or TokenCostModel()
        self.limiter = limiter or ConcurrencyLimiter(
            max_concurrency=self.policy.max_concurrency,
            max_queue_size=self.policy.max_queue_size,
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        async with self.limiter.slot() as queue_wait_ms:
            started = time.perf_counter()
            call = self.client.generate(messages, tools=tools)
            if self.policy.request_timeout_seconds is not None:
                response = await asyncio.wait_for(
                    call,
                    timeout=self.policy.request_timeout_seconds,
                )
            else:
                response = await call
            request_elapsed_ms = (time.perf_counter() - started) * 1000

        source_meta = dict(response.metadata)
        source_inference = source_meta.get("inference")
        ttft_ms: float | None = None
        if isinstance(source_inference, dict):
            raw_ttft = source_inference.get("ttft_ms")
            if isinstance(raw_ttft, (int, float)):
                ttft_ms = float(raw_ttft)
        if ttft_ms is None:
            raw_ttft = source_meta.get("ttft_ms")
            if isinstance(raw_ttft, (int, float)):
                ttft_ms = float(raw_ttft)

        decode_tps: float | None = None
        if response.output_tokens:
            if ttft_ms is not None and request_elapsed_ms > ttft_ms:
                decode_ms = request_elapsed_ms - ttft_ms
                decode_tps = response.output_tokens / (decode_ms / 1000)
            elif request_elapsed_ms > 0:
                # This is an end-to-end approximation when the provider does not
                # expose TTFT. Metadata makes this limitation inspectable.
                decode_tps = response.output_tokens / (request_elapsed_ms / 1000)

        estimated_cost = self.cost_model.estimate(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        measurement = InferenceMeasurement(
            profile=self.policy.profile,
            queue_wait_ms=queue_wait_ms,
            request_elapsed_ms=request_elapsed_ms,
            ttft_ms=ttft_ms,
            decode_tokens_per_second=decode_tps,
            estimated_cost=estimated_cost,
            cost_label=self.cost_model.label,
            capabilities=self.capabilities,
            hints=self.policy.hints,
            unsupported_hints=self.capabilities.unsupported_hints(self.policy.hints),
            max_output_tokens=self.policy.max_output_tokens,
            context_budget_tokens=self.policy.context_budget_tokens,
        )
        source_meta["inference"] = measurement.to_metadata()
        return LLMResponse(
            text=response.text,
            tool_calls=response.tool_calls,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms if response.latency_ms is not None else request_elapsed_ms,
            metadata=source_meta,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    request_id: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    request_id: str
    success: bool
    elapsed_ms: float
    queue_wait_ms: float | None = None
    ttft_ms: float | None = None
    decode_tokens_per_second: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    samples: tuple[BenchmarkSample, ...]
    elapsed_ms: float
    success_rate: float
    error_rate: float
    throughput_requests_per_second: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    avg_queue_wait_ms: float | None
    avg_ttft_ms: float | None
    avg_decode_tokens_per_second: float | None
    total_estimated_cost: float


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


class BenchmarkRunner:
    """Small provider-neutral async benchmark runner for repeatable local tests."""

    def __init__(self, client: LLMClient, *, concurrency: int = 4) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.client = client
        self.concurrency = concurrency

    async def run(self, requests: Sequence[BenchmarkRequest]) -> BenchmarkReport:
        gate = asyncio.Semaphore(self.concurrency)
        suite_started = time.perf_counter()

        async def execute(item: BenchmarkRequest) -> BenchmarkSample:
            async with gate:
                started = time.perf_counter()
                try:
                    response = await self.client.generate(
                        list(item.messages),
                        tools=list(item.tools) or None,
                    )
                except Exception as exc:
                    return BenchmarkSample(
                        request_id=item.request_id,
                        success=False,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                inference = response.metadata.get("inference") if response.metadata else None
                inference = inference if isinstance(inference, dict) else {}
                return BenchmarkSample(
                    request_id=item.request_id,
                    success=True,
                    elapsed_ms=elapsed_ms,
                    queue_wait_ms=_as_float(inference.get("queue_wait_ms")),
                    ttft_ms=_as_float(inference.get("ttft_ms")),
                    decode_tokens_per_second=_as_float(
                        inference.get("decode_tokens_per_second")
                    ),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    estimated_cost=_as_float(inference.get("estimated_cost")),
                )

        samples = tuple(await asyncio.gather(*(execute(item) for item in requests)))
        elapsed_ms = (time.perf_counter() - suite_started) * 1000
        successes = tuple(sample for sample in samples if sample.success)
        latencies = [sample.elapsed_ms for sample in successes]
        queue_waits = [sample.queue_wait_ms for sample in successes if sample.queue_wait_ms is not None]
        ttfts = [sample.ttft_ms for sample in successes if sample.ttft_ms is not None]
        decode_rates = [
            sample.decode_tokens_per_second
            for sample in successes
            if sample.decode_tokens_per_second is not None
        ]
        total = len(samples)
        success_rate = (len(successes) / total) if total else 0.0
        return BenchmarkReport(
            samples=samples,
            elapsed_ms=elapsed_ms,
            success_rate=success_rate,
            error_rate=1.0 - success_rate if total else 0.0,
            throughput_requests_per_second=(len(successes) / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            avg_queue_wait_ms=statistics.fmean(queue_waits) if queue_waits else None,
            avg_ttft_ms=statistics.fmean(ttfts) if ttfts else None,
            avg_decode_tokens_per_second=statistics.fmean(decode_rates) if decode_rates else None,
            total_estimated_cost=sum(sample.estimated_cost or 0.0 for sample in successes),
        )


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
