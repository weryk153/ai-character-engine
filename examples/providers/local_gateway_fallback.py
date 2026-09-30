"""Offline demo: failed local readiness disables the endpoint and gateway uses fallback."""
import asyncio

from ai_character_engine.llm import (
    DeploymentHealthChecker,
    LLMResponse,
    LocalDeploymentConfig,
    Message,
    ModelEndpoint,
    ModelGatewayClient,
    StaticModelRouter,
    build_validated_model_endpoint,
)


class DownProbe:
    async def get_json(self, url, *, headers=None, timeout_seconds):
        raise ConnectionError("local runtime is offline")


class CloudFallback:
    async def generate(self, messages, *, tools=None):
        return LLMResponse(text="fallback response", model="cloud-demo")


async def main():
    config = LocalDeploymentConfig.local_vllm(model="Qwen/Qwen3-8B")
    local, status = await build_validated_model_endpoint(
        endpoint_id="local-vllm",
        config=config,
        checker=DeploymentHealthChecker(DownProbe()),
        client=CloudFallback(),  # never called because endpoint is disabled
    )
    gateway = ModelGatewayClient(
        endpoints=(local, ModelEndpoint("cloud", CloudFallback())),
        router=StaticModelRouter("local-vllm", "cloud", route_name="local-first"),
    )
    result = await gateway.generate([Message(role="user", content="hello")])
    print("local ready:", status.ready)
    print("answer:", result.text)
    print("gateway:", result.metadata["gateway"])


if __name__ == "__main__":
    asyncio.run(main())
