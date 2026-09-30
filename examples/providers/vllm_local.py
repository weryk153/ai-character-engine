"""Minimal local vLLM deployment configuration."""
from ai_character_engine.llm import LocalDeploymentConfig

config = LocalDeploymentConfig.local_vllm(
    model="Qwen/Qwen3-8B",
    device="cuda",
    quantization="awq",
    context_length=32768,
)
print(config.public_metadata())
