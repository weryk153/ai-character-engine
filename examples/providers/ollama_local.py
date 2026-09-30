"""Minimal Ollama deployment configuration.

Install project dependencies and pass this config to deployment_llm_factory() or
build_deployment_client() when a running Ollama server is available.
"""
from ai_character_engine.llm import LocalDeploymentConfig

config = LocalDeploymentConfig.local_ollama(
    model="qwen3:8b",
    device="apple_silicon",
    quantization="q4",
    context_length=32768,
)
print(config.public_metadata())
