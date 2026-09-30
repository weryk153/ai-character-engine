"""Remote/private vLLM configuration with secrets kept out of metadata."""
import os

from ai_character_engine.llm import LocalDeploymentConfig

config = LocalDeploymentConfig.remote_vllm(
    model="Qwen/Qwen3-32B",
    base_url=os.getenv("VLLM_BASE_URL", "https://gpu.example.internal/v1"),
    api_key=os.getenv("VLLM_API_KEY"),
    device="cuda",
    context_length=65536,
)
print(config.public_metadata())
