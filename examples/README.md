# Application examples

Run these from an installed source checkout. They demonstrate integration entry
points; their credentials, stores and host policies must be configured explicitly.

| Entry point | Purpose | Requirements |
|---|---|---|
| `basic_chat.py` | Interactive Responses-backed character | API key and explicit model |
| `tool_chat.py` | Character with host-registered tools | API key and explicit model |
| `companion_chat.py` | Chat with a character that remembers and changes | Running OpenAI-compatible endpoint |
| `session_runtime.py` | Persist and restore session state | Offline; uses temporary stores |
| `character_service.py` | HTTP/SSE/WebSocket reference host | `service` extra; offline model |
| `autonomy_host.py` | Host scheduling and shutdown lifecycle | Offline |
| `providers/ollama_local.py` | Local Ollama connection | Running compatible endpoint |
| `providers/vllm_local.py` / `providers/vllm_remote.py` | Private inference host | Running endpoint and model |
| `providers/local_gateway_fallback.py` | Host-configured inference fallback | Configured inference endpoints |
| `integrations/live_voice_chat.py` | Optional local voice host | Audio/provider dependencies and model files |
| `integrations/hardware_acceptance.py` | Device-specific voice acceptance | Same host plus real devices |
| `integrations/sft_dataset.py` | Compile character training data | Offline sample dataset |
| `integrations/lora_training_plan.py` | Inspect an explicit training plan | Offline sample dataset |
| `integrations/lora_train_hf.py` | Run an opted-in training job | `training` extra, model resources |

Sample evaluation/training datasets are under `data/`. They do not establish
production model quality. Device launchers are under `tools/launchers`;
[macOS voice](../docs/mac_live_voice.md) and [GPT-SoVITS](../docs/gpt_sovits.md)
document their requirements.

Component-level regression scenarios are maintained under `tests/scenarios`.
Release collectors and fault/performance probes are under `tools/acceptance` and
are described in the [release process](../docs/release.md).
