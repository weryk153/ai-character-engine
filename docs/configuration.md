# Configuration

The host supplies dependencies, clients, stores, limits and lifecycle policy.
Core construction is explicit; there is no global cloud control plane or renderer
configuration that changes runtime authority.

## Environment used by sample hosts

| Variable | Consumer | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | Responses sample / provider SDK | Authentication for the chosen endpoint |
| `AI_CHARACTER_MODEL` | Interactive chat samples | Explicit model identifier |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama sample | Host-provided local inference endpoint and model |
| `VLLM_BASE_URL`, `VLLM_MODEL` | vLLM samples | Private inference endpoint and model |
| `VLLM_API_KEY` | Remote vLLM sample | Optional endpoint authentication |

`.env.example` is a template. The sample hosts load `.env`; embedding applications
can construct clients directly. Keep credentials and local endpoint settings out
of version control. The core does not read these variables to make autonomous
routing decisions.

## Optional dependency groups

| Extra | Purpose |
|---|---|
| `service` | FastAPI, HTTP streaming and WebSocket host |
| `local` | HTTP client support for local inference integrations |
| `qdrant` | Qdrant-backed retrieval integration |
| `voice-audio` | Audio I/O and HTTP voice integrations |
| `voice-sherpa` | Sherpa ONNX speech recognition integration |
| `voice-mac` | macOS speech host dependencies |
| `training` / `training-qlora` | Explicit offline model-training workflow |
| `dev` | Tests and development tools |

Training dependencies and audio providers are opt-in. They are not required to
run a text character or import the engine. A package extra installs dependencies;
it does not download models, start servers or certify hardware compatibility.

## Runtime choices

- Define identity and response constraints in `CharacterProfile`.
- Pass an `LLMClient` implementation to `CharacterRuntime`.
- Register permitted tools and schemas in `ToolRegistry`; handlers stay host-owned.
- Configure context, retrieval and state policies explicitly for the character.
- Assign stable session and memory scopes. Choose persistent store paths and
  backup/retention rules in the host.
- Set worker timeouts, concurrency and idempotency policy before dispatching work.

`OpenAICompatibleChatClient` exposes endpoint, timeout, retry and concurrency
settings. Retry policy must account for tool side effects; a successful remote
request and a successful local commit are separate events.
