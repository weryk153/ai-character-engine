# Documentation

Use these references to install, integrate and operate AI Character Engine.

| Document | Use it for |
|---|---|
| [Agent skill](../skills/ai-character-engine/SKILL.md) | Let a coding agent install and integrate the SDK |
| [Getting started](getting-started.md) | Install the SDK, verify it and connect a model |
| [Configuration](configuration.md) | Choose dependencies, endpoints, stores and host settings |
| [Character companion](companion.md) | Talk to one character through a single object, with memory, state and background cognition wired |
| [Companion service for games](companion-service.md) | Run a game's NPCs as companions over HTTP and WebSocket, with saves and the game's clock |
| [Architecture](architecture.md) | Understand state ownership and component boundaries |
| [API reference](api-reference.md) | Browse all 374 stable root exports and their source definitions |
| [Extensions](extensions.md) | Integrate providers, plugins, renderers and host services |
| [Operations](operations.md) | Persistence, workers, telemetry, failure handling and shutdown |
| [Compatibility](compatibility.md) | API guarantees, contracts and persistence migrations |
| [Release process](release.md) | Build, validate and reproduce distribution evidence |

The [examples index](../examples/README.md) lists supported entry points and their
requirements. The [stable contract](public_api_v1_stable.json) is the
machine-readable source for the root API. Sealed manifests under `tests/fixtures/api` are
compatibility fixtures, not additional installation requirements.

Optional local audio integrations have separate [macOS host notes](mac_live_voice.md)
and [GPT-SoVITS configuration](gpt_sovits.md). These are host-specific facilities;
the core SDK remains independent of audio devices and renderers.

## Agent skill

Copy `skills/ai-character-engine` into your agent’s skills directory. For Codex,
use `$CODEX_HOME/skills` (or `~/.codex/skills` when `CODEX_HOME` is unset). Keep the
whole folder, including `references/` and `agents/`. If you already have a local
copy, compare it before replacing your own customizations.

Invoke `$ai-character-engine` with a concrete task, such as “connect this app to
my local model and save sessions.” The skill includes provider, tool, memory,
session and host guidance; it does not start model services or install the SDK
just by being added to the skills directory.
