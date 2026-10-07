---
name: ai-character-engine
description: Install and integrate the AI Character Engine Python SDK into an application. Use for its companion, model clients, character runtime, memory and sessions, tools, cognition, and host-neutral interfaces; not for general character writing or avatar artwork.
---

# AI Character Engine

Connect AI Character Engine to the user's project and deliver a character integration that runs. This skill targets SDK 1.3.0; check the installed version and the signatures before calling anything.

## Find the engine and the host

Tell the **host project** (the application that gets a character) apart from the **engine checkout** (where the API, examples and installation live). Integration code belongs in the host; UI, renderer and vendor features do not go into the core.

Prefer a path the user names, or the current workspace. When an engine checkout has to be found, look for a repository whose remote is `weryk153/ai-character-engine`. Below, `ENGINE_ROOT` is a verified checkout, not the directory this skill is installed in.

Use an existing checkout; do not clone again, and do not move or delete other versions. When the source is needed, clone `https://github.com/weryk153/ai-character-engine.git`. Day-to-day use is clone and pull; no ZIP is needed.

## Install and the first turn

1. Use the host's virtual environment. 1.3.0 is verified on Python 3.11–3.13; an existing Python 3.10 environment cannot be used as it is.
2. Run `python -m pip install "$ENGINE_ROOT"`; use an editable install only when the engine's own source has to change. Pick extras from `ENGINE_ROOT/docs/configuration.md` as needed; extras download no models and start no services.
3. Check `python -c "import ai_character_engine as ace; print(ace.__version__, ace.__file__)"` for the version and where it loads from.
4. Keep the model, endpoint, credentials and character the user named. When the model identifier is missing, read the existing configuration or ask; do not assume one is installed, and do not switch to a cloud service on your own.
5. Get two turns of one character working before adding anything. Keep one runtime per session; rebuilding it on every message loses the conversation.

A host that talks to one character — a chat window, a voice application, a desktop avatar — starts from `CharacterCompanion` (`ENGINE_ROOT/docs/companion.md`, `examples/companion_chat.py`): memory, mood, goals, reflection, interruption and persistence behind one `reply()`. Runnable model, tool and session code is in [integration](references/integration.md). Read [cognition and hosts](references/cognition-and-hosts.md) only for advanced needs.

## Pick the entry point by need

Paths are relative to `ENGINE_ROOT`. Read the relevant example first, not the whole API reference.

| Need | Start from | Note |
|---|---|---|
| One character for a host | `docs/companion.md`, `examples/companion_chat.py` | Turn hidden reasoning off for a local reasoning model, as the example does |
| Local model | `README.md`, `src/ai_character_engine/llm/local.py` | `examples/basic_chat.py` uses OpenAI Responses, not the local Chat Completions entry |
| Tools | `examples/tool_chat.py` | Register schema and handler together on `ToolRegistry` |
| Continue after a restart | `examples/session_runtime.py` | The example uses a temporary directory; the host provides a durable location and the session identity |
| Long-term memory | `tests/scenarios/long_term_memory.py`, `tests/scenarios/memory_revision.py` | History, session save and memory retrieval are different mechanisms |
| Beliefs and goals | `tests/scenarios/reflection_long_term_cognition.py`, `tests/scenarios/goal_motivation_runtime.py` | Managers, workers and commits have to be configured; plain chat starts no cognition |
| Background work, several models | `src/ai_character_engine/tasks/`, `tests/scenarios/cognitive_routing.py` | Model routing and worker scheduling are configured separately |
| Proactive interaction | `examples/autonomy_host.py` | The host provides timers, triggers, cancellation and shutdown |
| Web and mobile | `examples/character_service.py`, `docs/extensions.md` | An offline example; the default `NoopAuthHook` is not real authentication |
| Generic host, voice, renderer | `src/ai_character_engine/host/bridge.py`, `examples/README.md` | Connect through the neutral interfaces; text working is not UI or audio acceptance |

`tests/scenarios/` holds fixed offline outputs and hand-made evidence for understanding the interfaces and the expected behaviour; they are no evidence of what a real model can do.

## Integration boundaries

- Models, renderers, voice and hosts connect through adapters. The host owns input capture, display, playback, permissions and lifecycle.
- Installation and basic use depend on `ai_character_engine` alone. Use the generic interfaces first; look at and install an optional adapter package only when the user explicitly needs it. Dependencies point from host and adapter to core.
- Background workers read snapshots and produce proposals that the coordinator checks and commits; do not write character state or storage directly.
- Memory records events, reflections are interpretations, beliefs are supported hypotheses, goals are action intentions; none of them becomes a world fact or a tool permission on its own.
- Keep the scopes of user, character and session apart; do not share one undifferentiated runtime or memory file.
- An existing host keeps the branch and configuration the user named. Replace an agent, model or service only within the task; leave the other working pipelines alone.

## Verify and deliver

Verify the boundary that was actually connected: two turns of context, tool arguments and results, session restore after a rebuild, host input and output. Offline stand-ins verify the wiring; the quality of a real model is verified separately. A model saying "I remember" is no substitute for checking the data layer.

Run the tests that concern this integration. Follow `ENGINE_ROOT/docs/release.md` and its release gates only when the engine itself was changed and is being accepted for release; ordinary use of this skill makes no ZIP, tag or package.

When delivering, name the files changed, how to start, the model endpoint, where data is stored and what was actually tested. List plainly what could not be verified for lack of a model, hardware or credentials.
