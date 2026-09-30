# Changelog

## 1.0.0

- Stable 374-symbol root Python API, with explicit compatibility and deprecation contracts.
- Stateful character turns, scoped memory, persistence and deterministic schema migration.
- Bounded cognition, coordinated commits, multi-character interaction and world perception.
- Provider/host-neutral extension and distributed-worker contracts.
- Optional VRM adapter and service integrations.
- Apache-2.0 licensing across both synchronized distributions.
- Nine-platform/interpreter release evidence, isolated installs and deterministic acceptance gates.
- `CharacterCompanion`: the engine assembled for a host that talks to one character. Memory, mood, goals, reflection, conversations, interruption and persistence behind one `reply()`; a `ModelAccess` that lets the reply go first on a single local model; memories the host can show and edit; `aside()` for a model call of the host's own that gives way to the reply like her workers; `take_back()` for a reply the host did not use; `remember_remark()` for a remark she made on her own; `goals_shown` and `thoughts_shown` for what she keeps in mind.
- Context written into the conversation (`transcript` placement): only what is new since the last turn, kept beside the history, so a local model's prompt cache survives from turn to turn.
- Streaming with tools, a tool reminder in the turn context, a ready-made relationship state policy, and a vision provider usable with a local reasoning model.

Sealed compatibility fixtures preserve the behaviour of pre-1.0 versions; their
development history is not published. Release evidence is delivered with the
corresponding artifacts.
