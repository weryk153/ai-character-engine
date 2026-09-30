# Cognition and hosts

Pick the section the task needs. Paths are relative to `ENGINE_ROOT`.

## Reflection, beliefs and goals

Plain text chat does not start the cognition pipeline by itself. Check that `long_term_cognition`, `goal_manager`, the workers and the commit coordinator are connected; `CharacterCompanion` connects them for a host with one character.

- `LongTermCognitionManager` holds reflections and beliefs. Belief consolidation needs at least two independent sources by default; generating the same event several times does not add sources.
- Claims with enough support that contradict each other become contested and leave the ordinary active-belief context; the newest answer does not overwrite the old one.
- `GoalManager` keeps goals and motivation and manages the active, blocked, paused, completed and retired states; a goal grants no tool permission.
- `CognitiveCommitCoordinator` handles `TaskProposal`s. Workers read a read-only snapshot; provenance, confidence, age and revision, and policy are checked before a commit. A kind with no configured manager may need review rather than a direct write.

Read `tests/scenarios/reflection_long_term_cognition.py` and `tests/scenarios/goal_motivation_runtime.py` first, then `src/ai_character_engine/commit/`. The examples build sources by hand to exercise the mechanism; real source ids and provenance must point at actual events, never copied example values used as support.

## Foreground, background and the division of models

`CharacterRuntime` handles the foreground turn; `MultiTaskRuntime` manages background work. Model routing for cognitive roles and worker scheduling are configured separately. Roles can share one model; a demonstration does not require several.

Background tasks have limits on concurrency, timeouts and cancellation; the host manages start and shutdown. The foreground not queuing behind the background does not mean the same GPU has no contention; `CharacterCompanion` makes the background give way to the reply. Streaming, fallback timing and cancellation are verified on the chosen provider.

Starting points: `tests/scenarios/cognitive_routing.py`, `tests/scenarios/specialist_collaboration.py`, `docs/architecture.md`. A specialist's output is advice; it does not become a state write or a tool side effect on its own.

## Proactive interaction

`examples/autonomy_host.py` shows timers, idle detection, `AutonomyScheduler`, `AutonomyController`, cancellation when the user speaks, and cleanup on shutdown. Set the triggers and cooldowns the product needs, then connect the host's lifecycle.

Creating a goal schedules no reminder. When needed, verify that tasks do not overlap, expired candidates are removed, a user message takes the turn over, and nothing is scheduled after shutdown.

## A generic host

Read the host's agent, input format, output callbacks, Python version and model configuration first, and keep its working UI, audio and character-model pipelines.

The entry is `CharacterHostBridge`, imported from `ai_character_engine.host`. Read `process` and the streaming interface in `src/ai_character_engine/host/bridge.py`, then convert the host's input, output and callbacks on the host side. This path needs no product compatibility package and replaces no agent of the host by itself.

A trial connection traces at least the host's input to the runtime and the output to sentence splitting and presentation. Restoring history and relationships is verified separately; subtitles or expressions parsing does not mean that TTS speaks or that the microphone and the whole UI are accepted.

## Voice and renderers

Read `examples/README.md` and `docs/extensions.md` as needed, and tell input devices, ASR, model, TTS, playback and avatar mapping apart. Connect through the generic voice and avatar interfaces first, then read the documentation of the provider or renderer the user chose; do not install a particular implementation by default.

An extra installs Python dependencies only; model files, services, device permissions and renderer assets are configured separately. Report only what was measured of offline use, full duplex, echo, latency and hardware; the existence of an interface proves nothing. Keep the local or remote endpoint the user named.

## Services and several characters

`examples/character_service.py` is the HTTP, SSE and WebSocket entry. Install the `service` extra and start it as `docs/extensions.md` says. The example uses an offline client; a real service connects a real model and an authentication and admission policy. `NoopAuthHook` and CORS are not a login.

Each character has its own state, cognition and scope. `WorldRuntime` keeps world facts, and a character receives only the observations its perception policy allows; a public event is not delivered to every character by itself. See `tests/scenarios/multi_character_runtime.py` and `tests/scenarios/world_environment.py`, and verify that two characters know different things.
