# Changelog

## 1.1.0

- Her mood is one of eight words (`CHARACTER_MOODS` in `ai_character_engine.companion`: neutral, happy, sad, angry, surprised, embarrassed, calm, worried) with an intensity and the time it was set, and it fades with time: the intensity halves every `mood_half_life_seconds` (300) and below `mood_floor` (0.15) she is neutral again. The `- emotion:` line of the note and `CompanionSnapshot.emotion` read it faded.
- A `mood` worker (`BackgroundCognitionKind.CHARACTER_MOOD`, role `CognitiveRole.MOOD`, commit target `state.mood_candidate`) reads both sides of the conversation, her lines under her name, and judges her mood, also from what she said herself; every `mood_every` (2) turns. The emotion worker still reads the user's lines only.
- The relationship rules name her mood in the new words: `sad` where they said `hurt`, `worried` where they said `concerned`, as strong as the user's emotion was. A reading of her mood from the same turn stands; nothing replaces a mood set for a later turn.
- `CompanionSnapshot` has `mood_intensity`, `mood_updated_at` and `mood_half_life_seconds`; `CharacterCompanion` takes `clock` and calls `on_mood_change` when a background result changed her mood. `CharacterState` and `StatePatch` carry `mood_intensity` and `mood_updated_at`; a state saved before has no mood yet.
- `EmotionExpressionPolicy` has a face for each of her moods (none for neutral) and weighs it by the intensity it is given.
- Hosts that set `CharacterState.emotion` directly must also set `mood_intensity` (and `mood_updated_at`), because a mood with intensity 0 now reads as neutral.

## 1.0.0

- Stable 374-symbol root Python API, with explicit compatibility and deprecation contracts.
- Stateful character turns, scoped memory, persistence and deterministic schema migration.
- Bounded cognition, coordinated commits, multi-character interaction and world perception.
- Provider/host-neutral extension and distributed-worker contracts.
- Optional VRM adapter and service integrations.
- Apache-2.0 licensing across both synchronized distributions.
- Nine-platform/interpreter release evidence, isolated installs and deterministic acceptance gates.
- `CharacterCompanion`: the engine assembled for a host that talks to one character. Memory, mood, goals, reflection, conversations, interruption and persistence behind one `reply()`; a `ModelAccess` that lets the reply go first on a single local model; memories the host can show and edit; `aside()` for a model call of the host's own that gives way to the reply like her workers; `take_back()` for a reply the host did not use; `speak_up()` for speaking up on her own, from what she wants and thinks; `remember_remark()` for a remark a host made her say; `goals_shown` and `thoughts_shown` for what she keeps in mind.
- Context written into the conversation (`transcript` placement): only what is new since the last turn, kept beside the history, so a local model's prompt cache survives from turn to turn.
- Streaming with tools, a tool reminder in the turn context, a ready-made relationship state policy, and a vision provider usable with a local reasoning model.
- `CharacterCompanion` remembers what she said about herself, for every conversation: a `self_memory` worker (`BackgroundCognitionKind.SELF_MEMORY_EXTRACTION`, role `CognitiveRole.SELF_MEMORY`, commit target `memory.self_candidate`) reads her own lines and keeps what she stated about herself, each with an exact quote of hers; said again, it replaces the one she held. They stand in the notes as `- you said about yourself: ...`, and are taken out again once forgotten. `self_memories()` and `rewrite_self_memories()` show and edit them, and `rewrite_self_memories(..., from_before=True)` brings in memories from before the engine kept them as her oldest; settings `self_memory_every` (2), `self_memories_kept` (40) and `self_memories_shown` (12).
- Background workers read each conversation on its own: `BackgroundCognitionRuntime.conversation` names the conversation of the next turn, and cadence then counts that conversation's turns. Before, two conversations taken in turns with a worker every second turn left one of them unread. `CharacterCompanion` sets it.
- What `CharacterCompanion` passes on is checked in the engine, so that a host needs no filter of its own: sentences of nothing but punctuation, and assistant-style support closings (`ASSISTANT_SPEAK`, among the last two sentences and outside quotes), are left out of replies and remarks; a remark that quotes her latest remarks back counts as repeating her, and one that only acknowledges (`ACKNOWLEDGEMENTS`) is asked again like a repetition. `speak_up(..., statement_only=True)` leaves out questions.
- `ai_character_engine.companion` exports `SELF_MEMORY_LINE`, `ASSISTANT_SPEAK` and `ACKNOWLEDGEMENTS`, for a host that tells what the engine offers before relying on it.
- When a reply or remark is asked again, the reason is a note for that attempt only. It was added to the user's message, which the conversation kept and memory extraction read as the user's words.

Sealed compatibility fixtures preserve the behaviour of pre-1.0 versions; their
development history is not published. Release evidence is delivered with the
corresponding artifacts.
