# Changelog

## 1.1.1

- `ContextBuilder` fades her mood by default (`mood_half_life_seconds` 300, `mood_floor` 0.15), so a bare `CharacterRuntime` tells the model the same faded mood the live face shows; before, the `- emotion:` line gave the mood as stored unless a companion set the builder. A host that builds `CharacterState(emotion=...)` directly must give it a `mood_intensity`, or the line reads neutral. The same holds for the other ways a bare runtime gets a mood: one set through a `StatePatch` without an intensity (0.5 by default) fades to neutral after about 8.7 minutes, and a snapshot saved before 1.1.0 restored into a bare runtime, or `state.emotion = ...` assigned directly, reads neutral. Setting `mood_half_life_seconds = None` still shows the mood as stored.
- The defaults are named once: `DEFAULT_MOOD_HALF_LIFE_SECONDS` and `DEFAULT_MOOD_FLOOR` in `ai_character_engine.state.mood`, used by every reader of her mood.
- `CharacterCompanion` wires a copy of a host's `RelationshipStatePolicy` (clock, mood settings, `mood_read_for`), so two companions sharing one instance no longer overwrite each other's; later changes to the policy go through `companion.runtime.state_policy`.
- Cognitive evaluation cases keep `mood_intensity` and `mood_updated_at` through `to_dict`/`from_dict`.
- `effective_mood` treats a half-life of 0 or less, or one that is not finite, as no fading instead of raising `ZeroDivisionError` (or amplifying her mood); this reaches `ContextBuilder` and `BackgroundCognitionRuntime`, whose attributes are not validated.
- `on_mood_change` is called even when saving the state fails after her mood changed.
- The `mood` worker reads an intensity written as a number in a string (`"0.6"`), clamped like a number, as it already did for confidence.
- An `async def` `on_mood_change` is scheduled on the running loop (it was never awaited); a failure is logged. `docs/companion.md` says the listener runs on the event loop under the turn lock and must be quick and non-blocking.
- `CompanionSnapshot` has `mood_floor` (last field, defaulted), so a host that fades the face itself stops where the engine does.
- A saved `mood_updated_at` later than the load time (another machine's clock, milliseconds read as seconds) is dated as of loading, so her mood fades.
- The `mood` worker's prompt says a line marked `Event` is something that happened, not said by anyone.

## 1.1.0

- Her mood is one of eight words (`CHARACTER_MOODS` in `ai_character_engine.companion`: neutral, happy, sad, angry, surprised, embarrassed, calm, worried) with an intensity and the time it was set, and it fades with time: the intensity halves every `mood_half_life_seconds` (300) and below `mood_floor` (0.15) she is neutral again. The `- emotion:` line of the note and `CompanionSnapshot.emotion` read it faded.
- A `mood` worker (`BackgroundCognitionKind.CHARACTER_MOOD`, role `CognitiveRole.MOOD`, commit target `state.mood_candidate`) reads both sides of the conversation, her lines under her name, and judges her mood as of her latest line, also from what she said herself; the latest exchange is set apart from the earlier conversation, which is background only. It is also given a short summary of who she is (`CharacterProfile.background`, falling back to `description` when background is empty, plus personality, about 400 characters) and judges her mood relative to it; its intensity is anchored (about 0.2 slight, 0.5 clear, 0.8 or more only for a major event) and small talk with no particular feeling is neutral. Evidence is at most 3 short quotes. Every `mood_every` (2) turns. The emotion worker still reads the user's lines only.
- The relationship rules name her mood in the new words: `sad` where they said `hurt`, `worried` where they said `concerned`, as strong as the user's emotion was. They change her mood only on a notable observation (sad, worried or happy) and no longer set `calm` on an unremarkable turn; such a turn leaves her mood as it was. In `CharacterCompanion` they move her mood only on turns the mood worker does not read (`relationship_patch(..., set_mood=False)`, `RelationshipStatePolicy.mood_read_for`); on the others the observation still moves trust and favorability. Nothing replaces a mood set for a later turn.
- Her mood holds until something at least as strong comes along: a reading of her mood, or the rules, weighed against her mood as it stands now. A neutral reading leaves her mood to fade by itself, the same mood again is as strong as the stronger of the two, and another mood takes over only when it is at least as strong. A reading that leaves her mood still stands for its turn.
- The `mood` worker's near words count as the mood they mean (`relieved` is calm, `annoyed` angry, `shy` embarrassed, and so on); any other word off the list is still no reading.
- `CompanionSnapshot` has `mood_intensity`, `mood_updated_at` and `mood_half_life_seconds`; `CharacterCompanion` takes `clock` and calls `on_mood_change` when a background result changed her mood. `CharacterState` and `StatePatch` carry `mood_intensity` and `mood_updated_at`; a state saved before has no mood yet.
- `EmotionExpressionPolicy` has a face for each of her moods (none for neutral) and weighs it by the intensity it is given.
- Hosts that set `CharacterState.emotion` directly must also set `mood_intensity` (and `mood_updated_at`), because a mood with intensity 0 now reads as neutral.
- `build_system_prompt` now leaves out the `Background:` section when `CharacterProfile.background` (whitespace-normalised) is already contained in `description` (whitespace-normalised), so a host whose `description` already holds the persona does not show it twice. Every host's reply prompt can be affected.

Known limitations:

- Small talk tends to read as a mild happy.
- A scolding is sometimes read as embarrassed.
- Surprised is under-used.
- On turns without a mood reading, the rules give only happy, sad or worried.

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
