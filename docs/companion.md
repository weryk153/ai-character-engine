# Character companion

`CharacterCompanion` is the engine assembled for a host that talks to one
character: a chat window, a voice application, a desktop avatar. It gives the
host one object with a `reply()` method and takes care of the rest.

| The host gets | Without writing |
|---|---|
| A streamed reply, with tools | the wiring of runtime, host bridge and task runtime |
| A character whose mood, trust and relationship stage move | a state policy |
| Memory, goals and reflections that reach the next reply | background workers, commit rules, stores |
| Conversations that keep their own history and memory | scope and history switching |
| Interruption that keeps what was actually heard | cancellation and history repair |
| State that survives a restart | persistence |

Use the parts directly (see [architecture](architecture.md)) when the host needs
several characters in one world, its own commit policies or distributed workers.

## Use

```python
import asyncio

from ai_character_engine import CharacterProfile
from ai_character_engine.companion import CharacterCompanion
from ai_character_engine.llm.local import OpenAICompatibleChatClient


def model(temperature: float) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        model="qwen/qwen3.5-9b",
        base_url="http://127.0.0.1:1234/v1",
        request_options={
            "temperature": temperature,
            "max_tokens": 400,
            # A local reasoning model thinks before it answers, and the thought
            # alone can use up the output limit. LM Studio takes this switch;
            # Ollama takes {"think": False}, vLLM
            # {"chat_template_kwargs": {"enable_thinking": False}}.
            "extra_body": {"reasoning_effort": "none"},
        },
    )


async def main() -> None:
    companion = CharacterCompanion(
        character=CharacterProfile(
            id="guide", name="Guide", description="A patient, curious guide."
        ),
        llm=model(0.7),
        background_llm=model(0.2),
        storage_dir="companion-data/guide",
    )
    try:
        for text in ("Call me Alex.", "What name did I ask you to use?"):
            result = await companion.reply(
                text,
                conversation_id="first-chat",
                on_text_delta=lambda delta: print(delta, end="", flush=True),
            )
            print()
        await companion.settle()
        print(companion.snapshot())
        print(companion.memories("first-chat"))
    finally:
        await companion.close()


asyncio.run(main())
```

`background_llm` may be one client for every worker or a mapping from worker
name (`emotion`, `reply_check`, `mood`, `memory`, `self_memory`, `goal`,
`reflection`, `summary`) to a client. A worker without a client does not run,
except `reply_check`: in a mapping that names other workers but no
`reply_check` client it still reads her opening (see below), which needs no
model. `background_llm={}` turns background cognition off, the reply check
included. Background workers must return JSON, so give them a low temperature.

Without `storage_dir` nothing is written to disk.

## What happens in a turn

1. Background work gives way: calls in progress are abandoned and redone after
   the reply, so the reply has the model to itself.
2. The reply is generated and streamed. Text is forwarded as it is generated even
   when tools are registered (`stream_text_with_tools`), one sentence at a time.
   A sentence that repeats one of her latest lines or is nothing but
   punctuation is left out, and so is a closing that talks like an assistant
   offering help ("let me know if you need anything else", "如果還有其他問題..."):
   one of the last two sentences, in her own voice, not in quotes. Such a
   sentence waits until two more have come or the reply has ended. The
   conversation keeps what was passed on. A closing mark such as `」` stays
   with its sentence, and so does a `*` that closes an action left open; an
   emoji or other symbol is not punctuation. A reply of nothing but
   punctuation ("……") is her silence: nothing is passed on and the
   conversation keeps it as it was; an empty line is never kept. A reply of
   which nothing is left for repetition or assistant talk, and which used no
   tool, is asked again once.
3. Trust grows a little; an observation of the user's emotion committed since the
   last turn moves trust, favorability and, between readings of her mood, her
   mood (`ai_character_engine.state.relationship`; see Her mood below).
4. Background jobs for this turn are scheduled. The emotion of the turn is read
   first and alone; the others follow one at a time.
5. Results are committed when they arrive, also when turns have happened in
   between (up to `max_turns_late`; memory however late), unless a newer job of
   the same kind makes them obsolete.

After the user's emotion, and before her mood, the `reply_check` worker
(`BackgroundCognitionKind.REPLY_CHECK`, role `CognitiveRole.REPLY_CHECK`) reads
her reply back against who she is (`background`, or `description`, as for her
mood), her previous reply and what she replied to. It looks for five slips and
nothing else: `broke_character` (talking about prompts, notes, speech
recognition, a model or being an AI), `leaked_markup` (a tag or direction said
as words; `[joy]` keywords and `*actions*` are not), `off_persona` (contradicting
a fact of her persona), `repeated` (nearly the same opening or main sentence
as her previous reply) and `wrong_language` (the whole reply in another
language). Each slip quotes the sentence of her reply it is in (`evidence`);
`off_persona` and `repeated` also quote what they are held against (`against`,
from her persona or her previous reply). Each comes with a fix of at most 40
characters, written in `CompanionSettings.language` when the host names one,
else in the language of the user's latest line; a fix in another writing
(Chinese, Japanese, Korean or Latin letters) is dropped, and so is a kind off
the list or a quote she did not say in that reply. At most two are kept.

Her opening said again is not left to the model, which never reported it.
Once tags in square brackets, actions between asterisks or in parentheses and
quote marks are taken out, her reply repeats her previous one when it opens
with the same first clause, of four characters of Chinese, Japanese and the
like (「哈↗哈↘哈↗」 is one, 「哈哈哈」 is not) or three words, or with the same
first six such characters or three words. Then a `repeated` slip is made with
that opening, as she wrote it, as its evidence and a fixed fix in the language
above ("Do not open with the same words again.", 「開頭別再用同一句，換個起手。」,
and so on). It comes first and stands for any `repeated` the model reported.
It is made every turn the worker runs, also when the model call fails or a
mapping that names other workers has no `reply_check` client.

A small model calls nearly every reply a slip of some kind, so each
is held to what its words can show: `off_persona` quotes the fact of her
persona it contradicts and shares words with it; `repeated` quotes her previous
reply and is its opening again or a whole sentence again; `broke_character`
names a word of what runs behind the conversation (AI, model, prompt, 語音辨識,
システム and the like) that her persona does not; `leaked_markup` has markup
left once tags in square brackets and actions are taken out; `wrong_language` is written
in another script than both the user's line and her persona. `off_persona` and
`broke_character` are then asked about once more, one sentence and one
question, and kept only on a yes that names a word of her sentence (for
`off_persona`, one the fact does not hold). This is one more call for each
such slip; most turns make none.

What she said is never changed and her reply waits for nothing: the fixes go
into the note of her next turn, one line each
(`About your last reply: ...`), and only that turn. They are dropped when
another turn came first, when the next turn is in another conversation, or when
the reply was cut short or rewritten (`interrupt`, `replace_reply`, `take_back`)
in between. The commit target `context.reply_note_candidate` writes nothing
itself. It runs every `reply_check_every` turns (1; 0 turns it off).

The note's label is English like the engine's other notes, whatever language
she speaks. An expression keyword in square brackets (`[joy]`) is never a
`leaked_markup` slip, even said as words: a host that keeps such keywords in
her reply on purpose, for her to read the face she made, would otherwise get a
note on every reply that has one.

### Facts of the user that no longer agree

When a memory of the user is written, the earlier ones of its conversation on
the same topic are picked out without a model: they share a topic (work, home,
partner, school, pet, age, name), a name or words, two characters of Chinese or
Japanese, or alike embeddings; at most five (`conflict_candidates` in the
commit result of `memory.append_candidate`). Only when there are any is the
`memory_conflict` worker asked how the new fact stands to each, and it may answer
only for a fact it was shown, with one of three relations:

- `supersedes`: time moved on (a new job, a move, a break-up). The earlier fact
  is marked `superseded` (`superseded_by` the new one) and leaves her mind;
  nothing is asked.
- `contradicts`: both cannot be true and nothing tells which is later (another
  age, another name). Both stay; the new one has `conflict_with` in its
  metadata, and the note of her next turn in that conversation tells her, once,
  to ask the user which is right (`For the next reply only: 關於使用者：之前記得「…」，現在聽到「…」——…`,
  in the writing of `language`, else of the facts; English otherwise).
- `refines`: a detail added. Both stay as they are (`refines` in metadata).

A wrong `supersedes` erases a true fact, so it is held to the user's words:
each `supersedes` and `contradicts` is asked about once more, and either
stands only when both facts cannot be true at once. `supersedes` stands only
when, besides, the words the model copies from the user's line say a change
themselves (換工作, 搬到, 分手, 畢業, quit, moved...), and the earlier fact is one
fact on a topic the new one is about: 「住在台北，在台積電上班」 is two, and a job
change does not replace it. Otherwise it is a contradiction to ask about. A
plan (下個月要搬, going to) and one day against a habit (這週末沒去爬山 against
週末常去爬山) are neither. The coordinator applies these rules again to every
`memory.conflict_candidate`, whoever proposed it. A failed or unclear answer
changes nothing, and the memory itself is committed before the model is
asked; a failed search for candidates leaves `conflict_candidates` empty.

`memory_conflicts(conversation_id)` lists the contradictions not settled, as
`MemoryConflict` (the newer fact and the earlier one, with ids, the model's
reason and `relation`), for a memory page; `replaced_memories(conversation_id)`
lists the facts a newer one replaced (`relation="supersedes"`), and forgetting
the newer line with `rewrite_memories` brings the earlier one back; `resolve_conflict(keep_id,
conversation_id=)` keeps one and marks the other superseded. When the user
answers, the memory worker writes the answer as a new fact, and it settles
the contradiction itself if it is judged to replace one side. All of it is kept
on the memories, so it survives a restart, and a contradiction is asked about
once also across one. `memory_conflicts=False` turns it off. One client for
every worker serves it too; a host that passes a mapping adds a
`memory_conflict` key.

What the character knows, wants and thinks is written into the conversation as
notes that say only what is new, never into the system prompt, so that each
prompt extends the one before it and the inference server can reuse its work
(see [architecture](architecture.md)).

## What the host can change

| Call | Use it for |
|---|---|
| `reply(..., notes=[...])` | An instruction for this reply only, marked as such in the note of that turn |
| `reply(..., notes=["- ..."])` | What the host knows, one line each starting with `- `. Pass it on every turn: a line is written into the conversation once, and taken out again when it is no longer passed |
| `reply(..., remember_as=fn)` | Keep the reply the way the host shows it: `fn(text)` is what the conversation keeps |
| `reply(..., before_turn=fn)` | Rewrite the character or register tools once the turn has the companion to itself; for a host shared by several callers |
| `replace_reply(text)` | Change what is kept of the newest reply afterwards |
| `take_back(conversation_id)` | The host did not use the newest reply (it repeated her last one, say) and asks again: the reply and the words it answered leave the conversation. Only when that reply came from the conversation's last turn: after a turn that failed, the reply before it was heard whole and stays |
| `companion.character = profile` | Rewrite the persona between turns; the id cannot change |
| `companion.tools.register(definition, handler)` | Give the character a tool |
| `memory_conflicts(conversation_id)` / `resolve_conflict(keep_id, conversation_id=)` | Two facts of the user that cannot both be true and are not settled, and settling one as the user answered: the fact kept stays, the other leaves her mind |
| `replaced_memories(conversation_id)` | Facts of the user a newer one replaced; forgetting the newer line in `rewrite_memories` brings the earlier one back |
| `memories(conversation_id)` / `rewrite_memories(conversation_id, lines, edited_from=shown)` | Show what she remembers of a conversation, and take the user's edit back: a shown line that is gone is forgotten, a new line is remembered; what arrived while the page was open stays |
| `self_memories()` / `rewrite_self_memories(lines, edited_from=shown)` | What she said about herself, in any conversation, oldest first (`in_conversation=` for those in her mind in one conversation), and the user's edit of it, the same way as `rewrite_memories`. Beyond `self_memories_kept` the oldest are forgotten. For memories brought in from before the engine kept them, pass `from_before=True` (and `edited_from=[]` to only add): they are dated before every one she holds, so they are in mind last and forgotten first |
| `speak_up(conversation_id, notes=[...])` | She speaks up on her own; the host decides when. What she says comes from her: what is still open, what she wants and thinks, or turning to the user. `notes` suggest material for this remark only; `instruction` replaces the engine's own, for a host that asks in the language she speaks. What she says is generated whole and checked before any of it is passed on: the sentences a reply leaves out are left out, and so are sentences that quote back the words of her latest remarks. A remark that repeats her, or that only acknowledges ("OK.", "嗯。") when nobody said anything, is asked again, and after three attempts she stays quiet. `statement_only=True` leaves out questions as well, for a host whose user stayed quiet through her last questions. What she said stays in the conversation, the instruction does not; `keep=False` for a host that keeps what she said itself |
| `remember_remark(conversation_id, text)` | Keep a remark she made on her own when the turn that made it was kept out of memory (a long instruction the host does not want kept): what she said stays in the conversation after a short event, so she neither repeats it nor says it again when the user answers |
| `aside(make_call)` | A model call of the host's own on the same local model (a memory of its own to tidy, a translation): it waits while she replies, gives way once to a reply that starts, and takes its turn after her workers |

## Conversations

`conversation_id` selects the history and the memory scope: what she
remembers of the user belongs to the conversation it was said in.

Who she is belongs to the character and is the same in every conversation: her
state and mood, her long-term goals and beliefs, and what she said about her
tastes, traits, habits, history, relationships and opinions. What she is doing
stays in the conversation it is done in (`plans_stay_in_conversation`, on by
default): what she said she is working on or plans to do there, what she
thinks of the user there, her short-term goals and her thoughts. Each record
the memory, self-memory, goal and reflection workers keep is marked with the
conversation it came from (`metadata["conversation_id"]`); what she is doing
is not in her mind in another conversation, and back in its own it is again,
also after a restart. A short-term goal also leaves her mind after
`short_term_goal_max_age_hours` (24) untouched, in its own conversation too; a
long-term goal after `goal_max_age_days`. A host that names no conversation
has one, `None`; for it, what was kept before 1.2.0 is the difference (below).

The `self_memory` worker reads her own lines, never the user's, and keeps what
she stated about herself, each with a kind: `identity`, `trait`, `taste`,
`habit`, `history`, `relationship`, `opinion` are hers in every conversation;
`working_on`, `plan` and `view_of_user` (what she thinks of the user, "Mei
thinks the user ...") only in the conversation they were said in.
`SELF_MEMORY_KINDS` and `CONVERSATION_SELF_MEMORY_KINDS` in
`ai_character_engine.companion` list them. A kind the model names otherwise is
brought onto the list (`habits` is `habit`, `physical_trait` is `trait`,
`feeling` is `view_of_user`); one still off the list is hers in every
conversation. An `opinion`, `trait`, `habit`, `history` or `relationship`
whose summary names the user ("the user", "you", 用戶, 你, 對方, あなた,
사용자 …) is kept as `view_of_user` whatever the model called it: a small
model files its judgements of the user under those kinds however it is told,
and they would otherwise follow her into every conversation. Each item needs
an exact quote of hers. Said again, in the same words whatever the
punctuation, it replaces the one she held, also one held in another
conversation; anything said differently is a fact of its own, since
nearly the same words can say the opposite ("likes" and "dislikes"). What the
user never heard, the rest of a reply cut short by `interrupt()` or a reply
taken back before the worker's result came in, is not kept; where the
conversation no longer reaches back that far, nothing says it was not heard,
and it is kept. The newest `self_memories_shown` of those in her mind stand in
the conversation as `- you said about yourself: ...` lines of the note, and
the system prompt asks her to stay consistent with them. One she no longer
holds, edited away or pushed out by newer ones, is taken out of the notes
again. `self_memories_kept` is counted apart for what is hers everywhere and
for each conversation.

`self_memories()` lists all of them, wherever they were said, for a host's
memory page; `self_memories(in_conversation=...)` lists those in her mind in
that conversation. A line `rewrite_self_memories` gets in place of one that is
gone keeps the kind and the conversation of the line it replaced.
`snapshot().goals` and `.thoughts` are those of the conversation at hand.

What was kept before 1.2.0 names no conversation: a short-term goal, a thought
or a `working_on`/`plan`/`view_of_user` self memory of before counts as one of
another conversation and is no longer in her mind; `self_memories()` still
lists it. A self memory of before whose kind says it is hers everywhere
(`habit`, say) stays in mind whatever it says; the user removes it on the
memory page. `plans_stay_in_conversation=False` keeps everything in every
conversation, as before 1.2.0 (the `short_term_goal_max_age_hours` window still
applies).

What the workers keep about the user is the user's own words. A memory needs
an exact quote of the user; an item without one is dropped, and so is one
whose quote repeats words of hers in the transcript or in her reply (four
letters or digits or more, spacing and punctuation aside): a sentence she is
teaching, said after her, is no fact about the user. A thought's evidence
keeps only exact quotes of the user, each typed by the line it came from
(`CognitionEvidenceRef.evidence_type`); her own lines and retellings are
dropped, and a thought left with none is not proposed. The emotion worker is
given the user's earlier lines apart from the latest one and judges the latest
one only.

Known limits: `self_memories_kept` holds for each conversation, so what she
keeps in all of them together grows with the number of
conversations. A goal proposed again in another conversation is merged into
the one she has and moves to that conversation, out of the first. The
`short_term_goal_max_age_hours` window is wall time, not the companion's
`clock`: hours the host was not running count too. The check for her words
also reads her reply, so a fact the user states and she says back
(「今天很累」, then 「今天很累嗎？」), or an answer that repeats the words of
her question, can be dropped.

A host that keeps its own transcripts hands them over with
`load_conversation(conversation_id, messages)` before the first reply. A
conversation the companion has already seen is not replaced;
`has_conversation(conversation_id)` tells the host whether loading is needed.
The companion keeps the history of the `conversations_kept` most recent
conversations in memory and does not store transcripts on disk.

## Her mood

Her mood is one of eight words, `CHARACTER_MOODS` in
`ai_character_engine.companion`: neutral, happy, sad, angry, surprised,
embarrassed, calm, worried. With it come an intensity, 0 to 1, and the time
it was set.

Two things set it. The `mood` worker reads both sides of the recent
conversation, her lines under her name, and judges how she feels at her latest
line, also from what she said herself: talking about something sad, being
praised. The latest exchange is set apart from the earlier conversation, which
is background only. It is also given who she is, her `CharacterProfile.background`
(falling back to `description` when background is empty) and personality cut
to about 400 characters, so it reads her relative to her personality: a shy
character's stammer as embarrassed, a tsundere's habitual barbs as her manner
rather than anger. Hosts should put who the character actually is in
`background`; a host that instead crams its whole system prompt into
`description` will have the mood worker read that prompt as her persona.
`background` also appears in her conversation system prompt
(`ContextBuilder.build_system_prompt`), as its own `Background:` section,
unless `description` already contains it (whitespace-normalised), so a host
whose `description` already holds the persona does not show it twice.
Its intensity is anchored (about 0.2
slight, 0.5 clear, 0.8 or more only for a major event), and small talk with no
particular feeling reads as neutral. It runs every `mood_every` turns; a near word the model
answers with (relieved, annoyed, shy and the like) counts as the mood it means,
any other word off the list is no reading. On the turns it does not read, and
without it, the observation of the
user's emotion moves her mood by rules (`ai_character_engine.state.relationship`):
sad when the user turns on her, worried when the user feels bad, happy when
the user is warm, as strong as the user's emotion was. An unremarkable turn
leaves her mood as it was. On a turn the worker reads, the rules leave her mood
to that reading, whichever of the two comes in first; the observation still
moves trust and favorability. Nothing replaces what was set for a later turn.

A host that passes its own `RelationshipStatePolicy` (`state_policy=`) hands
the companion a template: the companion works on a copy, set to its own clock,
mood settings and readings, so one instance can serve several companions.
Later changes to the policy (its `rules`, say) go through
`companion.runtime.state_policy`, not the instance the host passed.

A mood holds until something at least as strong comes along. Both the reading
and the rules are weighed against her mood as it stands now (faded): a neutral
reading leaves it to fade by itself, the same mood again is as strong as the
stronger of the two, and another mood takes over only when it is at least as
strong (`blend_mood` in `ai_character_engine.state.mood`).

Her mood fades as time passes, talked to or not: the intensity halves every
`mood_half_life_seconds`, and below `mood_floor` she is neutral again. It is
worked out when read; nothing runs in between. What reads her mood reads it
faded: the `- emotion:` line of the note and `snapshot().emotion`.
`EmotionExpressionPolicy` weighs a face by the intensity it is given.

A host that shows her face reads `snapshot()`: `emotion` is her mood now;
`mood_intensity` and `mood_updated_at` are the intensity as it was set and
when, in seconds since the epoch (`mood_intensity` is 0 once she is neutral
again); `mood_half_life_seconds` and `mood_floor` let the host fade the face
itself between snapshots and stop where the engine does.

`on_mood_change`, a function of the host's, is called with the snapshot when a
background result changed her mood. It runs on the event loop while the turn
lock is held, so it must be quick and must not block: hand the snapshot on (to
a queue, a websocket send task) rather than doing the work there. An
`async def` listener is scheduled on the running loop instead of being awaited
under the lock; such tasks are not awaited or cancelled by `close()`. `clock`, a
function returning seconds since the epoch, replaces the system clock for all of
this.

## Pictures

Give the companion a `VisionPipeline` and pass `frames` to `reply()`. Each
picture is described by the vision provider first and the description is part
of what the character is told for that turn; the conversation keeps the user's
words, not the picture or its description. `companion.sees` tells whether a
pipeline was given.

With a local model that reasons before it answers, switch the reasoning off in
the provider's `request_options` and bound the answer; on a 9B model that took
one picture from 23 s to 3 s. A host whose frames may repeat, such as a camera
picture sent with every message, passes `FrameGate(deduplicate=False)`: a turn
whose frames are all rejected fails.

## Interruption

Call `interrupt(heard)` with the part of the reply the user actually heard.

| When | Effect |
|---|---|
| Before she began to answer | `reply()` raises `TurnInterrupted`; nothing is sent to the model |
| While the reply is generated | `reply()` raises `TurnInterrupted`; the user's words and `heard` are kept |
| After the host cancelled the task awaiting `reply()` | the same, recorded afterwards |
| During playback of a finished reply | the stored reply is cut down to `heard` |

A host with one listener needs no more than `interrupt(heard)`: it is for the
reply being generated, and otherwise for the turn asked for last. A host that
serves several listeners at once says whom it means:
`interrupt(heard, conversation_id=..., turn_id=...)`, where `turn_id` is the name
the host gave the turn in `reply(..., turn_id=...)`. It tells two turns of one
conversation apart. An interruption meant for a reply that has ended never
reaches a reply someone else is being given: it cuts that very reply, also
while the next one of the same conversation is being generated in a second
window, and it can cut only the newest reply of its conversation.

What she was asked to forget, a belief that was retracted and what the host no
longer knows are taken out of the notes they were written in. The inference
server then reads the conversation again from that note on.

## Settings

`CompanionSettings` holds how often each worker runs and how patient the
companion is. The defaults were measured with one local 9B model shared by
conversation and background work; a host with a faster or separate background
model can run every worker on every turn.

| Setting | Default | Meaning |
|---|---|---|
| `emotion_every`, `mood_every`, `memory_every`, `self_memory_every`, `goal_every`, `reflection_every`, `summary_every` | 1, 2, 2, 2, 4, 6, 0 | Run the worker every N turns of a conversation; 0 turns it off. Each run reads every line of its conversation since the run before |
| `reply_check_every` | 1 | Check her reply every N turns; 0 turns it off. A run reads only that reply, her reply before it and the user's line it answers |
| `memory_conflicts` | true | Hold a memory of the user just written against the earlier ones on its topic: no call without earlier ones on its topic, else one, and one more for each `supersedes` or `contradicts` answered (at most 6) |
| `call_timeout_seconds` | 60 | One background model call |
| `max_turns_late` | 3 | A result this many turns late is still used |
| `foreground_patience_seconds` | 120 | Background work resumes after this long without an end of reply |
| `goal_max_age_days` | 7 | Goals untouched for this long leave the context |
| `plans_stay_in_conversation` | true | What she is doing, short-term goals and thoughts are in mind only in the conversation they came from; false keeps them in every conversation, as before 1.2.0 (the `short_term_goal_max_age_hours` window still applies) |
| `short_term_goal_max_age_hours` | 24 | A short-term goal untouched for this long leaves the context; 0 leaves it to `goal_max_age_days` |
| `goals_shown`, `thoughts_shown` | 3, 2 | How many goals (the most pressing) and thoughts (the newest) she keeps in mind; the rest stay stored |
| `conversations_kept` | 8 | Conversations whose history is kept in memory |
| `max_history_messages` | 40 | Messages of the conversation sent to the model |
| `stream_text_with_tools` | true | Forward text at once when tools are registered |
| `records_kept` | 256 | How many finished background tasks, commit decisions and events are kept to be asked about; a host runs for days |
| `memories_recalled` | 40 | How many memories of the conversation she is given at most; what the newest message is about comes first |
| `language` | empty | The language memories, goals and thoughts are written in; empty means the language the user writes in |
| `self_memories_kept`, `self_memories_shown` | 40, 12 | How many things she said about herself she keeps (the oldest beyond are forgotten), and how many of the newest stand in the conversation |
| `mood_half_life_seconds`, `mood_floor` | 300, 0.15 | Her mood's intensity halves every this many seconds; below the floor she is neutral again |

## Telling what the engine offers

A host that runs on more than one version of the engine asks before it relies
on a newer part: `hasattr(CharacterCompanion, "self_memories")` for what she
said about herself, `"statement_only" in
inspect.signature(CharacterCompanion.speak_up).parameters` for remarks without
a question. `ai_character_engine.companion` exports `SELF_MEMORY_LINE`, the
start of her self memories in the note, and `ASSISTANT_SPEAK` and
`ACKNOWLEDGEMENTS`, what the checks on what she says look for. From 1.1.0 it also
exports `CHARACTER_MOODS`, and `CompanionSnapshot` has `mood_half_life_seconds`;
from 1.1.1 it also has `mood_floor`. From 1.2.0 it exports `SELF_MEMORY_KINDS`
and `CONVERSATION_SELF_MEMORY_KINDS`, `CompanionSettings` has
`plans_stay_in_conversation`, `short_term_goal_max_age_hours` and
`reply_check_every`, and `self_memories()` takes `in_conversation`. It also
exports `MemoryConflict`; `CompanionSettings` has `memory_conflicts`, and
`hasattr(CharacterCompanion, "memory_conflicts")` tells whether facts that no
longer agree are found.

## Lifecycle

`settle()` waits for background work under way. `flush()` saves state and keeps
the companion running. `close()` stops background work and saves state; a closed
companion refuses further turns.

One companion serves one character and must be the only writer of its
`storage_dir`. To replace a companion from synchronous code, call `retire()`
on the old one before creating the new one, and await `close()` on it later.
A retired companion finishes the reply it is in the middle of, takes no further
turn, and neither schedules nor commits background work. A turn that waited for
that reply raises `CompanionClosed`; the host gives it to the successor. What
the last reply changed in her state is not saved: the successor has read the
state by then.
A companion belongs to the event loop of its first turn;
`usable_in_running_loop()` tells a host whether it may reuse one.
