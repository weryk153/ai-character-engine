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
name (`emotion`, `memory`, `self_memory`, `goal`, `reflection`, `summary`) to a
client. A worker
without a client does not run; `background_llm={}` turns background cognition
off. Background workers must return JSON, so give them a low temperature.

Without `storage_dir` nothing is written to disk.

## What happens in a turn

1. Background work gives way: calls in progress are abandoned and redone after
   the reply, so the reply has the model to itself.
2. The reply is generated and streamed. Text is forwarded as it is generated even
   when tools are registered (`stream_text_with_tools`), one sentence at a time.
   A sentence that repeats one of her latest lines or is nothing but
   punctuation is left out, and so is a closing that talks like an assistant
   offering help ("let me know if you need anything", "如果還有其他問題..."):
   one of the last two sentences, in her own voice, not in quotes. Such a
   sentence waits until two more have come or the reply has ended. The
   conversation keeps what was passed on. A
   reply of which nothing is left for repetition or assistant talk, and which
   used no tool, is asked again once.
3. Trust grows a little; an observation of the user's emotion committed since the
   last turn moves mood, trust and favorability
   (`ai_character_engine.state.relationship`).
4. Background jobs for this turn are scheduled. The emotion of the turn is read
   first and alone; the others follow one at a time.
5. Results are committed when they arrive, also when turns have happened in
   between (up to `max_turns_late`; memory however late), unless a newer job of
   the same kind makes them obsolete.

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
| `memories(conversation_id)` / `rewrite_memories(conversation_id, lines, edited_from=shown)` | Show what she remembers of a conversation, and take the user's edit back: a shown line that is gone is forgotten, a new line is remembered; what arrived while the page was open stays |
| `self_memories()` / `rewrite_self_memories(lines, edited_from=shown)` | What she said about herself, in any conversation, oldest first, and the user's edit of it, the same way as `rewrite_memories`. Beyond `self_memories_kept` the oldest are forgotten. For memories brought in from before the engine kept them, pass `from_before=True` (and `edited_from=[]` to only add): they are dated before every one she holds, so they are in mind last and forgotten first |
| `speak_up(conversation_id, notes=[...])` | She speaks up on her own; the host decides when. What she says comes from her: what is still open, what she wants and thinks, or turning to the user. `notes` suggest material for this remark only; `instruction` replaces the engine's own, for a host that asks in the language she speaks. What she says is generated whole and checked before any of it is passed on: the sentences a reply leaves out are left out, and so are sentences that quote back the words of her latest remarks. A remark that repeats her, or that only acknowledges ("OK.", "嗯。") when nobody said anything, is asked again, and after three attempts she stays quiet. `statement_only=True` leaves out questions as well, for a host whose user stayed quiet through her last questions. What she said stays in the conversation, the instruction does not; `keep=False` for a host that keeps what she said itself |
| `remember_remark(conversation_id, text)` | Keep a remark she made on her own when the turn that made it was kept out of memory (a long instruction the host does not want kept): what she said stays in the conversation after a short event, so she neither repeats it nor says it again when the user answers |
| `aside(make_call)` | A model call of the host's own on the same local model (a memory of its own to tidy, a translation): it waits while she replies, gives way once to a reply that starts, and takes its turn after her workers |

## Conversations

`conversation_id` selects the history and the memory scope. State, goals and
reflections belong to the character and are shared by all conversations.

So does what she said about herself. The `self_memory` worker reads her own
lines, never the user's, and keeps what she stated about herself: her tastes,
habits, history, what she is working on. Each item needs an exact quote of
hers. Said again, in the same words whatever the punctuation, it replaces the
one she held; anything said differently is a fact of its own, since nearly the
same words can say the opposite ("likes" and "dislikes"). What the user never
heard, the rest of a reply cut short by `interrupt()` or a reply taken back
before the worker's result came in, is not kept. The newest
`self_memories_shown` stand in every conversation as `- you said about
yourself: ...` lines of the note, and the system prompt asks her to stay
consistent with them. One she no longer holds, edited away or pushed out by
newer ones, is taken out of the notes again.

A host that keeps its own transcripts hands them over with
`load_conversation(conversation_id, messages)` before the first reply. A
conversation the companion has already seen is not replaced;
`has_conversation(conversation_id)` tells the host whether loading is needed.
The companion keeps the history of the `conversations_kept` most recent
conversations in memory and does not store transcripts on disk.

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
| `emotion_every`, `memory_every`, `self_memory_every`, `goal_every`, `reflection_every`, `summary_every` | 1, 2, 2, 4, 6, 0 | Run the worker every N turns of a conversation; 0 turns it off. Each run reads every line of its conversation since the run before |
| `call_timeout_seconds` | 60 | One background model call |
| `max_turns_late` | 3 | A result this many turns late is still used |
| `foreground_patience_seconds` | 120 | Background work resumes after this long without an end of reply |
| `goal_max_age_days` | 7 | Goals untouched for this long leave the context |
| `goals_shown`, `thoughts_shown` | 3, 2 | How many goals (the most pressing) and thoughts (the newest) she keeps in mind; the rest stay stored |
| `conversations_kept` | 8 | Conversations whose history is kept in memory |
| `max_history_messages` | 40 | Messages of the conversation sent to the model |
| `stream_text_with_tools` | true | Forward text at once when tools are registered |
| `records_kept` | 256 | How many finished background tasks, commit decisions and events are kept to be asked about; a host runs for days |
| `memories_recalled` | 40 | How many memories of the conversation she is given at most; what the newest message is about comes first |
| `language` | empty | The language memories, goals and thoughts are written in; empty means the language the user writes in |
| `self_memories_kept`, `self_memories_shown` | 40, 12 | How many things she said about herself she keeps (the oldest beyond are forgotten), and how many of the newest stand in the conversation |

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
