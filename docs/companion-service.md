# Companion service for games

`ai-character-engine-serve` runs [CharacterCompanion](companion.md) for a game:
each NPC of each save slot is a companion, and the game talks to them with JSON
over HTTP and one WebSocket per slot. Godot, Unity or anything that can send an
HTTP request can use it; [examples/godot](../examples/godot/README.md) is a
complete game and a client addon to copy.

```sh
uv run --extra service ai-character-engine-serve --config game.toml
```

## Configuration

A TOML file; paths are relative to it.

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `companion-data` | Where NPCs are kept: `slots/<slot>/<npc>/` per save slot, `meta/<npc>/` for memories across runs |
| `host`, `port` | `127.0.0.1`, `8765` | Where it listens. Any other host needs a `token` |
| `token` | none | Asked for as `Authorization: Bearer <token>`, and `?token=` on the WebSocket |
| `idle_close_seconds` | 600 | An NPC not used this long is closed; it comes back from `data_dir` when used |
| `[models.foreground]` | required | `base_url`, `model`, optional `api_key`, `temperature`, `max_tokens`: her replies |
| `[models.background]` | the foreground model | One model for her background work, or one table per worker (`[models.background.memory]`, …) |
| `[settings]` | | Any [CompanionSettings](companion.md#settings) field |

Every NPC shares one queue for the model: the background work of any of them
waits while any of them replies.

## The game's clock

A request that changes an NPC may carry `game_time`, seconds since 1970 in the
game's own calendar. It becomes that NPC's clock until the next one: what she
writes down is dated by it, her mood fades by it, her short-term goals and her
diary's day are measured by it. Turning it back (a loaded save) breaks nothing.
Without it the system clock is used.

## Endpoints

One NPC, at `/slots/{slot}/npcs/{npc}` (`slot` and `npc`: 1–64 letters, digits,
`-`, `_`):

| Request | Body | Answer |
|---|---|---|
| `PUT` | `character`: `name`, `description`, `personality[]`, `speaking_style[]`, `background`, `rules[]` | `{opened}`; again: her persona is replaced from her next turn |
| `POST …/reply` | `text`, `conversation_id`, `notes[]` | `{text, state}` |
| `GET …/state` | | `state`: `emotion`, `mood_intensity`, `trust`, `favorability`, `relationship_stage`, `goals`, `thoughts`, `user_state`, … |
| `GET …/inspect?conversation_id=` | | `state`, `memories`, `self_memories`, `diary`, `across_runs`, `clock` |
| `POST …/save` | | `{data}`: everything she keeps, base64 |
| `POST …/load` | `{data}` | She is closed and a new companion loads the save |

`notes` are what the game knows for this reply: a line starting with `- ` is a
fact ("- It is raining."), kept in the conversation while the game passes it.

A whole slot:

| Request | Body | Answer |
|---|---|---|
| `POST /slots/{slot}/save` | | `{npcs: {npc: data}}`: every NPC opened in the slot since the service started |
| `POST /slots/{slot}/load` | `{npcs}` | Every NPC named must be opened (`PUT`) first; one not named is emptied |
| `DELETE /slots/{slot}` | | A new game: its NPCs are closed and forgotten, their data removed |

Memories across runs, per NPC and not per slot, written only by the game:
`POST /npcs/{npc}/across-runs` (`text`, `tags[]`, `run`) → `{id}`;
`GET /npcs/{npc}/across-runs`; `DELETE /npcs/{npc}/across-runs/{id}`. Every
open companion of the NPC sees a change on her next turn.

`GET /health` → `{status, version}`.

## WebSocket

`/slots/{slot}/ws`. The game sends

- `{"type": "reply", "id", "npc", "text", "conversation_id", "notes", "game_time"}`
- `{"type": "interrupt", "id", "npc", "heard"}`: the player skipped her line;
  she remembers only `heard`.

and receives

- `{"type": "delta", "id", "npc", "text"}` as she speaks,
- `{"type": "done", "id", "npc", "text", "interrupted", "state"}`,
- `{"type": "error", "id", "npc", "code", "message"}`,
- `{"type": "state", "npc", "state"}` when her mood changed between replies.

`id` is the game's own name for the request. One NPC answers one line at a
time; the next waits.

## Errors

`{"code", "message"}`, and the same codes in WebSocket `error` messages:

| Code | HTTP | When |
|---|---|---|
| `invalid_request` | 400 | A malformed body, or a slot or NPC name that is not allowed |
| `unauthorized` | 401 | A token is set and the request has none or another |
| `npc_not_open` | 404 | The NPC was not opened in this slot since the service started: `PUT` it |
| `not_found` | 404 | No memory across runs with that id |
| `state_busy` | 409 | A save waited 120 s for her background work; try again |
| `bad_save` | 422 | Not a save, damaged, or of a newer format |
| `other_character` | 422 | A save of another NPC |
| `model_unavailable` | 502 | The model could not be reached |
| `model_timeout` | 504 | The model did not answer in time |
