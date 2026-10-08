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
| `idle_close_seconds` | 600 | An NPC not used this long, and with no background work under way, is closed; it comes back from `data_dir` when used |
| `allowed_origins` | none | Web pages allowed to open the WebSocket (a game exported to the web). A browser sends `Origin`; a desktop game does not, and any `Origin` not listed is refused |
| `[models.foreground]` | required | `base_url`, `model`, optional `api_key`, `temperature`, `max_tokens`, and `[models.foreground.extra_body]` sent as is: her replies. A reasoning model (Qwen 3.5, …) must be told not to think first, e.g. `reasoning_effort = "none"`, or her whole prompt can take longer than the call's 60 s |
| `[models.background]` | the foreground model, kept short | One model for her background work, or one table per worker (`[models.background.memory]`, …; `[models.background.actions]` picks the faces and gestures of her lines, and without it a per-worker table gets none). Left out: the foreground model with temperature 0.1, at most 600 tokens and only the not-thinking keys of its `extra_body` |
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

One NPC, at `/slots/{slot}/npcs/{npc}` (`slot` and `npc`: 1–64 lowercase
letters, digits, `-`, `_`; lowercase because `Slot1` and `slot1` are one
directory on macOS and Windows):

| Request | Body | Answer |
|---|---|---|
| `PUT` | `character`: `name`, `description`, `personality[]`, `speaking_style[]`, `background`, `rules[]`; `avatar` (optional): `expressions[]`, `motions{keyword: description}`, `mood_faces{mood: expression}` | `{opened}`; again: her persona (and avatar) is replaced from her next turn |
| `POST …/reply` | `text`, `conversation_id`, `notes[]` | `{text, state}`; with an `avatar`, also `actions[]` (see below): the reply waits for its picks at most `actions_timeout_seconds` (6) after her text is ready, and lines not picked by then are left out; the WebSocket sends every pick as it comes |
| `GET …/state` | | `state`: `emotion`, `mood_intensity`, `trust`, `favorability`, `relationship_stage`, `goals`, `thoughts`, `user_state`, … |
| `GET …/inspect?conversation_id=` | | `state`, `memories`, `self_memories`, `diary`, `across_runs`, `clock` |
| `POST …/save` | | `{data}`: everything she keeps, base64 |
| `POST …/load` | `{data}` | The save is checked, then she is closed and a new companion loads it. A line she is in the middle of is cut: its request answers `npc_reloaded` |

`notes` are what the game knows for this reply: a line starting with `- ` is a
fact ("- It is raining."), kept in the conversation while the game passes it.

A whole slot:

| Request | Body | Answer |
|---|---|---|
| `POST /slots/{slot}/save` | | `{npcs: {npc: data}}`: every NPC opened in the slot since the service started |
| `POST /slots/{slot}/load` | `{npcs}` | Every NPC named must be opened (`PUT`) first; one not named is emptied. Every save is checked before any NPC is closed: a load that cannot be done changes nothing |
| `DELETE /slots/{slot}` | | A new game: its NPCs are closed (a line in the middle is cut, `npc_reloaded`) and forgotten, their data removed |

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
- `{"type": "state", "npc", "state"}` when her mood changed between replies,
- `{"type": "actions", "id", "npc", "index", "line", "expression", "motion",
  "intensity", "voice"}` for an NPC opened with an `avatar`: a face and a
  gesture for one line of her reply, picked from the avatar's lists by her
  background model, her mood and the line before. Lines end at `。！？!?…`, a
  new line, or a full stop before a space; `index` counts them from 0. A line
  that gets nothing is skipped, and picks may come after `done`. `voice` is
  the tone of the line (the first expression picked in the reply, else her
  mood through `mood_faces`), for a voice with one reference per feeling.

`id` is the game's own name for the request, required on every message.
Messages are JSON in text frames. One NPC answers one line at a time; the
next waits. A wrong token or an `Origin` not allowed refuses the handshake
(HTTP 403). If the socket drops, a reply under way is still finished and kept
in her history; the game only misses its messages (`inspect` or `state` tell
where she is). The token never appears in the service's log.

The service answers only to `127.0.0.1`, `localhost` and `::1` as host names
while it listens on loopback, so that a page in the player's browser cannot
reach it by a name of its own.

## Errors

`{"code", "message"}`, and the same codes in WebSocket `error` messages:

| Code | HTTP | When |
|---|---|---|
| `invalid_request` | 400 | A malformed body, a `game_time` that is not a date, or a slot or NPC name that is not allowed |
| `unauthorized` | 401 | A token is set and the request has none or another |
| `npc_not_open` | 404 | The NPC was not opened in this slot since the service started: `PUT` it |
| `not_found` | 404 | No memory across runs with that id |
| `state_busy` | 409 | A save waited 120 s for her background work; try again |
| `npc_reloaded` | 409 | A load or a new game closed her while she was answering this request |
| `bad_save` | 422 | Not a save, damaged, or of a newer format |
| `other_character` | 422 | A save of another NPC |
| `model_unavailable` | 502 | The model could not be reached |
| `model_timeout` | 504 | The model did not answer in time |
| `not_found`, `method_not_allowed` | 404, 405 | A path or a method the service does not have |
| `internal_error` | 500 | Anything else; the service log has the details |
