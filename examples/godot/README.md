# Godot example: a day that repeats

One day on a lighthouse island, from 06:00 to 18:00. You talk to Mira, wait,
walk the shore, and may do two things that matter: warn her of the storm, give
back the key she lost. At 18:00 the day begins again and the island is new,
but Mira cannot shake the feeling she has lived it before. A save holds the
day as it was; load it and she remembers only what had happened by then.

It shows what an NPC of the engine does in a game: the game's own clock, saves
the game keeps, a new game, and memories that survive it. It talks to the
companion service over HTTP and one WebSocket, and needs Godot 4.3 or later.

## Run it

1. Start the service. With a language model (LM Studio, Ollama or any
   OpenAI-compatible server):

   ```sh
   cp examples/godot/game.example.toml examples/godot/game.toml   # set the model
   uv run --extra service ai-character-engine-serve --config examples/godot/game.toml
   ```

   Or without one, with scripted replies:

   ```sh
   uv run --extra service python examples/godot/tools/fake_service.py
   ```

2. Open `examples/godot` in Godot 4.3+ and press Play (F5).

To use another address, a token or a picture of her, copy `local.example.cfg`
to `local.cfg` (not in version control).

F1 opens the inspector: her state, what she remembers, what she keeps across
runs, her clock.

## Check it

```sh
uv run --extra service python examples/godot/tools/fake_service.py &
godot --headless --path examples/godot --script res://smoke_test.gd
```

It opens an NPC, talks to her over HTTP and the socket, saves and loads the
slot, starts it over, and writes and removes a memory across runs, then prints
`SMOKE OK`.

## Use it in your game

Copy `addons/companion_client/` into your project.

```gdscript
const Client := preload("res://addons/companion_client/companion_client.gd")

var client := Client.new()
add_child(client)
client.game_time = my_calendar.seconds()          # every request carries it
await client.open_npc("mira", {"name": "Mira", "description": "..."})
await client.connect_socket()
client.delta.connect(func(npc, id, text): dialogue.append_text(text))
client.say("mira", "Good morning!", "island", ["- It is raining."])

var saved: Dictionary = await client.save_slot()  # put saved.data.npcs in your save
await client.load_slot(npcs_from_your_save)       # after open_npc of each NPC
await client.new_game()                            # then open_npc again
await client.remember_across_runs("mira", "Someone once warned you of the storm.", run)
```

Every call returns `{"ok": true, "data": ...}` or `{"ok": false, "code": ...,
"message": ...}`; the codes are listed in `docs/companion-service.md`, plus the
addon's own `unreachable`, `timeout` (an HTTP call over `request_timeout`) and
`disconnected` (a `say()` with no connection, or one cut by a dropped socket).
