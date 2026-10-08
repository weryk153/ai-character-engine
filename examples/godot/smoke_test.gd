extends SceneTree
## Runs every call of the client once against a running service:
##   godot --headless --path examples/godot --script res://smoke_test.gd
## Exits 0 when all went as expected. Start tools/fake_service.py first (or a
## real service: then the replies take as long as your model). The service is
## at http://127.0.0.1:8765 unless COMPANION_URL says otherwise.

const Client := preload("res://addons/companion_client/companion_client.gd")
const MORNING := 986104800.0  # 2001-04-01 06:00 UTC

var client: Node
var failures := 0


func _initialize() -> void:
	client = Client.new()
	client.slot = "smoke"
	if OS.get_environment("COMPANION_URL") != "":
		client.base_url = OS.get_environment("COMPANION_URL")
	root.add_child(client)
	_run.call_deferred()


## The next `done` signal's arguments, or [] after `seconds`.
func _done_within(seconds: float) -> Array:
	var got := []
	var keep := func(npc, id, text, interrupted, state): got.assign([npc, id, text, interrupted, state])
	client.done.connect(keep, CONNECT_ONE_SHOT)
	var deadline := Time.get_ticks_msec() + int(seconds * 1000)
	while got.is_empty() and Time.get_ticks_msec() < deadline:
		await process_frame
	if client.done.is_connected(keep):
		client.done.disconnect(keep)
	return got


func _check(what: String, ok: bool, detail = "") -> void:
	print("%s %s %s" % ["ok  " if ok else "FAIL", what, "" if ok else str(detail)])
	if not ok:
		failures += 1


func _run() -> void:
	var health: Dictionary = await client.health()
	_check("health", health.ok, health)
	if not health.ok:
		quit(1)
		return
	await client.new_game()
	client.game_time = MORNING
	var npc := {"name": "Mira", "description": "The keeper of the lighthouse."}
	var avatar := {"expressions": ["smile", "thoughtful"], "motions": {"point": ""}, "mood_faces": {}}
	var opened: Dictionary = await client.open_npc("mira", npc, avatar)
	_check("open", opened.ok and opened.data.opened, opened)

	var answer: Dictionary = await client.reply("mira", "The tide is high today", "smoke")
	_check("reply", answer.ok and str(answer.data.text) != "", answer)

	var connected: bool = await client.connect_socket()
	_check("socket", connected)
	var deltas := []
	client.delta.connect(func(_npc, _id, text): deltas.append(text))
	var faces := []
	client.actions.connect(func(_npc, _id, picked): faces.append(picked))
	var id: String = client.say("mira", "Is the lamp lit?", "smoke")
	var finished := await _done_within(60.0)
	_check("streamed reply", finished.size() == 5 and finished[1] == id and deltas.size() > 0 and not finished[3], finished)
	for _wait in range(50):
		if not faces.is_empty():
			break
		await create_timer(0.1).timeout
	_check("line actions", not faces.is_empty() and faces[0].get("expression") in ["smile", "thoughtful"], faces)

	var saved: Dictionary = await client.save_slot()
	_check("save slot", saved.ok and saved.data.npcs.has("mira"), saved)
	var remembered: Dictionary = await client.remember_across_runs("mira", "Someone warned you of the storm.", 1)
	_check("remember across runs", remembered.ok, remembered)

	_check("new game", (await client.new_game()).ok)
	var gone: Dictionary = await client.state("mira")
	_check("npc gone after new game", not gone.ok and gone.code == "npc_not_open", gone)
	await client.open_npc("mira", npc)
	var loaded: Dictionary = await client.load_slot(saved.data.npcs)
	_check("load slot", loaded.ok, loaded)

	var seen: Dictionary = await client.inspect("mira", "smoke")
	_check("memory back after load", seen.ok and seen.data.memories.size() > 0, seen)
	_check("across runs kept", seen.ok and seen.data.across_runs.size() == 1, seen)
	_check("clock is the game's", seen.ok and is_equal_approx(float(seen.data.clock), MORNING), seen)

	var forgot: Dictionary = await client.forget_across_runs("mira", remembered.data.id)
	_check("forget across runs", forgot.ok, forgot)
	await client.new_game()
	print("SMOKE %s" % ("OK" if failures == 0 else "FAILED (%d)" % failures))
	quit(0 if failures == 0 else 1)
