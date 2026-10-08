extends Control
## One day on a lighthouse island, again and again.
##
## From 06:00 to 18:00 the player talks to Mira, waits, explores and may do two
## things that matter: warn her of the storm, give back her key. When the day
## ends they are kept across runs. At 18:00 the day begins again, the island as
## new, and Mira only has a feeling she has lived it before. A save holds the
## day as it was; loading it, she remembers only what had happened by then, and
## a deed done after the save is undone.

const Client := preload("res://addons/companion_client/companion_client.gd")
const DebugPanel := preload("res://addons/companion_client/debug_panel.gd")

const NPC := "mira"
const CONVERSATION := "island"
const DAY_START := 986104800.0  # 2001-04-01 06:00 UTC: the game's calendar
const DAY_LENGTH := 12 * 3600.0  # until 18:00
const TALK_TAKES := 10 * 60.0
const SAVE_PATH := "user://slot1.json"
const LOOP_PATH := "user://loop.cfg"
const DEEDS := {
	"warn": {
		"label": "Warn her of the storm",
		"say": "A storm will hit the island tonight. Please leave the lighthouse before dark.",
		"note": "- The player has just warned you that a storm will hit the island tonight.",
		"keep": "Someone once warned you, before it came, that a storm would hit the island.",
	},
	"key": {
		"label": "Give her the brass key",
		"say": "I found this on the beach. Is it yours?",
		"note": "- The player has just handed you the brass key to the lamp room, which you had lost.",
		"keep": "Someone once found the brass key you had lost and gave it back to you.",
	},
}

var client: Node
var character: Dictionary
var run := 1
var day_time := 0.0
var deeds_done: Dictionary = {}
var reply_id := ""
var reply_text := ""
var mood_line := ""
var ending := false

# What Mira's avatar could do; the example has no art, so her face is shown as a word.
const AVATAR := {
	"expressions": ["smile", "frown", "surprised", "thoughtful"],
	"motions": {"nod": "agreeing", "shrug": "unsure", "point": "pointing something out"},
	"mood_faces": {"happy": "smile", "sad": "frown", "angry": "frown", "surprised": "surprised", "worried": "thoughtful"},
}
var face := ""
var _status := Label.new()
var _log := RichTextLabel.new()
var _input := LineEdit.new()
var _portrait := TextureRect.new()
var _skip := Button.new()
var _actions: Array[Button] = []


func _ready() -> void:
	character = JSON.parse_string(FileAccess.get_file_as_string("res://npc.json"))
	client = Client.new()
	add_child(client)
	_read_local_settings()
	_build()
	var panel := DebugPanel.new()
	panel.client = client
	panel.npc = NPC
	panel.conversation_id = CONVERSATION
	add_child(panel)
	client.delta.connect(_on_delta)
	client.done.connect(_on_done)
	client.state_changed.connect(func(npc, state): if npc == NPC: _show(state))
	client.failed.connect(_on_failed)
	client.actions.connect(func(npc, _id, picked): if npc == NPC: _face(picked))
	run = _read_run()
	if not await client.connect_socket():
		_system("[color=red]No service at %s.[/color] Start it first (see README)." % client.base_url)
		return
	await _begin_day()


# --- the day -----------------------------------------------------------------------

func _begin_day() -> void:
	day_time = 0.0
	deeds_done.clear()
	_tick()
	var opened: Dictionary = await client.open_npc(NPC, character, AVATAR)
	if not opened.ok:
		_system("[color=red]%s[/color] %s" % [opened.code, opened.message])
		return
	_system("Day %d begins. The lighthouse stands on the hill." % run)
	var state: Dictionary = await client.state(NPC)
	if state.ok:
		_show(state.data)


func _end_day() -> void:
	_system("The sun sets. The storm comes. And then it is morning again.")
	# What the day held is kept now, not when it was done: a deed undone by
	# loading a save is not hers, and today she is still living it.
	var kept: Dictionary = await client.across_runs(NPC)
	for name in deeds_done:
		if kept.ok and not kept.data.any(func(memory): return name in memory.tags):
			await client.remember_across_runs(NPC, DEEDS[name].keep, run, [name])
	await client.new_game()
	run += 1
	_write_run()
	await _begin_day()


func _pass(seconds: float) -> void:
	if ending:
		return  # the day is already over; a second click does not end the next one
	day_time += seconds
	_tick()
	if day_time >= DAY_LENGTH:
		ending = true
		await _end_day()
		ending = false


func _tick() -> void:
	client.game_time = DAY_START + day_time
	_status.text = "%s | day %d | %s" % [mood_line if mood_line != "" else character.name, run, _clock()]


func _clock() -> String:
	var minutes := int(day_time / 60.0) + 6 * 60
	return "%02d:%02d" % [minutes / 60, minutes % 60]


# --- talking -------------------------------------------------------------------------

func _talk(text: String, notes: Array = []) -> void:
	if text.strip_edges() == "" or reply_id != "":
		return
	_busy(true)
	_log.append_text("\n[b]You:[/b] %s\n[b]%s:[/b] " % [text, character.name])
	reply_text = ""
	reply_id = client.say(NPC, text, CONVERSATION, notes)


func _on_delta(npc: String, id: String, text: String) -> void:
	if npc == NPC and id == reply_id:
		reply_text += text
		_log.append_text(text)


func _on_done(npc: String, id: String, _text: String, interrupted: bool, state: Dictionary) -> void:
	if npc != NPC or id != reply_id:
		return
	if interrupted:
		_log.append_text(" …")
	reply_id = ""
	_busy(false)
	_show(state)
	await _pass(TALK_TAKES)


func _on_failed(npc: String, id: String, code: String, message: String) -> void:
	_system("[color=red]%s[/color] %s" % [code, message])
	if npc == NPC and id == reply_id:
		reply_id = ""  # she did not answer; the player can speak again
	_busy(false)


func _skip_line() -> void:
	if reply_id != "":
		client.interrupt(NPC, reply_id, reply_text)


## Something that matters: kept across runs when the day ends (once, whichever
## day it was done).
func _deed(name: String) -> void:
	var deed: Dictionary = DEEDS[name]
	deeds_done[name] = true
	_talk(deed.say, [deed.note])


# --- saves ---------------------------------------------------------------------------

func _save() -> void:
	var saved: Dictionary = await client.save_slot()
	if not saved.ok:
		_system("[color=red]Not saved:[/color] %s" % saved.message)
		return
	var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	file.store_string(JSON.stringify({"run": run, "day_time": day_time, "deeds": deeds_done, "npcs": saved.data.npcs}))
	_system("Saved at %s." % _clock())


func _load() -> void:
	if not FileAccess.file_exists(SAVE_PATH):
		_system("Nothing saved yet.")
		return
	var save: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(SAVE_PATH))
	run = int(save.run)
	day_time = float(save.day_time)
	deeds_done = save.deeds
	_tick()
	await client.open_npc(NPC, character, AVATAR)
	var loaded: Dictionary = await client.load_slot(save.npcs)
	if not loaded.ok:
		_system("[color=red]Not loaded:[/color] %s" % loaded.message)
		return
	_system("Loaded: day %d, %s." % [run, _clock()])
	var state: Dictionary = await client.state(NPC)
	if state.ok:
		_show(state.data)


# --- what is shown -------------------------------------------------------------------

func _show(state: Dictionary) -> void:
	mood_line = "%s — %s | trust %.0f | favor %.0f" % [character.name, state.emotion, state.trust, state.favorability]
	if face != "":
		mood_line += " | face %s" % face
	_tick()


func _face(picked: Dictionary) -> void:
	face = str(picked.get("expression", "")) if picked.get("expression") != null else face
	var motion = picked.get("motion")
	if motion != null:
		_log.append_text(" [i](%s)[/i]" % motion)
	_tick()


func _system(text: String) -> void:
	_log.append_text("\n[i]%s[/i]\n" % text)


func _busy(talking: bool) -> void:
	_skip.disabled = not talking
	for button in _actions:
		button.disabled = talking


func _build() -> void:
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 16)
	add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	margin.add_child(column)

	_status.text = character.name
	column.add_child(_status)

	var middle := HBoxContainer.new()
	middle.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(middle)
	_portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_portrait.custom_minimum_size = Vector2(240, 0)
	_portrait.visible = _portrait.texture != null
	middle.add_child(_portrait)
	_log.bbcode_enabled = true
	_log.scroll_following = true
	_log.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	middle.add_child(_log)

	var talk := HBoxContainer.new()
	column.add_child(talk)
	_input.placeholder_text = "Say something to %s…" % character.name
	_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_input.text_submitted.connect(func(text): _talk(text); _input.clear())
	talk.add_child(_input)
	_skip.text = "Skip"
	_skip.disabled = true
	_skip.pressed.connect(_skip_line)
	talk.add_child(_skip)

	var row := HFlowContainer.new()
	column.add_child(row)
	_action(row, "Wait an hour", func(): _system("You wait."); _pass(3600.0))
	_action(row, "Walk the shore (2 h)", func(): _system("You walk the shore."); _pass(7200.0))
	for name in DEEDS:
		_action(row, DEEDS[name].label, _deed.bind(name))
	_action(row, "Save", _save)
	_action(row, "Load", _load)


func _action(row: Container, label: String, run_it: Callable) -> void:
	var button := Button.new()
	button.text = label
	button.pressed.connect(run_it)
	row.add_child(button)
	_actions.append(button)


# --- outside the service -------------------------------------------------------------

## local.cfg (not in version control): [server] base_url, token; [look] portrait.
func _read_local_settings() -> void:
	var local := ConfigFile.new()
	if local.load("res://local.cfg") != OK:
		return
	client.base_url = local.get_value("server", "base_url", client.base_url)
	client.token = local.get_value("server", "token", "")
	var path: String = local.get_value("look", "portrait", "")
	if path != "":
		var image := Image.load_from_file(path)
		if image != null:
			_portrait.texture = ImageTexture.create_from_image(image)


## The day number lives with the game, not with her: she forgets, the game does not.
func _read_run() -> int:
	var loop := ConfigFile.new()
	return int(loop.get_value("loop", "run", 1)) if loop.load(LOOP_PATH) == OK else 1


func _write_run() -> void:
	var loop := ConfigFile.new()
	loop.set_value("loop", "run", run)
	loop.save(LOOP_PATH)
