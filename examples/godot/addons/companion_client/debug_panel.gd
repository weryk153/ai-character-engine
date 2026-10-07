extends PanelContainer
## What an NPC holds, for whoever builds the game: her state, what she
## remembers, her diary, what she keeps across runs, her clock. F1 shows and
## hides it; it reads the service again each time it is shown.
##
##   var panel = preload("res://addons/companion_client/debug_panel.gd").new()
##   panel.client = client; panel.npc = "mira"; add_child(panel)

var client: Node
var npc := ""
var conversation_id := ""

var _text := RichTextLabel.new()


func _ready() -> void:
	visible = false
	set_anchors_and_offsets_preset(Control.PRESET_RIGHT_WIDE)
	custom_minimum_size = Vector2(380, 0)
	var box := VBoxContainer.new()
	add_child(box)
	var title := Label.new()
	title.text = "NPC inspector (F1)"
	box.add_child(title)
	var refresh := Button.new()
	refresh.text = "Refresh"
	refresh.pressed.connect(show_npc)
	box.add_child(refresh)
	_text.bbcode_enabled = true
	_text.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_text.selection_enabled = true
	box.add_child(_text)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F1:
		visible = not visible
		if visible:
			show_npc()
		get_viewport().set_input_as_handled()


func show_npc() -> void:
	if client == null or npc == "":
		return
	var seen: Dictionary = await client.inspect(npc, conversation_id)
	if not seen.ok:
		_text.text = "[color=red]%s[/color]: %s" % [seen.code, seen.message]
		return
	var data: Dictionary = seen.data
	var state: Dictionary = data.state
	var lines := PackedStringArray()
	lines.append("[b]%s[/b]  clock %s" % [npc, Time.get_datetime_string_from_unix_time(int(data.clock), true)])
	lines.append("mood %s (%.2f)  trust %.0f  favor %.0f  %s" % [state.emotion, state.mood_intensity, state.trust, state.favorability, state.relationship_stage])
	_section(lines, "Goals", state.goals)
	_section(lines, "Thoughts", state.thoughts)
	_section(lines, "Remembers (this conversation)", data.memories)
	_section(lines, "Said about herself", data.self_memories)
	_section(lines, "Across runs", data.across_runs.map(func(m): return "run %s: %s" % [str(m.run), m.text]))
	_section(lines, "Diary", data.diary.map(func(d): return "%s: %s" % [d.date, d.text]))
	_text.text = "\n".join(lines)


func _section(lines: PackedStringArray, title: String, items: Array) -> void:
	lines.append("\n[b]%s[/b]" % title)
	if items.is_empty():
		lines.append("  (none)")
	for item in items:
		lines.append("  - " + str(item))
