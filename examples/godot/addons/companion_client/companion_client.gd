extends Node
## Talks to `ai-character-engine-serve`: one save slot, its NPCs, one socket.
##
## Every call returns a Dictionary: {"ok": true, "data": ...} or
## {"ok": false, "code": ..., "message": ...}. Set `game_time` (seconds since
## the epoch, your game's calendar) and it goes with every request that changes
## an NPC; leave it at -1 to use the server's clock.
##
## Streaming: call `connect_socket()` once, then `say()`; the reply arrives as
## `delta` signals and ends with `done`, or `failed` (code "disconnected" when
## there is no connection, or it dropped before her answer). `state_changed`
## comes when an NPC's mood changed between replies.

signal delta(npc: String, id: String, text: String)
signal done(npc: String, id: String, text: String, interrupted: bool, state: Dictionary)
signal state_changed(npc: String, state: Dictionary)
signal failed(npc: String, id: String, code: String, message: String)

@export var base_url := "http://127.0.0.1:8765"
@export var token := ""
@export var slot := "current"
## Seconds an HTTP call may take; a save may wait up to 120 s for her
## background work.
@export var request_timeout := 150.0

var game_time: float = -1.0

var _socket := WebSocketPeer.new()
var _socket_wanted := false
var _was_open := false
var _next_id := 0
var _pending: Dictionary = {}  # id -> npc, until done or failed


func _process(_delta: float) -> void:
	if not _socket_wanted:
		return
	_socket.poll()
	if _socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
		if _was_open:
			_was_open = false
			for id in _pending.keys():
				failed.emit(_pending[id], id, "disconnected", "the connection to the service was lost")
			_pending.clear()
		return
	_was_open = true
	while _socket.get_available_packet_count() > 0:
		var message = JSON.parse_string(_socket.get_packet().get_string_from_utf8())
		if typeof(message) != TYPE_DICTIONARY:
			continue
		var npc := str(message.get("npc", ""))
		var id := str(message.get("id", ""))
		if message.get("type") in ["done", "error"]:
			_pending.erase(id)
		match message.get("type"):
			"delta":
				delta.emit(npc, id, str(message.get("text", "")))
			"done":
				done.emit(npc, id, str(message.get("text", "")), bool(message.get("interrupted", false)), message.get("state", {}))
			"state":
				state_changed.emit(npc, message.get("state", {}))
			"error":
				failed.emit(npc, id, str(message.get("code", "")), str(message.get("message", "")))


# --- streaming -----------------------------------------------------------------------

func connect_socket() -> bool:
	var url := base_url.replace("http://", "ws://").replace("https://", "wss://") + "/slots/%s/ws" % slot
	if token != "":
		url += "?token=" + token.uri_encode()
	_socket_wanted = true
	if _socket.connect_to_url(url) != OK:
		return false
	var deadline := Time.get_ticks_msec() + 10000
	while _socket.get_ready_state() == WebSocketPeer.STATE_CONNECTING and Time.get_ticks_msec() < deadline:
		_socket.poll()
		await get_tree().process_frame
	return _socket.get_ready_state() == WebSocketPeer.STATE_OPEN


## Her reply streamed; returns the id the signals carry.
func say(npc: String, text: String, conversation_id := "", notes: Array = []) -> String:
	_next_id += 1
	var id := "r%d" % _next_id
	var message := {"type": "reply", "id": id, "npc": npc, "text": text, "notes": notes}
	if conversation_id != "":
		message["conversation_id"] = conversation_id
	if game_time >= 0.0:
		message["game_time"] = game_time
	if _socket.get_ready_state() != WebSocketPeer.STATE_OPEN or _socket.send_text(JSON.stringify(message)) != OK:
		# Told after the caller has had the id back and connected its handlers.
		failed.emit.call_deferred(npc, id, "disconnected", "no connection to the service")
		return id
	_pending[id] = npc
	return id


## The player skipped her line: she remembers only `heard`.
func interrupt(npc: String, id: String, heard: String) -> void:
	_socket.send_text(JSON.stringify({"type": "interrupt", "id": id, "npc": npc, "heard": heard}))


# --- one NPC -------------------------------------------------------------------------

func open_npc(npc: String, character: Dictionary) -> Dictionary:
	return await _call(HTTPClient.METHOD_PUT, _npc(npc), {"character": character})


func reply(npc: String, text: String, conversation_id := "", notes: Array = []) -> Dictionary:
	var body := {"text": text, "notes": notes}
	if conversation_id != "":
		body["conversation_id"] = conversation_id
	return await _call(HTTPClient.METHOD_POST, _npc(npc) + "/reply", body)


func state(npc: String) -> Dictionary:
	return await _call(HTTPClient.METHOD_GET, _npc(npc) + "/state")


func inspect(npc: String, conversation_id := "") -> Dictionary:
	var query := "" if conversation_id == "" else "?conversation_id=" + conversation_id.uri_encode()
	return await _call(HTTPClient.METHOD_GET, _npc(npc) + "/inspect" + query)


# --- the slot ------------------------------------------------------------------------

## Every NPC of the slot as base64 strings: put `data["npcs"]` in your save.
func save_slot() -> Dictionary:
	return await _call(HTTPClient.METHOD_POST, "/slots/%s/save" % slot, {})


func load_slot(npcs: Dictionary) -> Dictionary:
	return await _call(HTTPClient.METHOD_POST, "/slots/%s/load" % slot, {"npcs": npcs})


## A new game: the slot's NPCs and their data are gone; open them again.
func new_game() -> Dictionary:
	return await _call(HTTPClient.METHOD_DELETE, "/slots/%s" % slot)


# --- across runs ---------------------------------------------------------------------

func remember_across_runs(npc: String, text: String, run := -1, tags: Array = []) -> Dictionary:
	var body := {"text": text, "tags": tags}
	if run >= 0:
		body["run"] = run
	return await _call(HTTPClient.METHOD_POST, "/npcs/%s/across-runs" % npc, body)


func across_runs(npc: String) -> Dictionary:
	return await _call(HTTPClient.METHOD_GET, "/npcs/%s/across-runs" % npc)


func forget_across_runs(npc: String, memory_id: String) -> Dictionary:
	return await _call(HTTPClient.METHOD_DELETE, "/npcs/%s/across-runs/%s" % [npc, memory_id])


func health() -> Dictionary:
	return await _call(HTTPClient.METHOD_GET, "/health")


# --- plumbing ------------------------------------------------------------------------

func _npc(npc: String) -> String:
	return "/slots/%s/npcs/%s" % [slot, npc]


func _call(method: int, path: String, body = null) -> Dictionary:
	var headers := PackedStringArray(["Content-Type: application/json"])
	if token != "":
		headers.append("Authorization: Bearer " + token)
	var payload := ""
	if body != null:
		if game_time >= 0.0 and not body.has("game_time"):
			body["game_time"] = game_time
		payload = JSON.stringify(body)
	var http := HTTPRequest.new()
	http.timeout = request_timeout
	add_child(http)
	var started := http.request(base_url + path, headers, method, payload)
	if started != OK:
		http.queue_free()
		return {"ok": false, "code": "unreachable", "message": "request not sent (%d)" % started}
	var response: Array = await http.request_completed
	http.queue_free()
	if response[0] == HTTPRequest.RESULT_TIMEOUT:
		return {"ok": false, "code": "timeout", "message": "no answer from %s in %.0f s" % [base_url, request_timeout]}
	if response[0] != HTTPRequest.RESULT_SUCCESS:
		return {"ok": false, "code": "unreachable", "message": "no answer from %s (%d)" % [base_url, response[0]]}
	var text: String = response[3].get_string_from_utf8()
	var data = JSON.parse_string(text) if text != "" else {}
	if response[1] >= 400:
		var error: Dictionary = data if typeof(data) == TYPE_DICTIONARY else {}
		return {"ok": false, "status": response[1], "code": str(error.get("code", "http_%d" % response[1])), "message": str(error.get("message", text))}
	return {"ok": true, "data": data}
