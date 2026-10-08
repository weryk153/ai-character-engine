"""HTTP and WebSocket endpoints of the companion service.

Every NPC lives at ``/slots/{slot}/npcs/{npc}``; a request that changes one
may carry ``game_time`` (seconds since the epoch, the game's own calendar),
which becomes that NPC's clock. Errors are ``{"code", "message"}``.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import dataclasses
import hmac
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ai_character_engine import CharacterProfile
from ai_character_engine._version import VERSION
from ai_character_engine.companion import (
    AvatarChoices,
    CharacterCompanion,
    CompanionClosed,
    CompanionSnapshot,
    StateBusy,
    StateFormatError,
    TurnInterrupted,
)
from ai_character_engine.companion import save_state
from ai_character_engine.llm.errors import LLMError

from .config import LOOPBACK_NAMES, ModelConfig, ServiceConfig, _loopback
from .line_picks import LinePicks

logger = logging.getLogger(__name__)
from .registry import InvalidRequest as InvalidRequestError
from .registry import NpcRegistry, ServiceError, Unauthorized, checked, openai_client


class StateBusyError(ServiceError):
    status, code = 409, "state_busy"


class NpcReloaded(ServiceError):
    status, code = 409, "npc_reloaded"


class BadSave(ServiceError):
    status, code = 422, "bad_save"


class OtherCharacter(ServiceError):
    status, code = 422, "other_character"


class NotFound(ServiceError):
    status, code = 404, "not_found"


class ModelUnavailable(ServiceError):
    status, code = 502, "model_unavailable"


class ModelTimeout(ServiceError):
    status, code = 504, "model_timeout"


# --- request bodies ----------------------------------------------------------------

# Seconds since the epoch that a date can hold (years 1-9999): a clock past
# them would break every later request of the NPC.
GameTime = Annotated[float, Field(allow_inf_nan=False, ge=-62135596800.0, le=253402300799.0)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CharacterIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    personality: list[str] = []
    speaking_style: list[str] = []
    background: str | None = None
    rules: list[str] = []


class AvatarIn(BaseModel):
    """What the game's avatar of her can do; her lines then get faces and gestures."""

    expressions: list[str] = []
    motions: dict[str, str] = {}
    mood_faces: dict[str, str] = {}


class OpenIn(BaseModel):
    character: CharacterIn
    game_time: GameTime | None = None
    avatar: AvatarIn | None = None


class ReplyIn(BaseModel):
    text: Text
    conversation_id: str | None = None
    notes: list[str] = []
    game_time: GameTime | None = None


class TimeIn(BaseModel):
    game_time: GameTime | None = None


class LoadIn(BaseModel):
    data: str
    game_time: GameTime | None = None


class SlotLoadIn(BaseModel):
    npcs: dict[str, str]
    game_time: GameTime | None = None


class AcrossRunsIn(BaseModel):
    text: Text
    tags: list[str] = []
    run: int | None = None
    game_time: GameTime | None = None


# --- helpers -------------------------------------------------------------------------


def state_of(snapshot: CompanionSnapshot) -> dict[str, Any]:
    return dataclasses.asdict(snapshot)


def _bytes(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadSave("data is not base64") from exc


def _text(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


async def _export(companion: CharacterCompanion) -> str:
    try:
        return _text(await companion.export_state())
    except StateBusy as exc:
        raise StateBusyError(str(exc)) from exc


def _save_of(npc: str, data: str) -> bytes:
    """The save, read and checked before anyone is closed for it."""
    raw = _bytes(data)
    try:
        manifest, _ = save_state.unpack(raw)
    except StateFormatError as exc:
        raise BadSave(str(exc)) from exc
    if manifest.get("character_id") != npc:
        raise OtherCharacter(f"a save of {manifest.get('character_id')!r}, not of {npc!r}")
    return raw


def _import(companion: CharacterCompanion, raw: bytes) -> None:
    try:
        companion.import_state(raw)
    except StateFormatError as exc:
        raise BadSave(str(exc)) from exc
    except ValueError as exc:
        raise OtherCharacter(str(exc)) from exc


def _causes(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


async def _reply(companion: CharacterCompanion, entry, body: ReplyIn, **options):
    try:
        return await companion.reply(
            body.text,
            conversation_id=body.conversation_id,
            notes=body.notes,
            before_turn=entry.take_character,
            **options,
        )
    except (TurnInterrupted, CompanionClosed, asyncio.CancelledError) as exc:
        if companion.usable_in_running_loop():
            raise  # an interruption by the player, or this request cancelled
        task = asyncio.current_task()
        if isinstance(exc, asyncio.CancelledError) and task is not None and task.cancelling():
            raise
        raise NpcReloaded("she was reloaded (a load or a new game) while she spoke") from exc
    except Exception as exc:
        # The turn reports a model failure as a HostBridgeError caused by it.
        for cause in _causes(exc):
            if isinstance(cause, TimeoutError):
                raise ModelTimeout(str(cause) or "the model did not answer in time") from exc
            if isinstance(cause, LLMError):
                raise ModelUnavailable(str(cause)) from exc
        raise


class RecentTurns:
    """A socket's latest requests: which NPC and conversation each was for,
    so that an interruption by id finds its turn. The oldest are let go."""

    def __init__(self, kept: int = 256) -> None:
        self._kept = kept
        self._turns: dict[str, tuple[str, str | None]] = {}

    def add(self, request: str, npc: str, conversation_id: str | None) -> None:
        self._turns.pop(request, None)
        self._turns[request] = (npc, conversation_id)
        while len(self._turns) > self._kept:
            del self._turns[next(iter(self._turns))]

    def get(self, request: str) -> tuple[str, str | None] | None:
        return self._turns.get(request)


class HideTokens(logging.Filter):
    """uvicorn logs every WebSocket path with its query: ``?token=`` in it is
    replaced before the line is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                TOKEN_IN_QUERY.sub(r"\1***", arg) if isinstance(arg, str) else arg for arg in record.args
            )
        return True


TOKEN_IN_QUERY = re.compile(r"([?&]token=)[^&\s\"]*")


def create_app(
    config: ServiceConfig,
    *,
    make_llm: Callable[[ModelConfig], Any] = openai_client,
    registry: NpcRegistry | None = None,
) -> FastAPI:
    npcs = registry or NpcRegistry(config, make_llm=make_llm)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async def close_idle() -> None:
            while True:
                await asyncio.sleep(min(30.0, config.idle_close_seconds / 2))
                await npcs.close_idle()

        task = asyncio.create_task(close_idle())
        try:
            yield
        finally:
            task.cancel()
            await npcs.close()

    app = FastAPI(title="AI Character Engine companion service", lifespan=lifespan)
    app.state.npcs = npcs
    replies: set[asyncio.Task] = set()
    if _loopback(config.host):
        # A page in the player's browser reaching 127.0.0.1 by a name of its
        # own (DNS rebinding) is turned away.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_NAMES))

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": exc.message})

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
        )
        return JSONResponse(status_code=400, content={"code": "invalid_request", "message": message})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, f"http_{exc.status_code}")
        return JSONResponse(status_code=exc.status_code, content={"code": code, "message": str(exc.detail)})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        logger.exception("request failed: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"code": "internal_error", "message": str(exc)})

    def _token_matches(given: str | None) -> bool:
        return given is not None and hmac.compare_digest(given.encode(), str(config.token).encode())

    def authorized(request: Request) -> None:
        if config.token and not _token_matches(request.headers.get("authorization", "").removeprefix("Bearer ")):
            raise Unauthorized("a token is needed: Authorization: Bearer <token>")

    guarded = [Depends(authorized)]

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    # --- one NPC ----------------------------------------------------------------------

    @app.put("/slots/{slot}/npcs/{npc}", dependencies=guarded)
    async def open_npc(slot: str, npc: str, body: OpenIn):
        character = CharacterProfile(id=checked(npc, "npc"), **body.character.model_dump())
        avatar = AvatarChoices(**body.avatar.model_dump()) if body.avatar is not None else None
        return {"opened": npcs.open(slot, npc, character, body.game_time, avatar=avatar)}

    @app.post("/slots/{slot}/npcs/{npc}/reply", dependencies=guarded)
    async def reply(slot: str, npc: str, body: ReplyIn):
        companion = npcs.companion(slot, npc, body.game_time)
        entry = npcs.entry(slot, npc)
        if entry.avatar is None:
            result = await _reply(companion, entry, body)
            return {"text": result.text, "state": state_of(companion.snapshot())}
        picks = LinePicks(companion.reply_actions(entry.avatar), lambda message: None)
        try:
            result = await _reply(companion, entry, body, on_text_delta=picks.feed)
        except BaseException:
            picks.task.cancel()
            raise
        picks.finish()
        await picks.task
        return {"text": result.text, "state": state_of(companion.snapshot()), "actions": picks.picked}

    @app.get("/slots/{slot}/npcs/{npc}/state", dependencies=guarded)
    async def state(slot: str, npc: str):
        return state_of(npcs.companion(slot, npc).snapshot())

    @app.get("/slots/{slot}/npcs/{npc}/inspect", dependencies=guarded)
    async def inspect(slot: str, npc: str, conversation_id: str | None = None):
        companion = npcs.companion(slot, npc)
        return {
            "state": state_of(companion.snapshot()),
            "memories": companion.memories(conversation_id),
            "self_memories": companion.self_memories(),
            "diary": [dataclasses.asdict(day) for day in companion.diary()],
            "across_runs": [dataclasses.asdict(memory) for memory in companion.across_runs()],
            "clock": npcs.entry(slot, npc).clock(),
        }

    @app.post("/slots/{slot}/npcs/{npc}/save", dependencies=guarded)
    async def save(slot: str, npc: str, body: TimeIn | None = None):
        companion = npcs.companion(slot, npc, body.game_time if body else None)
        return {"data": await _export(companion)}

    @app.post("/slots/{slot}/npcs/{npc}/load", dependencies=guarded)
    async def load(slot: str, npc: str, body: LoadIn):
        npcs.entry(slot, npc)
        raw = _save_of(npc, body.data)  # refused before she is closed
        npcs.entry(slot, npc).clock.set(body.game_time)
        _import(await npcs.reopen(slot, npc), raw)
        return {}

    # --- a whole slot ---------------------------------------------------------------

    @app.post("/slots/{slot}/save", dependencies=guarded)
    async def save_slot(slot: str, body: TimeIn | None = None):
        saved = {}
        for npc in npcs.npcs_of(slot):
            saved[npc] = await _export(npcs.companion(slot, npc, body.game_time if body else None))
        return {"npcs": saved}

    @app.post("/slots/{slot}/load", dependencies=guarded)
    async def load_slot(slot: str, body: SlotLoadIn):
        # Every save is read and checked before any NPC is closed for it.
        saves = {}
        for npc, data in body.npcs.items():
            npcs.entry(slot, npc)
            saves[npc] = _save_of(npc, data)
        for npc in npcs.npcs_of(slot):
            npcs.entry(slot, npc).clock.set(body.game_time)
            if npc in saves:
                _import(await npcs.reopen(slot, npc), saves[npc])
            else:
                await npcs.reopen(slot, npc, fresh=True)
        return {}

    @app.delete("/slots/{slot}", dependencies=guarded)
    async def new_game(slot: str):
        await npcs.delete_slot(checked(slot, "slot"))
        return {}

    # --- across runs ------------------------------------------------------------------

    @app.post("/npcs/{npc}/across-runs", dependencies=guarded)
    async def remember(npc: str, body: AcrossRunsIn):
        created_at = time.time() if body.game_time is None else body.game_time
        memory = npcs.across_runs(npc).add(body.text, tags=body.tags, created_at=created_at, run=body.run)
        return {"id": memory.id}

    @app.get("/npcs/{npc}/across-runs", dependencies=guarded)
    async def across_runs(npc: str):
        return [dataclasses.asdict(memory) for memory in npcs.across_runs(npc).memories]

    @app.delete("/npcs/{npc}/across-runs/{memory_id}", dependencies=guarded)
    async def forget(npc: str, memory_id: str):
        if not npcs.across_runs(npc).forget(memory_id):
            raise NotFound(f"no memory {memory_id} of {npc}")
        return {}

    # --- streaming --------------------------------------------------------------------

    @app.websocket("/slots/{slot}/ws")
    async def stream(websocket: WebSocket, slot: str):
        if config.token and not _token_matches(websocket.query_params.get("token")):
            await websocket.close(code=4401)
            return
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in config.allowed_origins:
            # A web page; a game does not send Origin.
            await websocket.close(code=4403)
            return
        try:
            checked(slot, "slot")
        except ServiceError:
            await websocket.close(code=4400)
            return
        await websocket.accept()
        outbox: asyncio.Queue[dict] = asyncio.Queue()
        turns = RecentTurns()
        running: dict[str, str] = {}  # request -> what the player saw of it, once interrupted

        def told(npc: str, snapshot: CompanionSnapshot) -> None:
            outbox.put_nowait({"type": "state", "npc": npc, "state": state_of(snapshot)})

        stop_listening = npcs.listen(slot, told)

        def fail(request: str, npc: str, code: str, message: str) -> None:
            outbox.put_nowait({"type": "error", "id": request, "npc": npc, "code": code, "message": message})

        async def send_all() -> None:
            while True:
                await websocket.send_json(await outbox.get())

        async def one_reply(request: str, npc: str, message: dict) -> None:
            running[request] = ""
            picks: LinePicks | None = None
            try:
                try:
                    body = ReplyIn.model_validate(message)
                except ValidationError as exc:
                    raise InvalidRequestError(
                        "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                    ) from exc
                companion = npcs.companion(slot, npc, body.game_time)
                entry = npcs.entry(slot, npc)
                turns.add(request, npc, body.conversation_id)
                if entry.avatar is not None:
                    picks = LinePicks(
                        companion.reply_actions(entry.avatar),
                        lambda message: outbox.put_nowait({"type": "actions", "id": request, "npc": npc, **message}),
                    )
                    # Picks may be made after her reply has ended: kept by the app.
                    replies.add(picks.task)
                    picks.task.add_done_callback(replies.discard)

                def delta(text: str) -> None:
                    outbox.put_nowait({"type": "delta", "id": request, "npc": npc, "text": text})
                    if picks is not None:
                        picks.feed(text)

                try:
                    result = await _reply(
                        companion, npcs.entry(slot, npc), body, on_text_delta=delta, turn_id=request
                    )
                    text, interrupted = result.text, False
                except TurnInterrupted:
                    text, interrupted = running.get(request, ""), True
                outbox.put_nowait(
                    {
                        "type": "done",
                        "id": request,
                        "npc": npc,
                        "text": text,
                        "interrupted": interrupted,
                        "state": state_of(companion.snapshot()),
                    }
                )
            except ServiceError as exc:
                fail(request, npc, exc.code, exc.message)
            except Exception as exc:  # noqa: BLE001 - reported to the game, the socket stays open
                logger.exception("reply %s of %s failed", request, npc)
                fail(request, npc, "internal_error", str(exc))
            finally:
                running.pop(request, None)
                if picks is not None:
                    picks.finish()

        def interrupt(request: str, message: dict) -> None:
            npc, conversation = turns.get(request) or (str(message.get("npc", "")), None)
            heard = str(message.get("heard", ""))
            if request in running:
                running[request] = heard
            try:
                npcs.companion(slot, npc).interrupt(heard, conversation_id=conversation, turn_id=request)
            except ServiceError as exc:
                fail(request, npc, exc.code, exc.message)

        sender = asyncio.create_task(send_all())
        try:
            while True:
                frame = await websocket.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if frame.get("text") is None:
                    fail("", "", "invalid_request", "send JSON in text frames")
                    continue
                try:
                    message = json.loads(frame["text"])
                except ValueError:
                    fail("", "", "invalid_request", "not JSON")
                    continue
                if not isinstance(message, dict):
                    fail("", "", "invalid_request", "send a JSON object")
                    continue
                kind, request = message.get("type"), message.get("id")
                npc = str(message.get("npc", ""))
                if kind not in ("reply", "interrupt"):
                    fail(str(request or ""), npc, "invalid_request", f"unknown type {kind!r}")
                elif not isinstance(request, str) or not request:
                    fail("", npc, "invalid_request", "every request needs an id of the game's own")
                elif kind == "reply":
                    # Kept by the app, not by this socket: a line under way is
                    # finished and kept in her history if the socket drops.
                    task = asyncio.create_task(one_reply(request, npc, message))
                    replies.add(task)
                    task.add_done_callback(replies.discard)
                else:
                    interrupt(request, message)
        except WebSocketDisconnect:
            pass
        finally:
            stop_listening()
            sender.cancel()

    return app
