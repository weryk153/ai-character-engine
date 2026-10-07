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
import json
import time
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_character_engine import CharacterProfile
from ai_character_engine._version import VERSION
from ai_character_engine.companion import (
    CharacterCompanion,
    CompanionSnapshot,
    StateBusy,
    StateFormatError,
    TurnInterrupted,
)
from ai_character_engine.llm.errors import LLMError

from .config import ModelConfig, ServiceConfig
from .registry import NpcRegistry, ServiceError, Unauthorized, checked, openai_client


class StateBusyError(ServiceError):
    status, code = 409, "state_busy"


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


class CharacterIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    personality: list[str] = []
    speaking_style: list[str] = []
    background: str | None = None
    rules: list[str] = []


class OpenIn(BaseModel):
    character: CharacterIn
    game_time: float | None = None


class ReplyIn(BaseModel):
    text: str = Field(min_length=1)
    conversation_id: str | None = None
    notes: list[str] = []
    game_time: float | None = None


class TimeIn(BaseModel):
    game_time: float | None = None


class LoadIn(BaseModel):
    data: str
    game_time: float | None = None


class SlotLoadIn(BaseModel):
    npcs: dict[str, str]
    game_time: float | None = None


class AcrossRunsIn(BaseModel):
    text: str = Field(min_length=1)
    tags: list[str] = []
    run: int | None = None
    game_time: float | None = None


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


def _import(companion: CharacterCompanion, data: str) -> None:
    try:
        companion.import_state(_bytes(data))
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
    except Exception as exc:
        # The turn reports a model failure as a HostBridgeError caused by it.
        for cause in _causes(exc):
            if isinstance(cause, TimeoutError):
                raise ModelTimeout(str(cause) or "the model did not answer in time") from exc
            if isinstance(cause, LLMError):
                raise ModelUnavailable(str(cause)) from exc
        raise


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

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": exc.message})

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
        )
        return JSONResponse(status_code=400, content={"code": "invalid_request", "message": message})

    def authorized(request: Request) -> None:
        if config.token and request.headers.get("authorization") != f"Bearer {config.token}":
            raise Unauthorized("a token is needed: Authorization: Bearer <token>")

    guarded = [Depends(authorized)]

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    # --- one NPC ----------------------------------------------------------------------

    @app.put("/slots/{slot}/npcs/{npc}", dependencies=guarded)
    async def open_npc(slot: str, npc: str, body: OpenIn):
        character = CharacterProfile(id=checked(npc, "npc"), **body.character.model_dump())
        return {"opened": npcs.open(slot, npc, character, body.game_time)}

    @app.post("/slots/{slot}/npcs/{npc}/reply", dependencies=guarded)
    async def reply(slot: str, npc: str, body: ReplyIn):
        companion = npcs.companion(slot, npc, body.game_time)
        result = await _reply(companion, npcs.entry(slot, npc), body)
        return {"text": result.text, "state": state_of(companion.snapshot())}

    @app.get("/slots/{slot}/npcs/{npc}/state", dependencies=guarded)
    async def state(slot: str, npc: str):
        return state_of(npcs.companion(slot, npc).snapshot())

    @app.get("/slots/{slot}/npcs/{npc}/inspect", dependencies=guarded)
    async def inspect(slot: str, npc: str, conversation_id: str | None = None):
        companion = npcs.companion(slot, npc)
        entry = npcs.entry(slot, npc)
        return {
            "state": state_of(companion.snapshot()),
            "memories": companion.memories(conversation_id),
            "self_memories": companion.self_memories(),
            "diary": [dataclasses.asdict(entry) for entry in companion.diary()],
            "across_runs": [dataclasses.asdict(memory) for memory in companion.across_runs()],
            "clock": entry.clock(),
        }

    @app.post("/slots/{slot}/npcs/{npc}/save", dependencies=guarded)
    async def save(slot: str, npc: str, body: TimeIn | None = None):
        companion = npcs.companion(slot, npc, body.game_time if body else None)
        return {"data": await _export(companion)}

    @app.post("/slots/{slot}/npcs/{npc}/load", dependencies=guarded)
    async def load(slot: str, npc: str, body: LoadIn):
        _bytes(body.data)  # refused before she is closed
        npcs.entry(slot, npc).clock.set(body.game_time)
        _import(await npcs.reopen(slot, npc), body.data)
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
        for npc, data in body.npcs.items():
            npcs.entry(slot, npc)
            _bytes(data)
        for npc in npcs.npcs_of(slot):
            npcs.entry(slot, npc).clock.set(body.game_time)
            if npc in body.npcs:
                _import(await npcs.reopen(slot, npc), body.npcs[npc])
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
        if config.token and websocket.query_params.get("token") != config.token:
            await websocket.close(code=4401)
            return
        try:
            checked(slot, "slot")
        except ServiceError:
            await websocket.close(code=4400)
            return
        await websocket.accept()
        outbox: asyncio.Queue[dict] = asyncio.Queue()
        turns: dict[str, tuple[str, str | None]] = {}
        heard: dict[str, str] = {}
        running: set[asyncio.Task] = set()

        def told(npc: str, snapshot: CompanionSnapshot) -> None:
            outbox.put_nowait({"type": "state", "npc": npc, "state": state_of(snapshot)})

        stop_listening = npcs.listen(slot, told)

        async def send_all() -> None:
            while True:
                await websocket.send_json(await outbox.get())

        async def one_reply(message: dict) -> None:
            request, npc = str(message.get("id", "")), str(message.get("npc", ""))
            try:
                body = ReplyIn.model_validate(message)
                companion = npcs.companion(slot, npc, body.game_time)
                turns[request] = (npc, body.conversation_id)

                def delta(text: str) -> None:
                    outbox.put_nowait({"type": "delta", "id": request, "npc": npc, "text": text})

                try:
                    result = await _reply(
                        companion, npcs.entry(slot, npc), body, on_text_delta=delta, turn_id=request
                    )
                    text, interrupted = result.text, False
                except TurnInterrupted:
                    text, interrupted = heard.pop(request, ""), True
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
                outbox.put_nowait({"type": "error", "id": request, "npc": npc, "code": exc.code, "message": exc.message})
            except Exception as exc:  # noqa: BLE001 - reported to the game, the socket stays open
                outbox.put_nowait({"type": "error", "id": request, "npc": npc, "code": "internal_error", "message": str(exc)})

        def interrupt(message: dict) -> None:
            request = str(message.get("id", ""))
            npc, conversation = turns.get(request, (str(message.get("npc", "")), None))
            heard[request] = str(message.get("heard", ""))
            try:
                npcs.companion(slot, npc).interrupt(
                    heard[request], conversation_id=conversation, turn_id=request
                )
            except ServiceError as exc:
                outbox.put_nowait({"type": "error", "id": request, "npc": npc, "code": exc.code, "message": exc.message})

        sender = asyncio.create_task(send_all())
        try:
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except ValueError:
                    outbox.put_nowait({"type": "error", "code": "invalid_request", "message": "not JSON"})
                    continue
                kind = message.get("type") if isinstance(message, dict) else None
                if kind == "reply":
                    task = asyncio.create_task(one_reply(message))
                    running.add(task)
                    task.add_done_callback(running.discard)
                elif kind == "interrupt":
                    interrupt(message)
                else:
                    outbox.put_nowait({"type": "error", "code": "invalid_request", "message": f"unknown type {kind!r}"})
        except WebSocketDisconnect:
            pass
        finally:
            stop_listening()
            for task in running:
                task.cancel()
            sender.cancel()

    return app
