from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from ai_character_engine import __version__

from .auth import AuthHook, NoopAuthHook
from .config import CharacterServiceConfig
from .errors import CharacterServiceError, ServiceUnauthorizedError
from .models import (
    ErrorResponse,
    HealthResponse,
    ImageMessageRequest,
    MessageRequest,
    MessageResponse,
    ReadinessResponse,
    SessionCreateRequest,
    SessionResponse,
    VisionMessageResponse,
    WebSocketClientMessage,
    WebSocketServerEvent,
)
from .service import CharacterService
from .streaming import BufferedCharacterStreamSource, ServiceStreamEvent


def _new_id() -> str:
    return uuid4().hex


def _request_ids(request: Request) -> tuple[str, str]:
    request_id = getattr(request.state, "request_id", None) or _new_id()
    trace_id = getattr(request.state, "trace_id", None) or request_id
    return request_id, trace_id


def create_app(
    *,
    service: CharacterService,
    config: CharacterServiceConfig | None = None,
    auth_hook: AuthHook | None = None,
) -> FastAPI:
    config = config or CharacterServiceConfig()
    auth = auth_hook or NoopAuthHook()
    app = FastAPI(title=config.title, debug=config.debug)
    app.state.character_service = service
    app.state.auth_hook = auth
    app.state.service_config = config

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=config.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or _new_id()
        request.state.trace_id = request.headers.get("X-Trace-ID") or request.state.request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    @app.exception_handler(CharacterServiceError)
    async def service_error_handler(request: Request, exc: CharacterServiceError):
        request_id, trace_id = _request_ids(request)
        body = ErrorResponse(
            error=exc.code,
            message=exc.message,
            request_id=request_id,
            trace_id=trace_id,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        request_id, trace_id = _request_ids(request)
        body = ErrorResponse(
            error="validation_error",
            message="request validation failed",
            request_id=request_id,
            trace_id=trace_id,
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        request_id, trace_id = _request_ids(request)
        body = ErrorResponse(
            error="internal_error",
            message="internal server error",
            request_id=request_id,
            trace_id=trace_id,
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    async def authenticate_http(request: Request) -> None:
        await auth.authenticate_http(request)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        await authenticate_http(request)
        return HealthResponse(version=__version__)

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready(request: Request):
        await authenticate_http(request)
        details = await service.readiness()
        is_ready = bool(details.get("ready", False))
        body = ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            version=__version__,
            details={k: v for k, v in details.items() if k != "ready"},
        )
        if is_ready:
            return body
        return JSONResponse(status_code=503, content=body.model_dump())

    @app.post("/sessions", response_model=SessionResponse, status_code=201)
    async def create_session(request: Request, payload: SessionCreateRequest) -> SessionResponse:
        await authenticate_http(request)
        return SessionResponse.model_validate(service.create_session(payload))

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(request: Request, session_id: str) -> SessionResponse:
        await authenticate_http(request)
        return SessionResponse.model_validate(service.get_session(session_id))

    @app.post("/sessions/{session_id}/close", response_model=SessionResponse)
    async def close_session(request: Request, session_id: str) -> SessionResponse:
        await authenticate_http(request)
        return SessionResponse.model_validate(service.close_session(session_id))

    @app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
    async def message(
        request: Request,
        session_id: str,
        payload: MessageRequest,
    ) -> MessageResponse:
        await authenticate_http(request)
        request_id, trace_id = _request_ids(request)
        return await service.send_message(
            session_id,
            payload.content,
            request_id=request_id,
            trace_id=trace_id,
            timeout_seconds=payload.timeout_seconds,
        )

    @app.post("/sessions/{session_id}/images", response_model=VisionMessageResponse)
    async def image_message(
        request: Request,
        session_id: str,
        payload: ImageMessageRequest,
    ) -> VisionMessageResponse:
        await authenticate_http(request)
        request_id, trace_id = _request_ids(request)
        return await service.send_image(
            session_id, payload, request_id=request_id, trace_id=trace_id
        )

    @app.post("/sessions/{session_id}/messages/stream")
    async def stream_message(
        request: Request,
        session_id: str,
        payload: MessageRequest,
    ) -> StreamingResponse:
        await authenticate_http(request)
        request_id, trace_id = _request_ids(request)
        managed = service.restore_active_session(session_id)

        async def event_stream():
            try:
                async for event in service.stream_managed_session(
                    managed,
                    payload.content,
                    request_id=request_id,
                    trace_id=trace_id,
                    timeout_seconds=payload.timeout_seconds,
                ):
                    if await request.is_disconnected():
                        break
                    yield event.to_sse()
            except asyncio.CancelledError:
                raise
            except CharacterServiceError as exc:
                yield ServiceStreamEvent(
                    "error",
                    {
                        "error": exc.code,
                        "message": exc.message,
                        "request_id": request_id,
                        "trace_id": trace_id,
                    },
                ).to_sse()
            except Exception:
                yield ServiceStreamEvent(
                    "error",
                    {
                        "error": "internal_error",
                        "message": "internal server error",
                        "request_id": request_id,
                        "trace_id": trace_id,
                    },
                ).to_sse()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
                "X-Trace-ID": trace_id,
            },
        )

    @app.websocket("/ws/sessions/{session_id}")
    async def websocket_session(websocket: WebSocket, session_id: str) -> None:
        request_id = websocket.headers.get("X-Request-ID") or _new_id()
        trace_id = websocket.headers.get("X-Trace-ID") or request_id
        try:
            await auth.authenticate_websocket(websocket)
            # Fail before accept when possible so invalid sessions do not create
            # a seemingly healthy socket. Keep this ManagedCharacterSession for
            # the lifetime of the socket so the connection is truly persistent.
            managed = service.restore_active_session(session_id)
        except CharacterServiceError as exc:
            await websocket.close(code=4404 if exc.status_code == 404 else 4409)
            return
        except Exception:
            await websocket.close(code=4401)
            return

        await websocket.accept()
        await websocket.send_json(
            WebSocketServerEvent(
                type="ready",
                request_id=request_id,
                trace_id=trace_id,
                data={"session_id": session_id},
            ).model_dump()
        )

        try:
            while True:
                raw = await websocket.receive_json()
                try:
                    client_message = WebSocketClientMessage.model_validate(raw)
                except ValidationError as exc:
                    await websocket.send_json(
                        WebSocketServerEvent(
                            type="error",
                            request_id=request_id,
                            trace_id=trace_id,
                            data={"error": "validation_error", "message": str(exc)},
                        ).model_dump()
                    )
                    continue

                if client_message.type == "ping":
                    await websocket.send_json(
                        WebSocketServerEvent(
                            type="pong",
                            request_id=request_id,
                            trace_id=trace_id,
                        ).model_dump()
                    )
                    continue

                content = (client_message.content or "").strip()
                if not content or len(content) > config.websocket_max_message_chars:
                    await websocket.send_json(
                        WebSocketServerEvent(
                            type="error",
                            request_id=request_id,
                            trace_id=trace_id,
                            data={
                                "error": "validation_error",
                                "message": "content is empty or exceeds websocket_max_message_chars",
                            },
                        ).model_dump()
                    )
                    continue

                turn_request_id = _new_id()
                turn_trace_id = trace_id
                try:
                    async for event in service.stream_managed_session(
                        managed,
                        content,
                        request_id=turn_request_id,
                        trace_id=turn_trace_id,
                        timeout_seconds=client_message.timeout_seconds,
                    ):
                        await websocket.send_json(
                            WebSocketServerEvent(
                                type=event.type,
                                request_id=turn_request_id,
                                trace_id=turn_trace_id,
                                data=event.data,
                            ).model_dump()
                        )
                except CharacterServiceError as exc:
                    await websocket.send_json(
                        WebSocketServerEvent(
                            type="error",
                            request_id=turn_request_id,
                            trace_id=turn_trace_id,
                            data={"error": exc.code, "message": exc.message},
                        ).model_dump()
                    )
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise

    return app


def build_default_stream_source(config: CharacterServiceConfig) -> BufferedCharacterStreamSource:
    return BufferedCharacterStreamSource(chunk_chars=config.stream_chunk_chars)
