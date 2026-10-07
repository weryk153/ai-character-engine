"""The service's own settings: where it listens, where NPCs live, which models.

Read from TOML::

    data_dir = "game-data"            # relative to the file
    # host = "127.0.0.1"              # anything else needs a token
    # port = 8765
    # token = "..."
    # idle_close_seconds = 600

    [models.foreground]
    base_url = "http://127.0.0.1:1234/v1"
    model = "qwen/qwen3.5-9b"
    # api_key, temperature, max_tokens
    [models.foreground.extra_body]    # sent as is; a reasoning model must be told not to think
    reasoning_effort = "none"

    [models.background]               # optional: the foreground model, kept short, when left out
    base_url = "http://127.0.0.1:1235/v1"
    model = "qwen/qwen3.5-9b"
    # or one table per worker: [models.background.memory] ...

    [settings]                        # CompanionSettings fields
    language = "English"
"""
from __future__ import annotations

import ipaddress
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ai_character_engine.companion import CompanionSettings
from ai_character_engine.companion.companion import _WORKERS

WORKER_NAMES = tuple(worker.name for worker in _WORKERS)
# What tells a reasoning model not to think first, under the names servers
# use. Her background work keeps these of the foreground model's extra_body
# and nothing else: penalties against repeating words punish the keys a JSON
# answer has to repeat.
REASONING_KEYS = ("reasoning_effort", "reasoning", "chat_template_kwargs", "enable_thinking", "think")
WORKER_TEMPERATURE = 0.1
WORKER_MAX_TOKENS = 600


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            name: value
            for name, value in (("temperature", self.temperature), ("max_tokens", self.max_tokens))
            if value is not None
        }
        if self.extra_body:
            options["extra_body"] = dict(self.extra_body)
        return options

    def for_background(self) -> ModelConfig:
        """The same model for her background work, which answers in JSON: a
        low temperature, a cap, and only the switch that stops it thinking
        first (without it a reasoning model thinks until the call times out)."""
        return replace(
            self,
            temperature=WORKER_TEMPERATURE,
            max_tokens=WORKER_MAX_TOKENS,
            extra_body={key: value for key, value in self.extra_body.items() if key in REASONING_KEYS},
        )


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    data_dir: Path
    foreground: ModelConfig
    # None: the foreground model, kept short (for_background); one model for
    # every worker; or one per worker.
    background: ModelConfig | Mapping[str, ModelConfig] | None = None
    host: str = "127.0.0.1"
    port: int = 8765
    token: str | None = None
    idle_close_seconds: float = 600.0
    settings: Mapping[str, Any] = field(default_factory=dict)
    # Web pages allowed to open the WebSocket (a game exported to the web);
    # a browser sends Origin, a desktop game does not.
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _loopback(self.host) and not self.token:
            raise ValueError(f"listening on {self.host} needs a token")
        if self.idle_close_seconds <= 0:
            raise ValueError("idle_close_seconds must be > 0")
        if isinstance(self.background, Mapping):
            unknown = set(self.background) - set(WORKER_NAMES)
            if unknown:
                raise ValueError(f"unknown background workers: {', '.join(sorted(unknown))}")
        self.companion_settings()  # fails here, not on the first NPC

    def companion_settings(self) -> CompanionSettings:
        return CompanionSettings(**dict(self.settings))


LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1", "[::1]")


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _model(raw: Any, where: str) -> ModelConfig:
    if not isinstance(raw, Mapping) or "base_url" not in raw or "model" not in raw:
        raise ValueError(f"{where} needs base_url and model")
    return ModelConfig(
        base_url=str(raw["base_url"]),
        model=str(raw["model"]),
        api_key=raw.get("api_key"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        extra_body=dict(raw.get("extra_body") or {}),
    )


def load_config(path: str | Path) -> ServiceConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    models = raw.get("models") or {}
    background = models.get("background")
    if isinstance(background, Mapping) and "base_url" not in background:
        background = {
            name: _model(value, f"models.background.{name}") for name, value in background.items()
        }
    elif background is not None:
        background = _model(background, "models.background")
    data_dir = Path(raw.get("data_dir", "companion-data"))
    if not data_dir.is_absolute():
        data_dir = path.parent / data_dir
    return ServiceConfig(
        data_dir=data_dir,
        foreground=_model(models.get("foreground"), "models.foreground"),
        background=background,
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
        token=None if raw.get("token") is None else str(raw["token"]),
        idle_close_seconds=float(raw.get("idle_close_seconds", 600.0)),
        settings=dict(raw.get("settings") or {}),
        allowed_origins=tuple(str(origin) for origin in raw.get("allowed_origins") or ()),
    )
