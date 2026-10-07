"""A game's NPCs over HTTP and WebSocket: CharacterCompanion as a service.

``ai-character-engine-serve --config game.toml`` (needs the ``service`` extra).
"""
from .app import create_app
from .config import ModelConfig, ServiceConfig, load_config
from .registry import GameClock, NpcRegistry

__all__ = ["GameClock", "ModelConfig", "NpcRegistry", "ServiceConfig", "create_app", "load_config"]
