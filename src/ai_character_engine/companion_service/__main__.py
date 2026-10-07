"""``ai-character-engine-serve --config game.toml``: a game's NPCs as a service."""
from __future__ import annotations

import argparse

from .app import create_app
from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-character-engine-serve", description=__doc__)
    parser.add_argument("--config", required=True, help="the service's TOML file")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
