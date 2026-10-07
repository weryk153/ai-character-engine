"""``ai-character-engine-serve --config game.toml``: a game's NPCs as a service."""
from __future__ import annotations

import argparse

from .app import HideTokens, create_app
from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-character-engine-serve", description=__doc__)
    parser.add_argument("--config", required=True, help="the service's TOML file")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    import logging

    import uvicorn

    server = uvicorn.Config(create_app(config), host=config.host, port=config.port)
    # After uvicorn has set its logging up: the token stays out of the log.
    for name in ("uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addFilter(HideTokens())
    uvicorn.Server(server).run()


if __name__ == "__main__":
    main()
