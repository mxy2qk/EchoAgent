from __future__ import annotations

import argparse

import uvicorn

from echo_agent.config import ServerSettings
from echo_agent.server.app import create_app
from echo_agent.storage import SQLiteProjectStore


def build_parser() -> argparse.ArgumentParser:
    settings = ServerSettings.from_env()
    parser = argparse.ArgumentParser(prog="echo-agent-server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--db", default=str(settings.database_path))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(SQLiteProjectStore(args.db))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
