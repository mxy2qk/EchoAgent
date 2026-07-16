from __future__ import annotations

import argparse

from echo_agent.dispatcher.cli import main as dispatcher_main
from echo_agent.server.cli import main as server_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="echo-agent")
    subparsers = parser.add_subparsers(dest="component", required=True)
    subparsers.add_parser("server")
    subparsers.add_parser("dispatcher")

    args, remaining = parser.parse_known_args(argv)
    if args.component == "server":
        return server_main(remaining)
    if args.component == "dispatcher":
        return dispatcher_main(remaining)
    raise ValueError(f"unknown component: {args.component}")


if __name__ == "__main__":
    raise SystemExit(main())
