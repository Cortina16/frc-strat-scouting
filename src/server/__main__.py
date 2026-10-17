"""CLI Runner for 3D Match Digital Twin Webserver.

Provides command-line entrypoint supporting configurable host, port, and auto-reload flags.
"""

from __future__ import annotations

import argparse
import sys
import uvicorn


from typing import Optional


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FRC 2026 3D Match Digital Twin FastAPI webserver.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="Port number to listen on (default: 3001)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reloading for code development",
    )
    parser.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Optional initial match ID to load",
    )
    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> None:
    parsed_args = parse_args(args)
    print(f"Starting 3D Match Digital Twin webserver at http://{parsed_args.host}:{parsed_args.port} (reload={parsed_args.reload})")
    uvicorn.run("src.server.app:app", host=parsed_args.host, port=parsed_args.port, reload=parsed_args.reload)


if __name__ == "__main__":
    main()
