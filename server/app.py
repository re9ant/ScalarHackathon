"""
server/app.py — OpenEnv-compatible app entry point.

This module re-exports the FastAPI app and provides a main() entry point
for openenv serve / uv run compatibility.
"""

import uvicorn

from .main import app

__all__ = ["app", "main"]


def main():
    """Entry point for `openenv serve` and `uv run`."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
