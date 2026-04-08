"""
server/app.py — OpenEnv-compatible app entry point.

This module re-exports the FastAPI app for openenv serve / uv run compatibility.
"""

from .main import app

__all__ = ["app"]
