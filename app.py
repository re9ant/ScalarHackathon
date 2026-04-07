"""
Entry point for running the CodeDebugger RL Environment server directly.

Usage:
    python app.py

Or via uvicorn:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from server.main import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
