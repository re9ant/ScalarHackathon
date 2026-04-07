"""
FastAPI server for the CodeDebugger RL Environment.

Exposes the OpenEnv-compatible HTTP API:
  GET  /health    - Health check
  GET  /tasks     - List all available tasks
  POST /reset     - Reset environment (start new episode)
  POST /step      - Take an action
  GET  /state     - Current environment state
  GET  /docs      - Auto-generated API docs (Swagger UI)
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .environment import CodeDebuggerEnvironment

# ─── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Environment (one shared instance per process) ─────────────────────────

env = CodeDebuggerEnvironment()

# ─── Pydantic Models ───────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Specific task ID to load. If None, a random task is chosen.")
    difficulty: Optional[str] = Field(None, description="Filter by difficulty: 'easy', 'medium', or 'hard'")
    category: Optional[str] = Field(None, description="Filter by category: 'syntax', 'runtime', 'logic', 'algorithm'")

    model_config = {"json_schema_extra": {
        "examples": [
            {"task_id": None, "difficulty": "easy"},
            {"task_id": "syn_001"},
        ]
    }}


class StepRequest(BaseModel):
    action: str = Field(
        ...,
        description="Action to take: 'submit_fix', 'run_code', 'get_hint', or 'skip'",
    )
    code: Optional[str] = Field(
        None,
        description="Python code string (required for 'submit_fix' and 'run_code' actions)",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "action": "submit_fix",
                "code": "x = 10\nif x > 5:\n    print('x is greater than 5')\n",
            },
            {"action": "get_hint"},
            {"action": "run_code", "code": "print(1 + 1)"},
            {"action": "skip"},
        ]
    }}


# ─── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 CodeDebugger RL Environment server starting up...")
    # Pre-warm: auto-reset to have a task ready
    env.reset()
    logger.info(f"✅ Environment ready. Initial task: {env.current_task['id']}")
    yield
    logger.info("👋 CodeDebugger RL Environment server shutting down.")


# ─── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="CodeDebugger RL Environment",
    description="""
## 🐛 CodeDebugger — A Mini RL Environment for Meta OpenEnv Hackathon

An RL environment where an LLM agent must debug broken Python code snippets.

### How It Works

1. **Reset** the environment with `POST /reset` to get a buggy code task
2. **Step** through the environment with `POST /step`, choosing one of:
   - `submit_fix` — Submit your fixed code (graded immediately)
   - `run_code` — Test code without submitting
   - `get_hint` — Get a hint about the bug (costs -0.1 reward)
   - `skip` — Skip the task (costs -1.0 reward)
3. **Check state** anytime with `GET /state`

### Reward Structure

| Event | Reward |
|-------|--------|
| Correct fix (no hints) | +1.0 |
| Correct fix (after testing) | +0.8 |
| Correct fix (after hint) | +0.5 |
| Wrong submission | -0.2 |
| Using a hint | -0.1 |
| Code test fails | -0.1 |
| Skip | -1.0 |
| Timeout (>15 steps) | -1.0 |
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"], summary="Health check")
async def health():
    """Returns server health status."""
    return {
        "status": "ok",
        "environment": "CodeDebugger",
        "initialized": env._initialized,
        "current_task": env.current_task["id"] if env.current_task else None,
    }


@app.get("/tasks", tags=["Tasks"], summary="List all available tasks")
async def list_tasks():
    """Returns metadata for all available debugging tasks."""
    tasks = env.available_tasks()
    return {
        "total": len(tasks),
        "tasks": tasks,
    }


@app.post("/reset", tags=["Environment"], summary="Reset environment with a new task")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and get a new buggy code task.
    
    Optionally filter by difficulty or category, or load a specific task by ID.
    """
    try:
        obs = env.reset(
            task_id=request.task_id,
            difficulty=request.difficulty,
            category=request.category,
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", tags=["Environment"], summary="Take an action in the environment")
async def step(request: StepRequest):
    """
    Execute an action and receive an observation + reward.
    
    Actions:
    - **submit_fix**: Submit fixed Python code to be graded
    - **run_code**: Execute code to see its output (test run, not graded)
    - **get_hint**: Get a hint about what's wrong (-0.1 reward)
    - **skip**: Skip the current task (-1.0 reward)
    """
    try:
        obs = env.step(action=request.action, code=request.code)
        return obs
    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state", tags=["Environment"], summary="Get current environment state")
async def state():
    """Returns the current environment state without taking any action."""
    return env.state


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
