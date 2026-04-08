"""
FastAPI server for the CodeDebugger RL Environment.

Implements the OpenEnv-compatible HTTP API with typed Pydantic request/response models.

Endpoints:
  GET  /health     → server health
  GET  /tasks      → list all available tasks
  POST /reset      → reset environment (returns CodeDebuggerObservation)
  POST /step       → take an action (returns CodeDebuggerObservation)
  GET  /state      → current environment state (returns EnvironmentState)
  GET  /docs       → Swagger UI (auto-generated)
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .environment import CodeDebuggerEnvironment
from .models import CodeDebuggerAction, CodeDebuggerObservation, EnvironmentState, TaskMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Single shared environment instance ────────────────────────────────────

env = CodeDebuggerEnvironment()


# ─── Request Models ─────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Parameters for resetting the environment."""

    task_id: Optional[str] = Field(
        None,
        description="Pin to a specific task ID. If None, a random task is chosen.",
    )
    difficulty: Optional[str] = Field(
        None,
        description="Filter random selection by difficulty: 'easy', 'medium', or 'hard'.",
    )
    category: Optional[str] = Field(
        None,
        description="Filter by bug category: 'syntax', 'runtime', 'logic', or 'algorithm'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"difficulty": "easy"},
                {"task_id": "syn_001"},
                {"difficulty": "hard", "category": "algorithm"},
            ]
        }
    }


class TaskListResponse(BaseModel):
    """Response for GET /tasks."""

    total: int = Field(..., description="Total number of available tasks.")
    tasks: List[TaskMetadata] = Field(..., description="List of task metadata.")


# ─── Lifespan ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CodeDebugger RL Environment starting...")
    env.reset()  # warm-up
    logger.info("Ready. Initial task: %s", env.current_task["id"])
    yield
    logger.info("Shutting down.")


# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    lifespan=lifespan,
    title="CodeDebugger RL Environment",
    description="""
## 🐛 CodeDebugger — Real-World Python Debugging RL Environment

**Meta OpenEnv Hackathon Round 1**

An RL environment where an AI agent must debug broken Python code snippets.
Fully compliant with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification.

### Real-World Domain

Software debugging is one of the highest-value tasks developers perform daily.
This environment provides a structured way to train and evaluate agents on
programmatic debugging across four bug categories:

| Category | Description |
|----------|-------------|
| `syntax` | Missing colons, parentheses, quotes, indentation |
| `runtime` | Division by zero, index errors, type mismatches |
| `logic` | Wrong operators, wrong return variable, off-by-one |
| `algorithm` | Sorting, Fibonacci, binary search, base-case bugs |

### Action Space (`POST /step`)

| Action | Code Required | Description |
|--------|--------------|-------------|
| `submit_fix` | ✅ | Submit fixed code to be graded |
| `run_code` | ✅ | Execute code in sandbox (no grade) |
| `get_hint` | ❌ | Get a hint (-0.1 reward) |
| `skip` | ❌ | Skip task (-1.0 reward) |

### Reward Function

| Event | Reward | Task Score |
|-------|--------|------------|
| Correct fix (no hints, first try) | +1.0 | 1.0 |
| Correct fix (after test runs) | +0.8 | 0.9 |
| Correct fix (after hint) | +0.5 | 0.75 |
| Wrong submission | -0.2 | — |
| Code execution error | -0.1 | — |
| Using a hint | -0.1 | — |
| Skip / Timeout | -1.0 | 0.0 |

### Baseline Tasks (reproducible evaluation)
- **Easy**: `syn_001` — Missing colon in if statement
- **Medium**: `log_001` — Off-by-one error in range (sum 1..10)
- **Hard**: `hard_001` — Mutable default argument bug
""",
    version="1.0.0",
    tags_metadata=[
        {"name": "Environment", "description": "Core RL environment endpoints"},
        {"name": "Tasks", "description": "Task discovery"},
        {"name": "System", "description": "Health and metadata"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ─── Routes ─────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"], summary="Health check")
async def health():
    """Returns server health and current task."""
    return {
        "status": "ok",
        "environment": "CodeDebugger",
        "version": "1.0.0",
        "initialized": env._initialized,
        "current_task": env.current_task["id"] if env.current_task else None,
        "total_tasks": 36,
        "baseline_tasks": CodeDebuggerEnvironment.BASELINE_TASKS,
    }


@app.get(
    "/tasks",
    response_model=TaskListResponse,
    tags=["Tasks"],
    summary="List all available debugging tasks",
)
async def list_tasks():
    """Returns metadata for all 36 debugging tasks."""
    raw = env.available_tasks()
    tasks = [TaskMetadata(**t) for t in raw]
    return TaskListResponse(total=len(tasks), tasks=tasks)


@app.post(
    "/reset",
    response_model=CodeDebuggerObservation,
    tags=["Environment"],
    summary="Reset the environment with a new debugging task",
)
async def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode. Returns the initial observation including the buggy code.

    - Specify `task_id` for reproducible evaluation.
    - Specify `difficulty` and/or `category` to filter random selection.
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
        logger.error("Reset failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post(
    "/step",
    response_model=CodeDebuggerObservation,
    tags=["Environment"],
    summary="Take an action in the current episode",
)
async def step(action: CodeDebuggerAction):
    """
    Execute one action and receive an observation with reward and feedback.

    The `action` field determines what happens:
    - **submit_fix**: Grade `code` against expected output
    - **run_code**: Execute `code` in sandbox (observe output, no grade)
    - **get_hint**: Reveal a hint about the bug (-0.1 reward)
    - **skip**: End episode immediately (-1.0 reward)
    """
    try:
        obs = env.step(action)
        return obs
    except Exception as e:
        logger.error("Step failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get(
    "/state",
    response_model=EnvironmentState,
    tags=["Environment"],
    summary="Get current environment state",
)
async def state():
    """Returns the current environment state without taking any action."""
    return env.state


@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
