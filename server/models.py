"""
Pydantic typed models for the CodeDebugger RL Environment.

Implements the OpenEnv spec: typed Action, Observation models.
All models are JSON-serializable and fully documented.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Action Model ───────────────────────────────────────────────────────────

class CodeDebuggerAction(BaseModel):
    """
    An action taken by the agent in the CodeDebugger environment.

    The agent chooses one of four action types per step:
    - submit_fix  : Submit fixed code to be graded (primary action)
    - run_code    : Execute code in sandbox to observe output (test run)
    - get_hint    : Request a hint about the bug (costs -0.1 reward)
    - skip        : Give up on the current task (costs -1.0 reward)
    """

    action: Literal["submit_fix", "run_code", "get_hint", "skip"] = Field(
        ...,
        description="The action type to perform.",
    )
    code: Optional[str] = Field(
        None,
        description="Python code string. Required for 'submit_fix' and 'run_code'. Ignored for 'get_hint' and 'skip'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "submit_fix",
                    "code": "x = 10\nif x > 5:\n    print('x is greater than 5')\n",
                },
                {"action": "get_hint"},
                {"action": "run_code", "code": "print(1 + 2)"},
                {"action": "skip"},
            ]
        }
    }


# ─── Observation Model ──────────────────────────────────────────────────────

class CodeDebuggerObservation(BaseModel):
    """
    An observation returned by the CodeDebugger environment after reset() or step().

    Contains the current task state, the buggy code challenge, and feedback
    from the last action.
    """

    # ── Episode metadata ─────────────────────────────────────────────────────
    episode_id: Optional[str] = Field(None, description="Unique ID for this episode.")
    task_id: Optional[str] = Field(None, description="Identifier of the current debugging task.")
    title: Optional[str] = Field(None, description="Human-readable task title.")
    difficulty: Optional[str] = Field(None, description="Task difficulty: 'easy', 'medium', or 'hard'.")
    category: Optional[str] = Field(
        None, description="Bug category: 'syntax', 'runtime', 'logic', or 'algorithm'."
    )

    # ── Task content ─────────────────────────────────────────────────────────
    buggy_code: Optional[str] = Field(
        None, description="The Python code snippet containing one or more bugs."
    )
    description: Optional[str] = Field(
        None, description="Natural-language description of what the code should do."
    )
    expected_output: Optional[str] = Field(
        None,
        description="The expected stdout output when the code is correctly fixed. "
        "Only revealed after the agent requests a hint.",
    )

    # ── Step tracking ────────────────────────────────────────────────────────
    steps_taken: int = Field(0, description="Number of steps taken in this episode.")
    max_steps: int = Field(15, description="Maximum steps allowed before timeout.")
    hint_used: bool = Field(False, description="Whether the agent has used a hint.")
    test_runs: int = Field(0, description="Number of run_code test actions taken.")

    # ── Reward and scoring ───────────────────────────────────────────────────
    reward: float = Field(
        0.0,
        description="Immediate reward for the last action. Range: -1.0 to 1.0.",
    )
    cumulative_reward: float = Field(
        0.0, description="Sum of all rewards so far in this episode."
    )
    task_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Normalized task completion score in [0.0, 1.0]. "
        "Only meaningful when done=True. 1.0 = perfect, 0.0 = failed.",
    )
    reward_history: List[float] = Field(
        default_factory=list,
        description="List of all rewards received in this episode.",
    )

    # ── Episode status ───────────────────────────────────────────────────────
    done: bool = Field(False, description="Whether the episode has ended.")
    success: bool = Field(False, description="Whether the agent solved the task.")
    message: str = Field("", description="Human-readable feedback from the last action.")
    last_action_error: Optional[str] = Field(
        None,
        description="Error message from the last action, if any. None if no error.",
    )

    # ── Optional per-action extras ───────────────────────────────────────────
    actual_output: Optional[str] = Field(
        None, description="Actual stdout from the last run_code or submit_fix action."
    )
    hint: Optional[str] = Field(
        None, description="Hint text, populated when action='get_hint'."
    )
    stdout: Optional[str] = Field(
        None, description="Raw stdout from run_code action."
    )
    stderr: Optional[str] = Field(
        None, description="Raw stderr from run_code action."
    )
    run_success: Optional[bool] = Field(
        None, description="Whether the run_code action executed without error."
    )

    model_config = {"json_schema_extra": {"examples": []}}


# ─── Task Metadata Model ─────────────────────────────────────────────────────

class TaskMetadata(BaseModel):
    """Lightweight metadata for a single debugging task (no solution or hint)."""

    id: str = Field(..., description="Unique task identifier.")
    title: str = Field(..., description="Human-readable task name.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Task difficulty level."
    )
    category: Literal["syntax", "runtime", "logic", "algorithm"] = Field(
        ..., description="Type of bug in the task."
    )


# ─── Environment State Model ─────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Current state of the CodeDebugger environment (returned by GET /state)."""

    initialized: bool = Field(..., description="Whether the environment has been reset at least once.")
    episode_id: Optional[str] = Field(None, description="Current episode ID.")
    task_id: Optional[str] = Field(None, description="Current task ID.")
    title: Optional[str] = Field(None, description="Current task title.")
    difficulty: Optional[str] = Field(None, description="Current task difficulty.")
    category: Optional[str] = Field(None, description="Current task category.")
    steps_taken: int = Field(0, description="Steps taken in current episode.")
    max_steps: int = Field(15, description="Maximum steps per episode.")
    hint_used: bool = Field(False, description="Whether a hint has been used.")
    test_runs: int = Field(0, description="Number of test runs taken.")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated so far.")
    done: bool = Field(False, description="Whether the current episode is complete.")
    success: bool = Field(False, description="Whether the current episode was solved.")
