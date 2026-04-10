"""
RL Environment for the CodeDebugger challenge.

Implements the full OpenEnv spec:
  - Typed Action (CodeDebuggerAction) and Observation (CodeDebuggerObservation)
  - reset() → CodeDebuggerObservation
  - step(action) → CodeDebuggerObservation (with reward, done, task_score)
  - state property → EnvironmentState

Real-world task: Debugging broken Python code snippets.
Not a game or toy — software debugging is a core daily task for all developers.
"""

import logging
from typing import Optional, Union
from uuid import uuid4

from .models import CodeDebuggerAction, CodeDebuggerObservation, EnvironmentState
from .tasks import get_task, get_task_metadata
from .grader import grade_submission, run_code

logger = logging.getLogger(__name__)


class CodeDebuggerEnvironment:
    """
    CodeDebugger RL Environment — OpenEnv compliant.

    Real-world domain: Python code debugging.
    An agent must read a broken code snippet and fix the bug by submitting
    corrected Python code that passes an automated execution grader.

    Action Space:
        CodeDebuggerAction with action ∈ {submit_fix, run_code, get_hint, skip}

    Observation Space:
        CodeDebuggerObservation — typed Pydantic model containing:
        - Task info: task_id, title, difficulty, category
        - Content: buggy_code, description, expected_output (after hint)
        - Progress: steps_taken, max_steps, hint_used, test_runs
        - Signals: reward ∈ [-1.0, 1.0], task_score ∈ [0.0, 1.0]
        - Status: done, success, message, last_action_error

    Reward Function:
        Intermediate (per step):
          get_hint()    → -0.1  (penalizes hint dependency)
          run_code()    → 0.0 (success) or -0.1 (code errors)
          submit wrong  → -0.2 (penalizes incorrect guesses)
          skip()        → -1.0 (episode-ending penalty)
          timeout       → -1.0 (episode-ending penalty)

        Final (on correct fix):
          No hints, no test runs → +1.0  (perfect)
          After test runs only   → +0.8  (good)
          After using hint       → +0.5  (partial)

    Task Score (0.0–1.0, per rubric):
        Normalized from final reward: maps {-1.0…+1.0} → {0.0…1.0}
        Only meaningful when done=True.
    """

    MAX_STEPS = 15

    # ─── Baseline task set ──────────────────────────────────────────────────
    # These 3 tasks form the fixed benchmark for reproducible baseline scores.
    BASELINE_TASKS = [
        "syn_001",   # Easy: missing colon in if statement
        "log_001",   # Medium: off-by-one in range (sum 1..10)
        "hard_001",  # Hard: mutable default argument
    ]

    def __init__(self):
        self.episode_id: Optional[str] = None
        self.current_task: Optional[dict] = None
        self.steps_taken: int = 0
        self.hint_used: bool = False
        self.test_runs: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.success: bool = False
        self.last_action_error: Optional[str] = None
        self.reward_history: list = []
        self._initialized = False

    # ─── OpenEnv Interface ──────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
    ) -> CodeDebuggerObservation:
        """
        Reset the environment with a new debugging task.

        Args:
            task_id: Pin to a specific task (for reproducible evaluation).
            difficulty: Filter by difficulty ('easy', 'medium', 'hard').
            category: Filter by category ('syntax', 'runtime', 'logic', 'algorithm').

        Returns:
            CodeDebuggerObservation with initial task state.
        """
        self.episode_id = str(uuid4())
        self.current_task = get_task(task_id=task_id, difficulty=difficulty, category=category)
        self.steps_taken = 0
        self.hint_used = False
        self.test_runs = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.success = False
        self.last_action_error = None
        self.reward_history = []
        self._initialized = True

        logger.info(
            "[Episode %s] Reset → task=%s (%s)",
            self.episode_id[:8],
            self.current_task["id"],
            self.current_task["difficulty"],
        )

        return self._make_obs(reward=0.0, message="New debugging task loaded. Fix the bug!")

    def step(
        self, action: Union[CodeDebuggerAction, dict]
    ) -> CodeDebuggerObservation:
        """
        Execute one action in the environment.

        Args:
            action: A CodeDebuggerAction (or compatible dict) with fields:
                    action (str), code (str | None)

        Returns:
            CodeDebuggerObservation with reward, done, task_score, and feedback.
        """
        # Accept both typed model and plain dict (backwards compat)
        if isinstance(action, dict):
            action = CodeDebuggerAction(**action)

        if not self._initialized:
            return self._error_obs("Not initialized. Call reset() first.")

        if self.done:
            return self._make_obs(
                reward=0.0,
                message="Episode already finished. Call reset() to start a new task.",
            )

        self.steps_taken += 1
        self.last_action_error = None

        # Check timeout
        if self.steps_taken > self.MAX_STEPS:
            self.done = True
            return self._make_obs(
                reward=-1.0,
                message=f"Timeout: {self.MAX_STEPS} steps exceeded.",
            )

        # Dispatch
        if action.action == "submit_fix":
            return self._handle_submit(action.code or "")
        elif action.action == "run_code":
            return self._handle_run(action.code or "")
        elif action.action == "get_hint":
            return self._handle_hint()
        elif action.action == "skip":
            return self._handle_skip()
        else:
            self.last_action_error = f"Unknown action: {action.action!r}"
            return self._make_obs(
                reward=-0.1,
                message=f"Unknown action '{action.action}'. Valid: submit_fix, run_code, get_hint, skip",
            )

    @property
    def state(self) -> EnvironmentState:
        """Return current environment state (no reward/obs details)."""
        if not self._initialized or not self.current_task:
            return EnvironmentState(initialized=False)
        return EnvironmentState(
            initialized=True,
            episode_id=self.episode_id,
            task_id=self.current_task["id"],
            title=self.current_task["title"],
            difficulty=self.current_task["difficulty"],
            category=self.current_task["category"],
            steps_taken=self.steps_taken,
            max_steps=self.MAX_STEPS,
            hint_used=self.hint_used,
            test_runs=self.test_runs,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done,
            success=self.success,
        )

    # ─── Action Handlers ────────────────────────────────────────────────────

    def _handle_submit(self, code: str) -> CodeDebuggerObservation:
        if not code.strip():
            self.last_action_error = "Empty code submission"
            return self._make_obs(reward=-0.1, message="submit_fix requires non-empty code.")

        task = self.current_task
        result = grade_submission(code, task["expected_output"])

        if result["passed"]:
            # Compute final reward based on how much help was needed
            if self.hint_used:
                final_reward = 0.5
            elif self.test_runs > 0:
                final_reward = 0.8
            else:
                final_reward = 1.0

            self.done = True
            self.success = True
            self._record(final_reward)

            logger.info(
                "[Episode %s] SOLVED task=%s steps=%d reward=%.1f",
                self.episode_id[:8], task["id"], self.steps_taken, final_reward,
            )

            return self._make_obs(
                reward=final_reward,
                message=(
                    f"Correct! Task '{task['title']}' solved in {self.steps_taken} steps. "
                    f"Reward: +{final_reward:.1f}"
                ),
                extra={"actual_output": result["actual_output"]},
            )
        else:
            self._record(-0.2)
            self.last_action_error = result.get("error") or result.get("message")
            return self._make_obs(
                reward=-0.2,
                message=result["message"],
                extra={"actual_output": result["actual_output"], "error": result["error"]},
            )

    def _handle_run(self, code: str) -> CodeDebuggerObservation:
        if not code.strip():
            self.last_action_error = "Empty code"
            return self._make_obs(reward=-0.1, message="run_code requires non-empty code.")

        self.test_runs += 1
        success, stdout, stderr = run_code(code)

        if success:
            reward = 0.0
            msg = f"Code ran successfully.\nOutput:\n{stdout[:500]}"
        else:
            reward = -0.1
            self.last_action_error = stderr
            msg = f"Execution error:\n{stderr[:400]}"

        self._record(reward)
        return self._make_obs(
            reward=reward,
            message=msg,
            extra={"stdout": stdout, "stderr": stderr, "run_success": success},
        )

    def _handle_hint(self) -> CodeDebuggerObservation:
        self.hint_used = True
        self._record(-0.1)
        hint = self.current_task.get("hint", "No hint available.")
        return self._make_obs(
            reward=-0.1,
            message=f"Hint: {hint}",
            extra={"hint": hint},
        )

    def _handle_skip(self) -> CodeDebuggerObservation:
        self.done = True
        self.success = False
        self._record(-1.0)
        return self._make_obs(reward=-1.0, message="Task skipped.")

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _record(self, reward: float):
        self.reward_history.append(round(reward, 4))
        self.cumulative_reward += reward

    @staticmethod
    def _normalize_score(reward: float) -> float:
        # Map reward from [-1.0, 1.0] to [0.0, 1.0]
        base_score = (reward + 1.0) / 2.0
        # Squeeze strictly between 0 and 1 (e.g., 0.01 and 0.99)
        return round(max(0.01, min(0.99, base_score)), 4)

    def _make_obs(
        self,
        reward: float,
        message: str,
        extra: Optional[dict] = None,
    ) -> CodeDebuggerObservation:
        task = self.current_task or {}
        reward = round(reward, 4)

        # task_score: normalized 0.0-1.0 only meaningful at episode end
        if self.done and self.success:
            task_score = self._normalize_score(reward)
        elif self.done:
            task_score = 0.01  
        else:
            task_score = 0.01  

        obs = CodeDebuggerObservation(
            episode_id=self.episode_id,
            task_id=task.get("id"),
            title=task.get("title"),
            difficulty=task.get("difficulty"),
            category=task.get("category"),
            buggy_code=task.get("buggy_code"),
            description=task.get("description"),
            # Only reveal expected_output after hint
            expected_output=task.get("expected_output") if self.hint_used else None,
            steps_taken=self.steps_taken,
            max_steps=self.MAX_STEPS,
            hint_used=self.hint_used,
            test_runs=self.test_runs,
            reward=reward,
            cumulative_reward=round(self.cumulative_reward, 4),
            task_score=task_score,
            reward_history=self.reward_history.copy(),
            done=self.done,
            success=self.success,
            message=message,
            last_action_error=self.last_action_error,
        )

        if extra:
            for k, v in extra.items():
                if hasattr(obs, k):
                    setattr(obs, k, v)

        return obs

    def _error_obs(self, message: str) -> CodeDebuggerObservation:
        return CodeDebuggerObservation(
            done=True,
            success=False,
            reward=-1.0,
            task_score=0.01,
            message=message,
            last_action_error=message,
        )

    def available_tasks(self) -> list:
        """Return lightweight metadata for all tasks."""
        return get_task_metadata()
