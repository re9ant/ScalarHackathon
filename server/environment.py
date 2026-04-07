"""
RL Environment for the CodeDebugger challenge.

This environment presents an LLM agent with buggy Python code snippets
and rewards it for successfully fixing them.

Compatible with the OpenEnv interface (reset/step/state).
"""

import logging
import time
from typing import Optional
from uuid import uuid4

from .tasks import get_task, get_all_task_ids, get_task_metadata
from .grader import grade_submission, run_code

logger = logging.getLogger(__name__)


class CodeDebuggerEnvironment:
    """
    CodeDebugger RL Environment.
    
    An agent must debug broken Python code snippets.
    
    Actions:
        - submit_fix(code): Submit a fixed version of the code
        - run_code(code): Test code without submitting (costs step)  
        - get_hint(): Get a hint about the bug (costs -0.1 reward)
        - skip(): Skip the current task (costs -1.0 reward)
    
    Observation:
        - task_id, title, difficulty, category
        - buggy_code: The broken code
        - description: What the code is supposed to do
        - expected_output: What correct output should look like
        - steps_taken, hint_used, reward_so_far
        - done, success
    
    Rewards:
        +1.0  - Correct fix, no hints, first attempt
        +0.8  - Correct fix after testing but no hints
        +0.5  - Correct fix after using hint
        -0.1  - Using hint
        -0.2  - Wrong answer on submit
        -0.5  - Tested code that errored
        -1.0  - Skip or timeout
    """

    MAX_STEPS = 15

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

    # ─── OpenEnv Interface ─────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None, difficulty: Optional[str] = None, category: Optional[str] = None) -> dict:
        """
        Reset the environment with a new task.
        
        Args:
            task_id: Specific task to use (optional)
            difficulty: Filter tasks by difficulty (optional)
            category: Filter tasks by category (optional)
        
        Returns:
            Initial observation dict
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
            f"[Episode {self.episode_id[:8]}] Reset with task: {self.current_task['id']} "
            f"({self.current_task['difficulty']})"
        )

        return self._make_observation(reward=0.0, message="Environment reset. A new debugging task awaits!")

    def step(self, action: str, **kwargs) -> dict:
        """
        Execute an action in the environment.
        
        Args:
            action: One of 'submit_fix', 'run_code', 'get_hint', 'skip'
            **kwargs: Action-specific arguments
                - code (str): For 'submit_fix' and 'run_code'
        
        Returns:
            Observation dict with reward, done, error fields
        """
        if not self._initialized:
            return self._error_obs("Environment not initialized. Call reset() first.")

        if self.done:
            return self._make_observation(
                reward=0.0,
                message="Episode already finished. Call reset() to start a new task.",
            )

        self.steps_taken += 1
        self.last_action_error = None

        # Check max steps
        if self.steps_taken > self.MAX_STEPS:
            self.done = True
            self.success = False
            reward = -1.0
            self._record_reward(reward)
            return self._make_observation(
                reward=reward,
                message=f"⏰ Time limit reached ({self.MAX_STEPS} steps). Task failed.",
            )

        # Dispatch action
        if action == "submit_fix":
            return self._handle_submit(kwargs.get("code", ""))
        elif action == "run_code":
            return self._handle_run(kwargs.get("code", ""))
        elif action == "get_hint":
            return self._handle_hint()
        elif action == "skip":
            return self._handle_skip()
        else:
            self.last_action_error = f"Unknown action: '{action}'"
            return self._make_observation(
                reward=-0.1,
                message=f"Unknown action '{action}'. Valid actions: submit_fix, run_code, get_hint, skip",
            )

    @property
    def state(self) -> dict:
        """Return current environment state (no reward info)."""
        if not self._initialized or self.current_task is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "episode_id": self.episode_id,
            "task_id": self.current_task["id"],
            "title": self.current_task["title"],
            "difficulty": self.current_task["difficulty"],
            "category": self.current_task["category"],
            "steps_taken": self.steps_taken,
            "max_steps": self.MAX_STEPS,
            "hint_used": self.hint_used,
            "test_runs": self.test_runs,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "done": self.done,
            "success": self.success,
        }

    # ─── Action Handlers ───────────────────────────────────────────────────

    def _handle_submit(self, code: str) -> dict:
        """Handle submit_fix action."""
        if not code or not code.strip():
            self.last_action_error = "No code provided"
            return self._make_observation(
                reward=-0.1,
                message="submit_fix requires a 'code' argument with your fixed code.",
            )

        task = self.current_task
        result = grade_submission(code, task["expected_output"])

        if result["passed"]:
            # Compute final reward based on performance
            base_reward = 1.0
            if self.hint_used:
                base_reward = 0.5
            elif self.test_runs > 0:
                base_reward = 0.8

            self.done = True
            self.success = True
            self._record_reward(base_reward)

            logger.info(
                f"[Episode {self.episode_id[:8]}] ✅ Task solved! "
                f"Steps: {self.steps_taken}, Reward: {base_reward}"
            )

            msg = (
                f"🎉 Correct! Task '{task['title']}' solved!\n"
                f"Reward: +{base_reward:.1f} | "
                f"Steps: {self.steps_taken} | "
                f"Hints used: {self.hint_used}"
            )
            return self._make_observation(reward=base_reward, message=msg)
        else:
            reward = -0.2
            self._record_reward(reward)
            self.last_action_error = result.get("error") or result.get("message")

            logger.info(
                f"[Episode {self.episode_id[:8]}] ❌ Wrong submission. "
                f"Expected: {repr(task['expected_output'][:50])}, "
                f"Got: {repr(result['actual_output'][:50])}"
            )

            return self._make_observation(
                reward=reward,
                message=result["message"],
                extra={"actual_output": result["actual_output"], "error": result["error"]},
            )

    def _handle_run(self, code: str) -> dict:
        """Handle run_code action (test without submitting)."""
        if not code or not code.strip():
            self.last_action_error = "No code provided"
            return self._make_observation(
                reward=-0.1,
                message="run_code requires a 'code' argument.",
            )

        self.test_runs += 1
        success, stdout, stderr = run_code(code)

        if success:
            reward = 0.0
            msg = f"Code ran successfully.\nOutput:\n{stdout[:500]}"
        else:
            reward = -0.1
            self.last_action_error = stderr
            msg = f"Code execution error:\n{stderr[:500]}"

        self._record_reward(reward)

        return self._make_observation(
            reward=reward,
            message=msg,
            extra={"stdout": stdout, "stderr": stderr, "run_success": success},
        )

    def _handle_hint(self) -> dict:
        """Handle get_hint action."""
        reward = -0.1
        self.hint_used = True
        self._record_reward(reward)

        hint = self.current_task.get("hint", "No hint available for this task.")
        logger.info(f"[Episode {self.episode_id[:8]}] Hint requested (cost: {reward})")

        return self._make_observation(
            reward=reward,
            message=f"💡 Hint: {hint}",
            extra={"hint": hint},
        )

    def _handle_skip(self) -> dict:
        """Handle skip action."""
        reward = -1.0
        self.done = True
        self.success = False
        self._record_reward(reward)

        logger.info(f"[Episode {self.episode_id[:8]}] Task skipped.")

        return self._make_observation(
            reward=reward,
            message=f"⏭️ Task skipped. Reward: {reward}.",
        )

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _record_reward(self, reward: float):
        self.reward_history.append(reward)
        self.cumulative_reward += reward

    def _make_observation(self, reward: float, message: str, extra: Optional[dict] = None) -> dict:
        """Build a standard observation dict."""
        task = self.current_task or {}
        obs = {
            # Episode info
            "episode_id": self.episode_id,
            "task_id": task.get("id"),
            "title": task.get("title"),
            "difficulty": task.get("difficulty"),
            "category": task.get("category"),
            # Task content
            "buggy_code": task.get("buggy_code"),
            "description": task.get("description"),
            "expected_output": task.get("expected_output") if self.hint_used else None,
            # Step info
            "steps_taken": self.steps_taken,
            "max_steps": self.MAX_STEPS,
            "hint_used": self.hint_used,
            "test_runs": self.test_runs,
            # Reward
            "reward": round(reward, 2),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "reward_history": [round(r, 2) for r in self.reward_history],
            # Status
            "done": self.done,
            "success": self.success,
            "message": message,
            "last_action_error": self.last_action_error,
        }
        if extra:
            obs.update(extra)
        return obs

    def _error_obs(self, message: str) -> dict:
        return {
            "episode_id": None,
            "done": True,
            "success": False,
            "reward": -1.0,
            "message": message,
            "last_action_error": message,
        }

    # ─── Utility ───────────────────────────────────────────────────────────

    def available_tasks(self) -> list:
        """Return metadata for all available tasks."""
        from .tasks import get_task_metadata
        return get_task_metadata()
