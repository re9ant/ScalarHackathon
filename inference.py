"""
inference.py — Meta OpenEnv Hackathon Round 1

LLM Agent for the CodeDebugger RL Environment.

Required Environment Variables (per hackathon spec):
    API_BASE_URL   - LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     - Model identifier used for inference (default: gpt-4.1-mini)  
    HF_TOKEN       - Hugging Face API token (REQUIRED, no default)

Optional:
    ENV_SERVER_URL - CodeDebugger environment server URL
                     (default: https://re9ant-codedebugger-rl-env.hf.space)

Output format (per hackathon spec — must match exactly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import re
import requests
from openai import OpenAI

# ─── Required Environment Variables ────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
# Accept both HF_TOKEN (hackathon spec) and OPENAI_API_KEY (OpenAI convention)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "https://re9ant-codedebugger-rl-env.hf.space")

if HF_TOKEN is None:
    raise ValueError(
        "API key required. Set HF_TOKEN (Hugging Face) or OPENAI_API_KEY (OpenAI)."
    )

# ─── OpenAI Client (required by hackathon spec) ─────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Environment HTTP Client ───────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def env_reset(task_id=None, difficulty=None) -> dict:
    """Reset the environment and get the initial observation."""
    payload = {}
    if task_id:
        payload["task_id"] = task_id
    if difficulty:
        payload["difficulty"] = difficulty
    resp = SESSION.post(f"{ENV_SERVER_URL}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: str, code=None) -> dict:
    """Execute one action in the environment."""
    payload = {"action": action}
    if code is not None:
        payload["code"] = code
    resp = SESSION.post(f"{ENV_SERVER_URL}/step", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def env_health() -> bool:
    """Check whether the environment server is reachable."""
    try:
        resp = SESSION.get(f"{ENV_SERVER_URL}/health", timeout=15)
        return resp.status_code == 200
    except Exception:
        return False


# ─── LLM Agent ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python debugger. Your task is to fix broken Python code snippets.

You will receive:
1. A buggy code snippet
2. A description of what the code is supposed to do  
3. The expected output (if you requested a hint)

You MUST respond with a JSON object in this EXACT format:
{
  "reasoning": "Brief explanation of the bug and your fix",
  "action": "submit_fix",
  "code": "your corrected Python code here"
}

Valid actions:
- "submit_fix"  : Submit your fixed code to be graded (PREFERRED)
- "run_code"    : Test code without submitting (costs a step but no reward penalty)
- "get_hint"    : Get a hint (-0.1 reward cost)
- "skip"        : Give up on this task (-1.0 reward)

Strategy for maximum reward:
1. Read the description and buggy code carefully
2. Identify the bug (syntax, logic, runtime, or algorithm error)
3. Submit your fix immediately — first-attempt correct fix gives +1.0 reward
4. Only use get_hint if you've tried twice and failed

Common bug categories:
- Syntax: missing colons, parentheses, quotes, indentation errors
- Logic: wrong operators (= vs ==, < vs >, // vs /), wrong return variable
- Runtime: division by zero, index out of range, type mismatches, None returns
- Algorithm: wrong loop bounds, wrong base case, wrong condition order
"""


def build_prompt(obs: dict, history: list) -> str:
    """Build the user message from the current observation."""
    lines = []
    lines.append(f"Task: {obs.get('title', 'Unknown')}")
    lines.append(f"Difficulty: {obs.get('difficulty', '?')} | Category: {obs.get('category', '?')}")
    lines.append(f"Step {obs.get('steps_taken', 0)} of {obs.get('max_steps', 15)}")
    lines.append("")
    lines.append("DESCRIPTION:")
    lines.append(obs.get("description", "No description"))
    lines.append("")
    lines.append("BUGGY CODE:")
    lines.append("```python")
    lines.append(obs.get("buggy_code", ""))
    lines.append("```")

    if obs.get("expected_output"):
        lines.append("")
        lines.append("EXPECTED OUTPUT:")
        lines.append(f"```\n{obs['expected_output']}\n```")

    if history:
        lines.append("")
        lines.append("PREVIOUS ATTEMPTS:")
        for i, h in enumerate(history[-4:], 1):
            lines.append(
                f"  {i}. action={h['action']} reward={h['reward']:.2f} | {h['message'][:120]}"
            )

    lines.append("")
    lines.append("Respond with ONLY a JSON object. No markdown fences, no explanations outside JSON.")
    return "\n".join(lines)


def call_llm(messages: list) -> dict:
    """Call the LLM via the OpenAI client and parse the JSON response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()

    # Extract JSON — handle cases where model wraps in markdown fences
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback heuristics
        if "submit_fix" in raw:
            code_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
            code = code_match.group(1) if code_match else ""
            return {"action": "submit_fix", "code": code, "reasoning": "recovered"}
        if "get_hint" in raw:
            return {"action": "get_hint", "reasoning": "recovered"}
        return {"action": "get_hint", "reasoning": "JSON parse failed"}


def safe_action_str(parsed: dict) -> str:
    """
    Format action for the [STEP] line.
    Must be a single token (no spaces, no newlines).
    """
    action = parsed.get("action", "unknown")
    code = parsed.get("code") or ""
    if code:
        snippet = code.replace("\n", "\\n").replace(" ", "_").strip()[:60]
        return f"{action}('{snippet}')"
    return f"{action}()"


# ─── Run Agent ─────────────────────────────────────────────────────────────

def run_inference(task_id=None, difficulty="easy", episodes=3):
    """
    Main agent loop.

    Runs `episodes` independent episodes against the environment.
    Each episode: reset → LLM loop → [START]/[STEP]*/[END] emitted to stdout.

    Runtime target: well under 20 minutes (each episode ~3-5 LLM calls max).
    Resource usage: <100MB RAM (all state is in the env server / API calls).
    """
    # ── Wait for environment server ────────────────────────────────────────
    print("Connecting to environment...", file=sys.stderr)
    for attempt in range(20):
        if env_health():
            print(f"Environment ready at {ENV_SERVER_URL}", file=sys.stderr)
            break
        print(f"  waiting... ({attempt + 1}/20)", file=sys.stderr)
        time.sleep(3)
    else:
        print("ERROR: Environment server not responding. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── Episode loop ───────────────────────────────────────────────────────
    for ep in range(episodes):
        obs = env_reset(task_id=task_id, difficulty=difficulty)

        task_name = obs.get("task_id", "unknown")
        benchmark = "codedebugger"

        # ── [START] ──────────────────────────────────────────────────────
        print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = []
        all_rewards = []
        step_num = 0

        # ── Step loop ─────────────────────────────────────────────────────
        while not obs.get("done", False):
            step_num += 1

            # Build prompt and call LLM
            user_msg = build_prompt(obs, history)
            messages.append({"role": "user", "content": user_msg})

            try:
                parsed = call_llm(messages)
            except Exception as e:
                parsed = {"action": "get_hint", "reasoning": f"LLM error: {e}"}

            action = parsed.get("action", "skip")
            code = parsed.get("code")

            messages.append({"role": "assistant", "content": json.dumps(parsed)})

            # Take step
            try:
                obs = env_step(action=action, code=code)
            except Exception as e:
                obs = {
                    "done": True,
                    "success": False,
                    "reward": -1.0,
                    "message": str(e),
                    "last_action_error": str(e),
                }

            reward = float(obs.get("reward", 0.0))
            done = bool(obs.get("done", False))
            last_error = obs.get("last_action_error")
            error_field = last_error if last_error else "null"

            all_rewards.append(reward)
            history.append({
                "action": action,
                "reward": reward,
                "message": obs.get("message", ""),
            })

            action_str = safe_action_str(parsed)

            # ── [STEP] ────────────────────────────────────────────────────
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_field}",
                flush=True,
            )

            if done:
                break

        # ── [END] ─────────────────────────────────────────────────────────
        success = bool(obs.get("success", False))
        score = float(obs.get("task_score", 0.0))
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        rewards_csv = ",".join(f"{r:.2f}" for r in all_rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={rewards_csv}",
            flush=True,
        )


# ─── Reproducible Baseline Tasks ───────────────────────────────────────────
# These 3 tasks form the fixed benchmark for reproducible baseline scores.
# Run with `python inference.py` to get consistent results across evaluations.
BASELINE_TASKS = [
    ("syn_001", "easy"),    # Easy:   Missing colon in if statement
    ("log_001", "medium"),  # Medium: Off-by-one in range (sum 1..10)
    ("hard_001", "hard"),   # Hard:   Mutable default argument
]


# ─── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Default behaviour (no args):
      - Runs the 3 fixed baseline tasks for reproducible scores
      - Uses API_BASE_URL, MODEL_NAME, HF_TOKEN (or OPENAI_API_KEY) from env
      - Total runtime well under 20 minutes
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CodeDebugger LLM Agent — Meta OpenEnv Hackathon Round 1"
    )
    parser.add_argument(
        "--task-id", default=None,
        help="Run a specific task ID (disables baseline mode)"
    )
    parser.add_argument(
        "--difficulty", default=None,
        choices=["easy", "medium", "hard"],
        help="Filter random task selection by difficulty",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of episodes (default: 3 baseline tasks)",
    )
    args = parser.parse_args()

    if args.task_id or args.difficulty or args.episodes:
        # Custom run
        run_inference(
            task_id=args.task_id,
            difficulty=args.difficulty or "easy",
            episodes=args.episodes or 1,
        )
    else:
        # Default: reproducible baseline evaluation
        print("Running reproducible baseline evaluation (3 tasks)...", file=sys.stderr)
        for task_id, difficulty in BASELINE_TASKS:
            run_inference(task_id=task_id, difficulty=difficulty, episodes=1)
