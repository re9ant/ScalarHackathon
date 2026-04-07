"""
inference.py — Meta OpenEnv Hackathon Round 1

LLM Agent for the CodeDebugger RL Environment.

This script runs an LLM agent that:
1. Connects to the CodeDebugger environment server
2. Receives buggy Python code tasks
3. Uses an LLM to reason about and fix the bugs
4. Emits [START], [STEP], [END] lines to stdout as required

Required env vars:
    API_BASE_URL   - LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     - Model to use (default: gpt-4.1-mini)
    HF_TOKEN       - Hugging Face API token (required, no default)
    ENV_SERVER_URL - CodeDebugger server URL (default: http://localhost:7860)

Output format (per hackathon spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import re
import requests
from openai import OpenAI

# ─── Environment Variables ─────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "https://re9ant-codedebugger-rl-env.hf.space")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ─── OpenAI Client ─────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Environment Client ────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def env_reset(difficulty: str = None, task_id: str = None) -> dict:
    """Reset the environment and get initial observation."""
    payload = {}
    if difficulty:
        payload["difficulty"] = difficulty
    if task_id:
        payload["task_id"] = task_id
    resp = SESSION.post(f"{ENV_SERVER_URL}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: str, code: str = None) -> dict:
    """Take a step in the environment."""
    payload = {"action": action}
    if code is not None:
        payload["code"] = code
    resp = SESSION.post(f"{ENV_SERVER_URL}/step", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def env_health() -> bool:
    """Check if the environment server is running."""
    try:
        resp = SESSION.get(f"{ENV_SERVER_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

# ─── LLM Agent ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python debugger. Your job is to fix broken Python code snippets.

You will be given:
1. A buggy code snippet
2. A description of what the code should do
3. The expected output (if you've requested a hint)

You must respond with a JSON object in EXACTLY this format:
{
  "reasoning": "Brief explanation of what's wrong and how to fix it",
  "action": "submit_fix" | "run_code" | "get_hint" | "skip",
  "code": "your fixed python code here"  // only for submit_fix and run_code
}

Strategy:
- ALWAYS try to fix the bug directly first (submit_fix) — you get higher reward if you don't need hints
- If you're unsure, use run_code to test your hypothesis first
- Only use get_hint if you're completely stuck after 2+ failed attempts
- Never skip unless completely impossible

Common bug types to look for:
- Syntax errors: missing colons, parentheses, quotes, indentation
- Logic errors: wrong operators (= vs ==, < vs >, // vs /), wrong variable returned
- Runtime errors: division by zero, index out of range, type mismatches
- Algorithm errors: wrong bounds, wrong base cases, wrong conditions
"""


def build_user_message(obs: dict, history: list) -> str:
    """Build the user message for the LLM from current observation."""
    parts = []
    
    parts.append(f"**Task: {obs.get('title', 'Unknown')}**")
    parts.append(f"Difficulty: {obs.get('difficulty', '?')} | Category: {obs.get('category', '?')}")
    parts.append(f"Steps taken: {obs.get('steps_taken', 0)}/{obs.get('max_steps', 15)}")
    parts.append("")
    
    parts.append("**What this code should do:**")
    parts.append(obs.get("description", "No description available"))
    parts.append("")
    
    parts.append("**Buggy code to fix:**")
    parts.append("```python")
    parts.append(obs.get("buggy_code", ""))
    parts.append("```")
    
    if obs.get("expected_output"):
        parts.append("")
        parts.append("**Expected output:**")
        parts.append(f"```\n{obs['expected_output']}\n```")
    
    if history:
        parts.append("")
        parts.append("**Previous attempts:**")
        for i, h in enumerate(history[-3:], 1):  # last 3 attempts
            parts.append(f"{i}. Action: {h['action']} | Reward: {h['reward']:.2f} | Result: {h['message'][:100]}")
    
    if obs.get("message") and obs.get("steps_taken", 0) > 0:
        parts.append("")
        parts.append(f"**Last result:** {obs['message'][:200]}")
    
    parts.append("")
    parts.append("Respond with a JSON object only. No markdown, no extra text.")
    
    return "\n".join(parts)


def call_llm(messages: list) -> dict:
    """Call the LLM and parse the JSON response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Try to extract JSON from the response
    # Sometimes models wrap it in markdown code blocks
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract action and code manually
        if "submit_fix" in raw:
            code_match = re.search(r'```python\n(.*?)```', raw, re.DOTALL)
            code = code_match.group(1) if code_match else ""
            return {"action": "submit_fix", "code": code, "reasoning": "Extracted from malformed response"}
        return {"action": "get_hint", "reasoning": "Failed to parse LLM response"}


def format_action_str(parsed: dict) -> str:
    """Format action for [STEP] output."""
    action = parsed.get("action", "unknown")
    code = parsed.get("code", "")
    if code:
        # Truncate and escape for single-line output
        code_snippet = code.replace("\n", "\\n").strip()[:80]
        return f"{action}(code='{code_snippet}...')" if len(code) > 80 else f"{action}(code='{code_snippet}')"
    return f"{action}()"


# ─── Main Agent Loop ───────────────────────────────────────────────────────

def run_agent(task_id: str = None, difficulty: str = "medium", max_episodes: int = 1):
    """
    Run the LLM agent for one or more episodes.
    
    Args:
        task_id: Specific task to run (None = random)
        difficulty: Task difficulty if random
        max_episodes: Number of episodes to run
    """
    # Wait for server to be ready
    for attempt in range(30):
        if env_health():
            break
        print(f"Waiting for environment server... ({attempt+1}/30)", file=sys.stderr)
        time.sleep(2)
    else:
        print("ERROR: Environment server not responding after 60s", file=sys.stderr)
        sys.exit(1)

    for episode in range(max_episodes):
        # ── Reset environment ──────────────────────────────────────────────
        obs = env_reset(difficulty=difficulty, task_id=task_id)
        
        task_name = obs.get("task_id", "unknown-task")
        benchmark = "code-debugger"
        
        # Emit [START]
        print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}", flush=True)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = []
        all_rewards = []
        step_num = 0
        
        # ── Agent loop ────────────────────────────────────────────────────
        while not obs.get("done", False):
            step_num += 1
            
            # Build user message
            user_msg = build_user_message(obs, history)
            messages.append({"role": "user", "content": user_msg})
            
            # Call LLM
            try:
                parsed = call_llm(messages)
            except Exception as e:
                parsed = {"action": "get_hint", "reasoning": f"LLM error: {e}"}
            
            action = parsed.get("action", "skip")
            code = parsed.get("code")
            reasoning = parsed.get("reasoning", "")
            
            # Add assistant reasoning to conversation history
            messages.append({
                "role": "assistant",
                "content": json.dumps(parsed),
            })
            
            # Take step in environment
            try:
                obs = env_step(action=action, code=code)
            except Exception as e:
                obs = {
                    "done": True,
                    "success": False,
                    "reward": -1.0,
                    "message": f"Server error: {e}",
                    "last_action_error": str(e),
                }
            
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            error = obs.get("last_action_error") or "null"
            
            all_rewards.append(reward)
            history.append({
                "action": action,
                "reward": reward,
                "message": obs.get("message", ""),
            })
            
            action_str = format_action_str(parsed)
            
            # Emit [STEP]
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error}",
                flush=True,
            )
            
            if done:
                break
        
        # ── Episode end ───────────────────────────────────────────────────
        success = obs.get("success", False)
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        
        # Emit [END]
        print(
            f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
            flush=True,
        )


# ─── Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodeDebugger LLM Agent")
    parser.add_argument("--task-id", default=None, help="Specific task ID to run")
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Task difficulty (default: medium)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run (default: 1)"
    )
    
    args = parser.parse_args()
    
    run_agent(
        task_id=args.task_id,
        difficulty=args.difficulty,
        max_episodes=args.episodes,
    )
