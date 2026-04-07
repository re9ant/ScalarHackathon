---
title: CodeDebugger RL Environment
emoji: 🐛
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: RL env where LLM agents debug broken Python code
---

# 🐛 CodeDebugger — Meta OpenEnv Hackathon Round 1

> **A Reinforcement Learning environment where an LLM agent debugs broken Python code snippets.**

Built for the [Meta × PyTorch × Hugging Face OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon) — Round 1.

---

## 🎯 What Is This?

**CodeDebugger** is a Mini-RL environment in the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) style.

An AI agent is presented with **buggy Python code** and must fix it. Tasks span 4 categories:
- 🔴 **Syntax errors** — missing colons, parentheses, indentation
- 🟠 **Runtime errors** — division by zero, index errors, type errors  
- 🟡 **Logic errors** — wrong operators, mutated lists, wrong return values
- 🟢 **Algorithm errors** — sorting bugs, Fibonacci off-by-one, binary search bounds

The agent can:
- `submit_fix` — Submit fixed code (graded by actually running it)
- `run_code` — Test code in a sandbox first
- `get_hint` — Get a hint (costs -0.1 reward)
- `skip` — Give up (-1.0 reward)

---

## 🔌 API Reference

### `GET /health`
Server health check.

### `GET /tasks`
List all 35+ available debugging tasks with metadata.

### `POST /reset`
Start a new episode with a buggy code task.

```json
{
  "difficulty": "easy",     // "easy" | "medium" | "hard" (optional)
  "category": "syntax",     // "syntax" | "runtime" | "logic" | "algorithm" (optional)
  "task_id": "syn_001"      // specific task ID (optional)
}
```

**Response:**
```json
{
  "episode_id": "...",
  "task_id": "syn_001",
  "title": "Missing colon in if statement",
  "difficulty": "easy",
  "category": "syntax",
  "buggy_code": "x = 10\nif x > 5\n    print(...)",
  "description": "This code tries to check if x > 5 ...",
  "expected_output": null,
  "steps_taken": 0,
  "max_steps": 15,
  "done": false,
  "reward": 0.0
}
```

### `POST /step`
Take an action in the current episode.

```json
{
  "action": "submit_fix",
  "code": "x = 10\nif x > 5:\n    print('x is greater than 5')\n"
}
```

**Response (correct fix):**
```json
{
  "done": true,
  "success": true,
  "reward": 1.0,
  "message": "🎉 Correct! Task solved!"
}
```

### `GET /state`
Get current environment state.

---

## 🏆 Reward Structure

| Event | Reward |
|-------|--------|
| ✅ Correct fix (first try, no hints) | **+1.0** |
| ✅ Correct fix (after test runs) | **+0.8** |
| ✅ Correct fix (after using hint) | **+0.5** |
| ❌ Wrong submission | **-0.2** |
| 💡 Using a hint | **-0.1** |
| 🔥 Code throws an error | **-0.1** |
| ⏭️ Skip task | **-1.0** |
| ⏰ Timeout (>15 steps) | **-1.0** |

---

## 🤖 Running the LLM Agent

```bash
# Clone the repo
git clone https://github.com/re9ant/ScalarHackathon
cd ScalarHackathon

# Install deps
pip install -r requirements.txt

# Set env vars
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token"
export ENV_SERVER_URL="https://re9ant-codedebugger-rl-env.hf.space"

# Run agent
python inference.py --difficulty easy --episodes 3
```

**Output format:**
```
[START] task=syn_001 env=code-debugger model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=submit_fix(code='x = 10\nif x > 5:\n    print(...)') reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00
```

---

## 🏗️ Architecture

```
ScalarHackathon/
├── inference.py          ← LLM agent (uses OpenAI client)
├── requirements.txt      ← Python dependencies
├── Dockerfile            ← Hugging Face Spaces deployment
├── README.md             ← This file
└── server/
    ├── __init__.py
    ├── main.py           ← FastAPI server (/reset, /step, /state, /health)
    ├── environment.py    ← RL Environment class (reset/step/state)
    ├── tasks.py          ← 35+ buggy code task bank
    └── grader.py         ← Safe code execution + output comparison
```

---

## 🔒 Security

All submitted code runs in an isolated subprocess with:
- 10-second timeout
- Blocked dangerous imports (`os.system`, `subprocess`, `socket`, etc.)
- Restricted environment variables
- No network access

---

## 📜 License

MIT License — [re9ant/ScalarHackathon](https://github.com/re9ant/ScalarHackathon)
