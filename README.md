---
title: CodeDebugger RL Environment
emoji: 🐛
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: RL env where LLM agents debug broken Python code
tags:
  - openenv
  - reinforcement-learning
  - code-debugging
  - python
---

# 🐛 CodeDebugger — Python Debugging RL Environment

> **A real-world Reinforcement Learning environment for the [Meta × PyTorch × Hugging Face OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)**

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-spec%20compliant-purple)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HF%20Space-Running-green)](https://huggingface.co/spaces/re9ant/CodeDebugger-RL-Env)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](Dockerfile)

---

## 🎯 Environment Description & Motivation

**CodeDebugger** trains AI agents to fix broken Python code — one of the highest-value tasks in software engineering.

Every developer spends significant time debugging: finding and fixing syntax errors, logic bugs, runtime crashes, and algorithm mistakes. This environment:

1. **Simulates real debugging work** — not a game or toy. Tasks are drawn from real common bug patterns.
2. **Provides programmatic grading** — code is actually executed; output is compared to expected.
3. **Challenges frontier models** — hard tasks (mutable defaults, closure captures, shallow copies) trip up even GPT-4.
4. **Teaches cautious reasoning** — agents learn to test before submitting, use hints sparingly, and leverage partial credit.

---

## 📐 Action Space

Actions are defined by the `CodeDebuggerAction` Pydantic model:

```python
class CodeDebuggerAction(BaseModel):
    action: Literal["submit_fix", "run_code", "get_hint", "skip"]
    code: Optional[str] = None  # required for submit_fix and run_code
```

| Action | Code Required | Effect | Reward |
|--------|:---:|--------|--------|
| `submit_fix` | ✅ | Runs code against grader, ends episode if correct | +1.0 / -0.2 |
| `run_code` | ✅ | Executes code in sandbox, shows stdout/stderr | 0.0 / -0.1 |
| `get_hint` | ❌ | Reveals a hint + expected output | -0.1 |
| `skip` | ❌ | Abandons task (episode ends) | -1.0 |

---

## 👁️ Observation Space

Observations are defined by the `CodeDebuggerObservation` Pydantic model:

```python
class CodeDebuggerObservation(BaseModel):
    # Task identity
    episode_id: str
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal["syntax", "runtime", "logic", "algorithm"]

    # Task content (what the agent sees)
    buggy_code: str          # The broken code to fix
    description: str         # What the code should do
    expected_output: str     # Revealed only after get_hint

    # Progress tracking
    steps_taken: int         # Steps used (max 15)
    max_steps: int           # = 15
    hint_used: bool
    test_runs: int

    # Reward signals
    reward: float            # Immediate reward ∈ [-1.0, 1.0]
    cumulative_reward: float
    task_score: float        # Final grader score ∈ [0.0, 1.0] (done=True only)
    reward_history: List[float]

    # Status
    done: bool
    success: bool
    message: str
    last_action_error: Optional[str]
```

---

## 🏆 Reward Function

The reward function provides **dense signal throughout the episode**, not just sparse terminal rewards.

### Intermediate rewards (training signal)

| Event | Reward | Rationale |
|-------|--------|-----------|
| get_hint() | -0.1 | Penalizes hint dependency |
| run_code() — success | 0.0 | Neutral (exploration encouraged) |
| run_code() — error | -0.1 | Minor penalty for bad code |
| submit_fix() — wrong | -0.2 | Penalizes careless guessing |
| skip() | -1.0 | Strong penalty for giving up |
| Timeout (>15 steps) | -1.0 | Prevents infinite loops |

### Terminal rewards (end of episode)

| Achievement | Reward | Task Score (0–1) |
|-------------|--------|:-:|
| Correct fix, no hints, no tests | **+1.0** | **1.00** |
| Correct fix, after test runs | **+0.8** | **0.90** |
| Correct fix, after hint | **+0.5** | **0.75** |
| Failed (wrong, skip, timeout) | ≤ 0.0 | **0.00** |

---

## 📋 Task Descriptions

### Easy Tasks (12 tasks)

| ID | Title | Category |
|----|-------|----------|
| `syn_001` | Missing colon in if statement | syntax |
| `syn_002` | Missing closing parenthesis | syntax |
| `syn_003` | Wrong indentation | syntax |
| `syn_004` | Missing quotes in string | syntax |
| `syn_005` | Missing colon in function def | syntax |
| `run_001` | Division by zero | runtime |
| `run_002` | Index out of range | runtime |
| `run_003` | Key error in dictionary | runtime |
| `run_004` | Type error: str + int | runtime |
| `run_005` | NoneType attribute error | runtime |
| `run_006` | Unpack too many values | runtime |
| `alg_006` | FizzBuzz wrong conditions | algorithm |

### Medium Tasks (15 tasks)

| ID | Title | Category |
|----|-------|----------|
| `log_001` | Off-by-one in range | logic |
| `log_002` | Wrong comparison operator (= vs ==) | logic |
| `log_003` | Swapped while loop condition | logic |
| `log_004` | Wrong variable in return | logic |
| `log_005` | String comparison case sensitivity | logic |
| `log_006` | List mutation in loop | logic |
| `log_007` | Accumulator not reset in loop | logic |
| `log_008` | Integer division instead of float | logic |
| `log_009` | Print inside function, not returned | logic |
| `log_010` | String immutability misunderstanding | logic |
| `alg_001` | Bubble sort swaps in wrong direction | algorithm |
| `alg_002` | Fibonacci off-by-one in range | algorithm |
| `alg_003` | Binary search wrong bounds | algorithm |
| `alg_005` | Palindrome check off-by-one | algorithm |
| `alg_007` | Count occurrences wrong logic | algorithm |

### Hard Tasks (9 tasks)

| ID | Title | Category |
|----|-------|----------|
| `hard_001` | **Mutable default argument** | algorithm |
| `hard_002` | Closure variable capture bug | algorithm |
| `hard_003` | Shallow copy vs deep copy | algorithm |
| `hard_004` | Generator exhaustion | algorithm |
| `hard_005` | Integer identity vs equality | algorithm |
| `hard_006` | Dict comprehension key collision | algorithm |
| `hard_007` | Missing return in recursive branch | algorithm |
| `hard_008` | Thread-unsafe counter | algorithm |
| `alg_004` | Factorial wrong base case | algorithm |

---

## 📊 Baseline Scores

Reproducible baseline run on the 3 fixed benchmark tasks using `meta-llama/Llama-3.1-8B-Instruct`:

| Task | Difficulty | Expected | Score |
|------|-----------|----------|-------|
| `syn_001` — Missing colon | Easy | 1.00 | **1.00** |
| `log_001` — Off-by-one | Medium | 0.75–1.00 | **0.90** |
| `hard_001` — Mutable default arg | Hard | 0.50–0.75 | **0.75** |

*Run with `python inference.py` — fixed tasks, reproducible across runs.*

---

## 🔌 API Reference

Base URL: `https://re9ant-codedebugger-rl-env.hf.space`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status |
| GET | `/tasks` | List all 36 tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take action (typed body) |
| GET | `/state` | Current environment state |
| GET | `/docs` | Swagger UI |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone
git clone https://github.com/re9ant/ScalarHackathon
cd ScalarHackathon

# Install dependencies
pip install -r requirements.txt

# Start environment server
uvicorn server.main:app --host 0.0.0.0 --port 7860

# Run agent (in another terminal)
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py
```

### Docker

```bash
docker build -t codedebugger-rl-env .
docker run -p 7860:7860 codedebugger-rl-env

# Test health
curl http://localhost:7860/health
```

### Run the inference agent

```bash
# Required environment variables
export API_BASE_URL=https://api-inference.huggingface.co/v1  # or API base URL
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct          # model identifier
export HF_TOKEN=your_huggingface_token                        # API key

# Default: 3 episodes, easy difficulty, reproducible
python inference.py

# Custom run
python inference.py --difficulty hard --episodes 5
python inference.py --task-id hard_001  # specific task
```

**Output format:**
```
[START] task=syn_001 env=codedebugger model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=submit_fix('x=10\nif_x>5:\n_print(...)') reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00
```

---

## 🏗️ Architecture

```
ScalarHackathon/
├── inference.py          ← LLM agent (OpenAI client, [START]/[STEP]/[END] output)
├── requirements.txt      ← Python dependencies
├── Dockerfile            ← Port 7860, python:3.11-slim
├── openenv.yaml          ← OpenEnv spec metadata
├── README.md             ← This file
└── server/
    ├── __init__.py
    ├── models.py         ← Typed Pydantic: Action, Observation, State
    ├── main.py           ← FastAPI server with OpenEnv endpoints
    ├── environment.py    ← RL environment: reset()/step()/state
    ├── tasks.py          ← 36 buggy code task bank
    └── grader.py         ← Sandboxed execution + output comparison
```

---

## 🔒 Security

All submitted code executes in an isolated subprocess with:
- **10-second timeout** — prevents infinite loops
- **Blocked imports** — `os.system`, `subprocess`, `socket`, `requests`, etc.
- **No network access** — restricted env vars
- **Temp file cleanup** — no persistent state

---

## 📜 License

MIT — [re9ant/ScalarHackathon](https://github.com/re9ant/ScalarHackathon)
