"""
Pre-Submission Checklist — Meta OpenEnv Hackathon Round 1
Runs all mandatory checks before submission.
"""
import sys, os, re, subprocess, requests, time

sys.stdout.reconfigure(encoding='utf-8')

SPACE_URL = "https://re9ant-codedebugger-rl-env.hf.space"
GITHUB_REPO = "https://github.com/re9ant/ScalarHackathon"
HF_SPACE = "https://huggingface.co/spaces/re9ant/CodeDebugger-RL-Env"

checks_passed = 0
checks_failed = 0

def check(name, ok, detail=""):
    global checks_passed, checks_failed
    status = "PASS" if ok else "FAIL"
    sym = "✅" if ok else "❌"
    print(f"  {sym} [{status}] {name}")
    if detail:
        print(f"         {detail}")
    if ok:
        checks_passed += 1
    else:
        checks_failed += 1

print("=" * 60)
print("PRE-SUBMISSION CHECKLIST")
print("=" * 60)

# ── 1. File structure ──────────────────────────────────────────────────────
print("\n[1] File Structure")
check("inference.py in root", os.path.exists("inference.py"))
check("Dockerfile in root", os.path.exists("Dockerfile"))
check("requirements.txt in root", os.path.exists("requirements.txt"))
check("openenv.yaml in root", os.path.exists("openenv.yaml"))
check("README.md in root", os.path.exists("README.md"))
check("server/ directory exists", os.path.isdir("server"))

# ── 2. inference.py content checks ──────────────────────────────────────────
print("\n[2] inference.py Compliance")
with open("inference.py") as f:
    src = f.read()

check("Uses OpenAI client", "from openai import OpenAI" in src)
check("API_BASE_URL env var with default", 'os.getenv("API_BASE_URL"' in src and 'https://api.openai.com/v1' in src)
check("MODEL_NAME env var with default", 'os.getenv("MODEL_NAME"' in src and 'gpt-4.1-mini' in src)
check("HF_TOKEN env var (no default)", 'os.getenv("HF_TOKEN")' in src)
check("HF_TOKEN None check", 'HF_TOKEN is None' in src)
check("[START] format", '[START]' in src and 'task=' in src and 'env=' in src and 'model=' in src)
check("[STEP] format", '[STEP]' in src and 'step=' in src and 'action=' in src and 'reward=' in src and 'done=' in src and 'error=' in src)
check("[END] format", '[END]' in src and 'success=' in src and 'steps=' in src and 'rewards=' in src)

# ── 3. openenv.yaml checks ──────────────────────────────────────────────────
print("\n[3] openenv.yaml")
with open("openenv.yaml") as f:
    yaml_src = f.read()
check("spec_version field", "spec_version" in yaml_src)
check("name field", "name" in yaml_src)
check("port: 7860", "7860" in yaml_src)

# ── 4. Dockerfile checks ──────────────────────────────────────────────────
print("\n[4] Dockerfile")
with open("Dockerfile") as f:
    df = f.read()
check("Port 7860 exposed", "7860" in df)
check("python:3.11 base image", "python:3.11" in df)
check("requirements.txt install", "requirements.txt" in df)
check("CMD uvicorn", "uvicorn" in df)

# ── 5. Live HF Space checks ─────────────────────────────────────────────────
print("\n[5] Live HF Space")
try:
    h = requests.get(f"{SPACE_URL}/health", timeout=20).json()
    check("Space /health returns 200", h.get("status") == "ok", f"status={h.get('status')}")
    check("Environment initialized", h.get("initialized") == True)
except Exception as e:
    check("Space /health reachable", False, str(e))

try:
    t = requests.get(f"{SPACE_URL}/tasks", timeout=15).json()
    check("3+ tasks available (have 36)", t.get("total", 0) >= 3, f"total={t.get('total')}")
except Exception as e:
    check("/tasks endpoint", False, str(e))

try:
    obs = requests.post(f"{SPACE_URL}/reset", json={"difficulty": "easy"}, timeout=15).json()
    check("/reset endpoint works", "task_id" in obs and "buggy_code" in obs, f"task={obs.get('task_id')}")
except Exception as e:
    check("/reset endpoint", False, str(e))

try:
    s = requests.post(f"{SPACE_URL}/step", json={"action": "get_hint"}, timeout=15).json()
    check("/step endpoint works", "reward" in s and "done" in s, f"reward={s.get('reward')}")
    r = s.get("reward", 0)
    check("Reward in [-1.0, 1.0] range", -1.0 <= r <= 1.0, f"reward={r}")
except Exception as e:
    check("/step endpoint", False, str(e))

try:
    st = requests.get(f"{SPACE_URL}/state", timeout=10).json()
    check("/state endpoint works", "steps_taken" in st and "done" in st)
except Exception as e:
    check("/state endpoint", False, str(e))

# ── 6. Task grader validation ──────────────────────────────────────────────
print("\n[6] Task Grader Validation (all 36 solutions)")
from server.tasks import TASKS
from server.grader import grade_submission

grader_pass = 0
grader_fail = []
for task in TASKS:
    r = grade_submission(task["solution"], task["expected_output"])
    if r["passed"]:
        grader_pass += 1
    else:
        grader_fail.append(task["id"])

check(f"All solutions grade correctly ({grader_pass}/36)", len(grader_fail) == 0,
      f"Failed: {grader_fail}" if grader_fail else "")

# ── 7. Output format regex validation ──────────────────────────────────────
print("\n[7] Output Format")
start_re = re.compile(r'^\[START\] task=\S+ env=\S+ model=\S+$')
step_re  = re.compile(r'^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=\S+$')
end_re   = re.compile(r'^\[END\] success=(true|false) steps=\d+ rewards=(-?\d+\.\d{2},?)+$')

check("[START] regex", bool(start_re.match("[START] task=syn_001 env=codedebugger model=gpt-4.1-mini")))
check("[STEP] regex",  bool(step_re.match("[STEP] step=1 action=get_hint() reward=-0.10 done=false error=null")))
check("[END] regex",   bool(end_re.match("[END] success=true steps=2 rewards=-0.10,1.00")))

# ── Summary ───────────────────────────────────────────────────────────────
total = checks_passed + checks_failed
print(f"\n{'='*60}")
print(f"RESULTS: {checks_passed}/{total} checks passed")
if checks_failed:
    print(f"FAILED:  {checks_failed} checks need attention")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Ready to submit!")
    print(f"\nSubmit this HF Space URL: {HF_SPACE}")
