"""
Validate ALL task solutions through the grader.
Every solution must:
  - Execute without error
  - Produce output matching expected_output exactly
  - Return reward=1.0 (correct on first submit)
Also checks rewards are in valid ranges.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from server.tasks import TASKS
from server.grader import grade_submission

print("=== Validating all task solutions ===\n")

passed = []
failed = []

for task in TASKS:
    task_id = task["id"]
    solution = task.get("solution", "")
    expected = task.get("expected_output", "")
    
    if not solution:
        failed.append((task_id, "NO SOLUTION DEFINED"))
        continue
    
    result = grade_submission(solution, expected)
    
    if result["passed"]:
        passed.append(task_id)
        print(f"  OK  {task_id:12s}  reward={result['reward']:.1f}  ({task['title'][:45]})")
    else:
        failed.append((task_id, result.get("error") or result.get("message", "?")[:80]))
        print(f"  FAIL {task_id:12s}  ({task['title'][:45]})")
        print(f"       error: {result.get('error') or result.get('message','?')[:100]}")
        print(f"       expected: {repr(expected[:60])}")
        print(f"       got:      {repr(result.get('actual_output','')[:60])}")

print(f"\n{'='*60}")
print(f"Results: {len(passed)}/{len(TASKS)} tasks passed")

if failed:
    print(f"\nFAILED ({len(failed)}):")
    for tid, reason in failed:
        print(f"  - {tid}: {reason[:80]}")
    sys.exit(1)
else:
    print("\nAll solutions produce correct output!")

# Also check reward ranges
print("\n=== Checking reward ranges ===")
from server.environment import CodeDebuggerEnvironment

reward_issues = []
for task in TASKS[:5]:  # spot-check 5 tasks
    env = CodeDebuggerEnvironment()
    obs = env.reset(task_id=task["id"])
    
    # hint
    o = env.step("get_hint")
    if not (-1.0 <= o["reward"] <= 1.0):
        reward_issues.append(f"{task['id']}: hint reward {o['reward']} out of range")
    
    # submit correct
    env2 = CodeDebuggerEnvironment()
    env2.reset(task_id=task["id"])
    o2 = env2.step("submit_fix", code=task["solution"])
    if not (-1.0 <= o2["reward"] <= 1.0):
        reward_issues.append(f"{task['id']}: submit reward {o2['reward']} out of range")

if reward_issues:
    print("REWARD ISSUES:")
    for r in reward_issues:
        print(f"  {r}")
else:
    print("All sampled rewards in [-1.0, 1.0] range OK")

print("\n=== VALIDATION COMPLETE ===")
