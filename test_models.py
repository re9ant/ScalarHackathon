"""Test typed models and environment compliance."""
import sys

from server.models import CodeDebuggerAction, CodeDebuggerObservation, EnvironmentState, TaskMetadata
from server.environment import CodeDebuggerEnvironment
from server.tasks import TASKS

print("=== Testing Typed Pydantic Models ===\n")

env = CodeDebuggerEnvironment()

# Test 1: reset() returns typed model
obs = env.reset(task_id="syn_001")
assert isinstance(obs, CodeDebuggerObservation), f"Expected Observation, got {type(obs)}"
assert obs.task_id == "syn_001"
assert obs.task_score == 0.0
print(f"  OK reset() -> {type(obs).__name__}: task={obs.task_id}, task_score={obs.task_score}")

# Test 2: step() with typed action
act = CodeDebuggerAction(action="get_hint")
obs2 = env.step(act)
assert isinstance(obs2, CodeDebuggerObservation)
assert obs2.reward == -0.1
assert obs2.hint_used == True
print(f"  OK step(typed_action): reward={obs2.reward}, hint_used={obs2.hint_used}")

# Test 3: step() with dict (backwards compat)
obs3 = env.step({"action": "run_code", "code": "print(42)"})
assert isinstance(obs3, CodeDebuggerObservation)
print(f"  OK step(dict): reward={obs3.reward}, run_success={obs3.run_success}")

# Test 4: correct submission, no hints -> reward=1.0, task_score=1.0
env2 = CodeDebuggerEnvironment()
env2.reset(task_id="syn_001")
task = next(t for t in TASKS if t["id"] == "syn_001")
obs4 = env2.step(CodeDebuggerAction(action="submit_fix", code=task["solution"]))
assert obs4.success == True
assert obs4.done == True
assert obs4.reward == 1.0, f"Expected 1.0, got {obs4.reward}"
assert obs4.task_score == 1.0, f"Expected task_score=1.0, got {obs4.task_score}"
print(f"  OK correct_submit (no hints): reward={obs4.reward}, task_score={obs4.task_score}")

# Test 5: after hint -> reward=0.5, task_score=0.75
env3 = CodeDebuggerEnvironment()
env3.reset(task_id="log_001")
task2 = next(t for t in TASKS if t["id"] == "log_001")
env3.step(CodeDebuggerAction(action="get_hint"))
obs5 = env3.step(CodeDebuggerAction(action="submit_fix", code=task2["solution"]))
assert obs5.success == True
assert obs5.reward == 0.5, f"Expected 0.5 after hint, got {obs5.reward}"
assert obs5.task_score == 0.75, f"Expected 0.75, got {obs5.task_score}"
print(f"  OK after_hint: reward={obs5.reward}, task_score={obs5.task_score}")

# Test 6: after run_code -> reward=0.8, task_score=0.9
env4 = CodeDebuggerEnvironment()
env4.reset(task_id="alg_001")
task3 = next(t for t in TASKS if t["id"] == "alg_001")
env4.step(CodeDebuggerAction(action="run_code", code=task3["solution"]))
obs6 = env4.step(CodeDebuggerAction(action="submit_fix", code=task3["solution"]))
assert obs6.success == True
assert obs6.reward == 0.8, f"Expected 0.8, got {obs6.reward}"
assert obs6.task_score == 0.9, f"Expected 0.9, got {obs6.task_score}"
print(f"  OK after_run: reward={obs6.reward}, task_score={obs6.task_score}")

# Test 7: state() returns typed EnvironmentState
st = env2.state
assert isinstance(st, EnvironmentState), f"Expected EnvironmentState, got {type(st)}"
assert st.done == True
print(f"  OK state() -> {type(st).__name__}: done={st.done}, success={st.success}")

# Test 8: task_score always 0.0-1.0
env5 = CodeDebuggerEnvironment()
env5.reset(task_id="syn_002")
o = env5.step(CodeDebuggerAction(action="submit_fix", code="print('wrong')"))
assert 0.0 <= o.task_score <= 1.0, f"task_score out of range: {o.task_score}"
env5.step(CodeDebuggerAction(action="skip"))
assert 0.0 <= env5.step(CodeDebuggerAction(action="skip")).task_score <= 1.0
print(f"  OK task_score range: always in [0.0, 1.0]")

# Test 9: Action validation (Pydantic catches bad action)
try:
    bad = CodeDebuggerAction(action="invalid_action")
    print(f"  FAIL: should have rejected invalid action")
except Exception as e:
    print(f"  OK action_validation: Pydantic rejects 'invalid_action' correctly")

print(f"\n=== ALL TYPED MODEL TESTS PASSED ===")
