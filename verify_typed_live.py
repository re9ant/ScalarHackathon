"""Verify live HF Space has the new typed model responses."""
import requests, time

base = "https://re9ant-codedebugger-rl-env.hf.space"
print("Checking Space with typed models...")

for i in range(20):
    try:
        r = requests.get(f"{base}/health", timeout=15)
        if r.status_code == 200:
            print(f"  Space healthy: {r.json()}")
            break
    except Exception as e:
        print(f"  [{i+1}/20] waiting... ({e})")
        time.sleep(10)

obs = requests.post(f"{base}/reset", json={"task_id": "syn_001"}, timeout=20).json()
assert obs.get("task_score") == 0.0
s = requests.post(f"{base}/step", json={"action": "get_hint"}, timeout=15).json()
assert s["reward"] == -0.1
r2 = requests.post(f"{base}/step", json={"action": "invalid_action"}, timeout=10)
assert r2.status_code == 422
print("  task_score field OK, typed action validation OK")
print("LIVE VERIFICATION PASSED")
