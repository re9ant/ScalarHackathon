"""
deploy_to_hf.py — Deploy CodeDebugger to Hugging Face Spaces

Usage:
    python deploy_to_hf.py --token YOUR_HF_TOKEN --username YOUR_HF_USERNAME

This script:
1. Creates a Hugging Face Space (or updates existing)
2. Pushes all project files to the Space
3. Checks the Space is in Running state
"""

import argparse
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, SpaceRuntime


def deploy(token: str, username: str, space_name: str = "CodeDebugger-RL-Env"):
    api = HfApi(token=token)
    repo_id = f"{username}/{space_name}"
    
    print(f"🚀 Deploying to: https://huggingface.co/spaces/{repo_id}")

    # ── Create or verify Space ──────────────────────────────────────────────
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"✅ Space already exists: {repo_id}")
    except Exception:
        print(f"📦 Creating new Space: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print(f"✅ Space created!")

    # ── Upload files ────────────────────────────────────────────────────────
    root = Path(__file__).parent

    files_to_upload = [
        "README.md",
        "Dockerfile",
        "requirements.txt",
        "app.py",
        "inference.py",
        "server/__init__.py",
        "server/models.py",
        "server/main.py",
        "server/environment.py",
        "server/tasks.py",
        "server/grader.py",
        "pyproject.toml",
        "openenv.yaml",
        "uv.lock",
        "server/app.py"
    ]

    print("\n📁 Uploading files...")
    for rel_path in files_to_upload:
        local_path = root / rel_path
        if not local_path.exists():
            print(f"  ⚠️  Skipping (not found): {rel_path}")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"  ✅ {rel_path}")

    print("\n⏳ Waiting for Space to build and start (this can take 2-5 minutes)...")

    # ── Wait for Running state ───────────────────────────────────────────────
    space_url = f"https://{username.lower()}-{space_name.lower().replace('_', '-')}.hf.space"
    
    for attempt in range(30):
        time.sleep(10)
        try:
            runtime = api.get_space_runtime(repo_id=repo_id)
            stage = runtime.stage
            print(f"  [{attempt+1}/30] Stage: {stage}")
            if stage == "RUNNING":
                print(f"\n🎉 Space is RUNNING!")
                print(f"   URL: {space_url}")
                print(f"   Health: {space_url}/health")
                print(f"   Docs:   {space_url}/docs")
                return space_url
            elif stage in ("BUILD_ERROR", "RUNTIME_ERROR"):
                print(f"\n❌ Space failed with stage: {stage}")
                print(f"   Check logs at: https://huggingface.co/spaces/{repo_id}/logs")
                sys.exit(1)
        except Exception as e:
            print(f"  [{attempt+1}/30] Checking... ({e})")

    print(f"\n⚠️  Timeout waiting for space. Check manually:")
    print(f"   https://huggingface.co/spaces/{repo_id}")
    return space_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Your Hugging Face write token (from hf.co/settings/tokens)")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--space-name", default="CodeDebugger-RL-Env", help="Space name (default: CodeDebugger-RL-Env)")
    args = parser.parse_args()

    url = deploy(args.token, args.username, args.space_name)
    print(f"\n✅ Done! Your space URL: {url}")
    print(f"\nUpdate ENV_SERVER_URL in inference.py to: {url}")
    print(f"Then run: python inference.py --difficulty easy")
