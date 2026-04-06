"""
Fix: Delete the invalid uv.lock from HF Space so Docker can build cleanly.
Run: python fix_uvlock.py
"""
from huggingface_hub import HfApi

REPO_ID = "shubham47404/support-triage-env"

api = HfApi()

try:
    user = api.whoami()
    print(f"Authenticated as: {user['name']}")
except Exception as e:
    print(f"Not authenticated: {e}")
    exit(1)

try:
    api.delete_file(
        path_in_repo="uv.lock",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Remove invalid uv.lock to allow clean Docker build",
    )
    print("Deleted uv.lock from Space.")
except Exception as e:
    print(f"Could not delete uv.lock (may already be gone): {e}")

print("\nDone! Docker will now rebuild automatically.")
print(f"Watch the build at: https://huggingface.co/spaces/{REPO_ID}")
