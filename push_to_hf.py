"""Upload the support_triage benchmark to a Hugging Face Docker Space."""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = os.getenv("HF_SPACE_REPO_ID", "shubham47404/support-triage-env")
REPO_TYPE = "space"
SPACE_SDK = "docker"

EXCLUDE_PATTERNS = {
    ".git",
    ".pytest_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "*.log",
    "*.out",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "debug.log",
    "demo_output.txt",
    "groq_out.txt",
    "groq_output.log",
    "push_to_hf.py",
    "validate_out*.txt",
}


def should_exclude(path: Path) -> bool:
    path_str = path.as_posix()
    for part in path.parts:
        if part in {".git", ".venv", "__pycache__", ".pytest_cache", ".uv-cache"}:
            return True
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern) for pattern in EXCLUDE_PATTERNS)


def main() -> None:
    root = Path(".").resolve()
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as exc:
        print(f"ERROR: Hugging Face authentication failed. Set HF_TOKEN or run huggingface-cli login.\n{exc}")
        sys.exit(1)

    try:
        repo = create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            exist_ok=True,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Space ready: {repo}")
    except Exception as exc:
        print(f"ERROR creating Space: {exc}")
        sys.exit(1)

    files_to_upload = sorted(
        path for path in root.rglob("*") if path.is_file() and not should_exclude(path.relative_to(root))
    )

    if not any(path.name == "Dockerfile" and path.parent == root for path in files_to_upload):
        print("ERROR: root Dockerfile is required for Hugging Face Docker Spaces.")
        sys.exit(1)

    print(f"Uploading {len(files_to_upload)} files to {REPO_ID}...")

    for index, path in enumerate(files_to_upload, start=1):
        relative_path = path.relative_to(root).as_posix()
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=relative_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"[{index}/{len(files_to_upload)}] Uploaded: {relative_path}")
        except Exception as exc:
            print(f"[{index}/{len(files_to_upload)}] FAILED: {relative_path} -> {exc}")
            sys.exit(1)

    print(f"SUCCESS: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
