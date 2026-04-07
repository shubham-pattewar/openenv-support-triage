"""
Helper script to push the support_triage environment to Hugging Face Spaces.
Run this from the support_triage directory:
    python push_to_hf.py
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "shubham47404/support-triage-env"
REPO_TYPE = "space"
SPACE_SDK = "docker"

# Files/folders to exclude from upload
EXCLUDE = {
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
    "validate_out*.txt",
    "push_to_hf.py",
    "fix_uvlock.py",
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE or part.startswith("."):
            return True
    for pattern in EXCLUDE:
        if "*" in pattern and path.match(pattern):
            return True
    return False


def main():
    api = HfApi()

    # Check auth
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"ERROR: Not authenticated. Run 'huggingface-cli login' first.\n{e}")
        sys.exit(1)

    # Create or get repo
    try:
        url = create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            exist_ok=True,
        )
        print(f"Space ready: {url}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    # Collect files to upload
    root = Path(".")
    files_to_upload = []
    for f in root.rglob("*"):
        if f.is_file() and not should_exclude(f):
            files_to_upload.append(f)

    print(f"\nUploading {len(files_to_upload)} files...")

    # Move Dockerfile to root if it's in server/
    dockerfile_src = root / "server" / "Dockerfile"
    dockerfile_dst = root / "Dockerfile"
    created_root_dockerfile = False
    if dockerfile_src.exists() and not dockerfile_dst.exists():
        import shutil
        shutil.copy(dockerfile_src, dockerfile_dst)
        files_to_upload.append(dockerfile_dst)
        created_root_dockerfile = True
        print("  Copied server/Dockerfile -> Dockerfile (required by HF Spaces)")

    # Upload all files
    errors = []
    for i, f in enumerate(files_to_upload, 1):
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=str(f.as_posix()),
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"  [{i}/{len(files_to_upload)}] Uploaded: {f}")
        except Exception as e:
            print(f"  [{i}/{len(files_to_upload)}] FAILED: {f} -> {e}")
            errors.append((f, e))

    # Cleanup
    if created_root_dockerfile:
        os.remove(dockerfile_dst)

    print()
    if errors:
        print(f"WARNING: {len(errors)} file(s) failed to upload:")
        for f, e in errors:
            print(f"  {f}: {e}")
    else:
        print("SUCCESS! All files uploaded.")
        print(f"\nYour Space is live at:")
        print(f"  https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
