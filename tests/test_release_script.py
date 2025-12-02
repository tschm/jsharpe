"""Tests for the release.sh script using a sandboxed git environment."""

import shutil
import subprocess
from pathlib import Path

import pytest

# Path to the release script in the actual workspace
WORKSPACE_ROOT = Path(__file__).parent.parent
RELEASE_SCRIPT_PATH = WORKSPACE_ROOT / ".github" / "scripts" / "release.sh"

MOCK_UV_SCRIPT = """#!/usr/bin/env python3
import sys
import re
import argparse

def get_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    match = re.search(r'version = "(.*?)"', content)
    return match.group(1) if match else "0.0.0"

def set_version(new_version):
    with open("pyproject.toml", "r") as f:
        content = f.read()
    new_content = re.sub(r'version = ".*?"', f'version = "{new_version}"', content)
    with open("pyproject.toml", "w") as f:
        f.write(new_content)

def bump_version(current, bump_type):
    major, minor, patch = map(int, current.split('.'))
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return current

def main():
    # Minimal argument parsing to match release.sh usage
    args = sys.argv[1:]
    if not args or args[0] != "version":
        sys.exit(1)

    # uv version --short
    if "--short" in args and "--bump" not in args:
        print(get_version())
        return

    # uv version --bump <type> --dry-run --short
    if "--bump" in args and "--dry-run" in args and "--short" in args:
        bump_idx = args.index("--bump") + 1
        bump_type = args[bump_idx]
        current = get_version()
        print(bump_version(current, bump_type))
        return

    # uv version <version> --dry-run
    if "--dry-run" in args and "--bump" not in args:
        # Just validation, return success
        return

    # uv version --bump <type> (actual update)
    if "--bump" in args and "--dry-run" not in args:
        bump_idx = args.index("--bump") + 1
        bump_type = args[bump_idx]
        current = get_version()
        new_ver = bump_version(current, bump_type)
        set_version(new_ver)
        return

    # uv version <version> (actual update)
    # args: ['version', '1.2.3']
    if len(args) == 2 and not args[1].startswith("-"):
        set_version(args[1])
        return

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Sets up a remote bare repo and a local clone with necessary files."""
    remote_dir = tmp_path / "remote.git"
    local_dir = tmp_path / "local"

    # 1. Create bare remote
    remote_dir.mkdir()
    subprocess.run(["git", "init", "--bare", str(remote_dir)], check=True)

    # 2. Clone to local
    subprocess.run(["git", "clone", str(remote_dir), str(local_dir)], check=True)

    # 3. Setup local repo content
    # Use monkeypatch to safely change cwd for the duration of the test
    monkeypatch.chdir(local_dir)

    # Create pyproject.toml
    with open("pyproject.toml", "w") as f:
        f.write('[project]\nname = "test-project"\nversion = "0.1.0"\n')

    # Create dummy uv.lock
    with open("uv.lock", "w") as f:
        f.write("")

    # Create bin/uv mock
    bin_dir = local_dir / "bin"
    bin_dir.mkdir()
    uv_path = bin_dir / "uv"
    with open(uv_path, "w") as f:
        f.write(MOCK_UV_SCRIPT)
    uv_path.chmod(0o755)

    # Copy release script
    script_dir = local_dir / ".github" / "scripts"
    script_dir.mkdir(parents=True)
    shutil.copy(RELEASE_SCRIPT_PATH, script_dir / "release.sh")
    script_path = script_dir / "release.sh"
    script_path.chmod(0o755)

    # Commit and push initial state
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    subprocess.run(["git", "push", "origin", "master"], check=True)

    yield local_dir


def test_release_bump_patch_local_only(git_repo):
    """Test bumping patch version without pushing (default)."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Run release script
    result = subprocess.run([str(script), "--bump", "patch"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "Updated version to 0.1.1" in result.stdout
    assert "Skipping push of commit" in result.stdout
    assert "Skipping push of tag" in result.stdout

    # Verify tag exists locally
    tags = subprocess.check_output(["git", "tag"], cwd=git_repo, text=True)
    assert "v0.1.1" in tags

    # Verify tag does NOT exist on remote
    remote_tags = subprocess.check_output(["git", "ls-remote", "--tags", "origin"], cwd=git_repo, text=True)
    assert "v0.1.1" not in remote_tags


def test_release_bump_patch_with_push(git_repo):
    """Test bumping patch version WITH push."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Run release script with --push
    result = subprocess.run([str(script), "--bump", "patch", "--push"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "Updated version to 0.1.1" in result.stdout
    assert "Pushing commit to master" in result.stdout
    assert "Pushing tag to origin" in result.stdout

    # Verify tag exists on remote
    remote_tags = subprocess.check_output(["git", "ls-remote", "--tags", "origin"], cwd=git_repo, text=True)
    assert "v0.1.1" in remote_tags


def test_release_dry_run(git_repo):
    """Test dry run does not make changes."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    result = subprocess.run([str(script), "--bump", "patch", "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "[DRY RUN]" in result.stdout

    # Verify NO tag exists locally
    tags = subprocess.check_output(["git", "tag"], cwd=git_repo, text=True)
    assert "v0.1.1" not in tags

    # Verify pyproject.toml not changed
    with open(git_repo / "pyproject.toml") as f:
        assert 'version = "0.1.0"' in f.read()


def test_uncommitted_changes_failure(git_repo):
    """Test script fails if there are uncommitted changes."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a dirty file
    with open(git_repo / "dirty_file.txt", "w") as f:
        f.write("dirty")

    # We don't add it, but git status --porcelain will show it as untracked?
    # Wait, git status --porcelain shows untracked files too.
    # Let's modify a tracked file to be sure.
    with open(git_repo / "pyproject.toml", "a") as f:
        f.write("\n# comment")

    result = subprocess.run([str(script), "--bump", "patch"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


def test_ambiguous_tag_warning(git_repo):
    """Test warning when a tag matches the branch name."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a tag named 'master' (since default branch is master in our test setup)
    subprocess.run(["git", "tag", "master"], cwd=git_repo, check=True)

    result = subprocess.run([str(script), "--bump", "patch", "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "conflicts with the branch name" in result.stdout
