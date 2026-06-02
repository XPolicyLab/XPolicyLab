"""Git information save and restore utilities.

This module provides tools to:
1. Save complete git state (commit, staged/unstaged changes, untracked files) to git_info.json
2. Restore git state from git_info.json in a clean repository
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

try:
    from git import Repo, InvalidGitRepositoryError
except ImportError:
    Repo = None
    InvalidGitRepositoryError = Exception

logger = logging.getLogger(__name__)

# Thresholds for file handling
MAX_FILE_SIZE = 1024 * 1024  # 1MB
GIT_INFO_VERSION = "1.0"


class GitInfoError(Exception):
    """Custom exception for git info operations."""
    pass


def _is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by reading first 8192 bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)
            return b'\x00' in chunk
    except (IOError, OSError):
        return False


def _get_file_content(repo_path: Path, file_path: str) -> dict:
    """Get file content with handling for binary and large files.

    Returns:
        dict with keys: path, content (or null), binary, truncated, size
    """
    full_path = repo_path / file_path
    result = {"path": file_path}

    if not full_path.exists():
        result["content"] = None
        result["missing"] = True
        return result

    file_size = full_path.stat().st_size
    result["size"] = file_size

    if _is_binary_file(full_path):
        result["content"] = None
        result["binary"] = True
        return result

    if file_size > MAX_FILE_SIZE:
        result["content"] = None
        result["truncated"] = True
        return result

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            result["content"] = f.read()
    except UnicodeDecodeError:
        result["content"] = None
        result["binary"] = True
    except (IOError, OSError) as e:
        result["content"] = None
        result["error"] = str(e)

    return result


def _get_diff(repo: "Repo", file_path: str, staged: bool = False) -> Optional[str]:
    """Get diff for a file.

    Args:
        repo: GitPython Repo object
        file_path: Path to the file relative to repo root
        staged: If True, get staged diff; otherwise get unstaged diff

    Returns:
        Diff string or None if no diff available
    """
    try:
        if staged:
            # Staged changes: diff between HEAD and index
            diff_output = repo.git.diff('--cached', '--', file_path)
        else:
            # Unstaged changes: diff between index and working tree
            diff_output = repo.git.diff('--', file_path)
        return diff_output if diff_output else None
    except Exception:
        return None


def check_repo_clean(repo_path: str = None) -> bool:
    """Check if the repository is completely clean.

    Args:
        repo_path: Path to the repository. If None, searches parent directories.

    Returns:
        True if repo is clean (no staged, unstaged, or untracked files)
    """
    if Repo is None:
        raise GitInfoError("GitPython is not installed")

    try:
        repo = Repo(repo_path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise GitInfoError(f"Not a git repository: {repo_path}")

    # Check for any changes
    if repo.is_dirty(untracked_files=True):
        return False

    return True


def save_git_info(output_dir: Union[str, Path], repo_path: str = None) -> Path:
    """Save git state to git_info.json at training start.

    Args:
        output_dir: Directory where git_info.json will be saved
        repo_path: Path to the repository. If None, searches parent directories.

    Returns:
        Path to the saved git_info.json file

    Raises:
        GitInfoError: If not in a git repository or GitPython not installed
    """
    if Repo is None:
        raise GitInfoError("GitPython is not installed")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = Repo(repo_path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise GitInfoError(f"Not a git repository: {repo_path or 'current directory'}")

    repo_root = Path(repo.working_dir)

    # Get current commit info
    try:
        head_commit = repo.head.commit
        commit_info = {
            "hash": head_commit.hexsha,
            "short_hash": head_commit.hexsha[:7],
            "message": head_commit.message.strip(),
            "author": f"{head_commit.author.name} <{head_commit.author.email}>",
            "date": head_commit.committed_datetime.isoformat(),
        }
    except Exception as e:
        raise GitInfoError(f"Could not get commit info: {e}")

    # Get current branch
    try:
        branch = repo.active_branch.name
    except TypeError:
        # Detached HEAD state
        branch = None

    # Initialize changes structure
    changes = {
        "staged": {
            "modified": [],
            "added": [],
            "deleted": [],
        },
        "unstaged": {
            "modified": [],
            "deleted": [],
        },
        "untracked": [],
    }

    # Get staged changes (diff between HEAD and index)
    staged_diff = repo.index.diff('HEAD')
    for diff_item in staged_diff:
        if diff_item.change_type == 'M':
            # Modified file
            diff_content = _get_diff(repo, diff_item.a_path, staged=True)
            changes["staged"]["modified"].append({
                "path": diff_item.a_path,
                "diff": diff_content,
            })
        elif diff_item.change_type == 'A':
            # Added file (new file staged)
            file_info = _get_file_content(repo_root, diff_item.a_path)
            changes["staged"]["added"].append(file_info)
        elif diff_item.change_type == 'D':
            # Deleted file
            changes["staged"]["deleted"].append(diff_item.a_path)
        elif diff_item.change_type == 'R':
            # Renamed file - treat as delete old + add new
            changes["staged"]["deleted"].append(diff_item.a_path)
            file_info = _get_file_content(repo_root, diff_item.b_path)
            changes["staged"]["added"].append(file_info)

    # Get unstaged changes (diff between index and working tree)
    unstaged_diff = repo.index.diff(None)
    for diff_item in unstaged_diff:
        if diff_item.change_type == 'M':
            diff_content = _get_diff(repo, diff_item.a_path, staged=False)
            changes["unstaged"]["modified"].append({
                "path": diff_item.a_path,
                "diff": diff_content,
            })
        elif diff_item.change_type == 'D':
            changes["unstaged"]["deleted"].append(diff_item.a_path)

    # Get untracked files
    for untracked_path in repo.untracked_files:
        file_info = _get_file_content(repo_root, untracked_path)
        changes["untracked"].append(file_info)

    # Build final git_info structure
    git_info = {
        "version": GIT_INFO_VERSION,
        "timestamp": datetime.now().isoformat(),
        "branch": branch,
        "commit": commit_info,
        "is_dirty": repo.is_dirty(untracked_files=True),
        "changes": changes,
    }

    # Save to file
    git_info_path = output_dir / "git_info.json"
    with open(git_info_path, 'w', encoding='utf-8') as f:
        json.dump(git_info, f, indent=2, ensure_ascii=False)

    return git_info_path


def _apply_diff(repo_root: Path, file_path: str, diff_content: str) -> bool:
    """Apply a diff to a file using git apply.

    Args:
        repo_root: Path to repository root
        file_path: Path to the file relative to repo root
        diff_content: The diff content to apply

    Returns:
        True if successful, False otherwise
    """
    if not diff_content:
        logger.warning(f"No diff content for {file_path}, skipping")
        return False

    # Ensure diff ends with newline (required by git apply)
    if not diff_content.endswith('\n'):
        diff_content += '\n'

    try:
        # Use git apply to apply the diff
        result = subprocess.run(
            ['git', 'apply', '--whitespace=nowarn'],
            input=diff_content.encode('utf-8'),
            cwd=repo_root,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to apply diff for {file_path}: {result.stderr.decode()}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Error applying diff for {file_path}: {e}")
        return False


def _write_file(repo_root: Path, file_path: str, content: str) -> bool:
    """Write content to a file.

    Args:
        repo_root: Path to repository root
        file_path: Path to the file relative to repo root
        content: Content to write

    Returns:
        True if successful, False otherwise
    """
    full_path = repo_root / file_path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.warning(f"Error writing file {file_path}: {e}")
        return False


def restore_git_state(git_info_path: Union[str, Path], repo_path: str = None) -> None:
    """Restore git state from git_info.json.

    Prerequisites:
        - Repository must be completely clean (no uncommitted changes)

    Restoration steps:
        1. Check if repository is clean, error if not
        2. Checkout to the saved commit
        3. Apply staged changes (write files + git add)
        4. Apply unstaged changes (write files only)
        5. Create untracked files

    Args:
        git_info_path: Path to the git_info.json file
        repo_path: Path to the repository. If None, searches parent directories.

    Raises:
        GitInfoError: If repository is not clean or other errors occur
    """
    if Repo is None:
        raise GitInfoError("GitPython is not installed")

    git_info_path = Path(git_info_path)
    if not git_info_path.exists():
        raise GitInfoError(f"git_info.json not found: {git_info_path}")

    # Load git info
    with open(git_info_path, 'r', encoding='utf-8') as f:
        git_info = json.load(f)

    # Get repository
    try:
        repo = Repo(repo_path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise GitInfoError(f"Not a git repository: {repo_path or 'current directory'}")

    repo_root = Path(repo.working_dir)

    # Check if repository is clean
    if repo.is_dirty(untracked_files=True):
        raise GitInfoError(
            "Repository is not clean. Please commit or stash all changes before restoring.\n"
            "Run 'git status' to see pending changes."
        )

    commit_hash = git_info["commit"]["hash"]
    short_hash = git_info["commit"]["short_hash"]

    # Verify commit exists
    try:
        repo.commit(commit_hash)
    except Exception:
        raise GitInfoError(
            f"Commit {short_hash} not found in repository. "
            "Please fetch the commit first: git fetch origin"
        )

    # Checkout to the saved commit
    logger.info(f"Checking out commit {short_hash}...")
    try:
        repo.git.checkout(commit_hash)
    except Exception as e:
        raise GitInfoError(f"Failed to checkout commit {short_hash}: {e}")

    changes = git_info.get("changes", {})
    warnings = []

    # Restore staged changes
    staged = changes.get("staged", {})

    # Staged added files
    for item in staged.get("added", []):
        path = item["path"]
        if item.get("binary") or item.get("truncated") or item.get("content") is None:
            warnings.append(f"Skipping staged added file (binary/truncated/missing): {path}")
            continue
        if _write_file(repo_root, path, item["content"]):
            repo.index.add([path])
            logger.info(f"Restored staged added: {path}")

    # Staged modified files
    for item in staged.get("modified", []):
        path = item["path"]
        diff_content = item.get("diff")
        if not diff_content:
            warnings.append(f"Skipping staged modified file (no diff): {path}")
            continue
        if _apply_diff(repo_root, path, diff_content):
            repo.index.add([path])
            logger.info(f"Restored staged modified: {path}")
        else:
            warnings.append(f"Failed to apply diff for staged file: {path}")

    # Staged deleted files
    for path in staged.get("deleted", []):
        try:
            full_path = repo_root / path
            if full_path.exists():
                full_path.unlink()
            repo.index.remove([path], working_tree=True)
            logger.info(f"Restored staged deleted: {path}")
        except Exception as e:
            warnings.append(f"Failed to delete staged file {path}: {e}")

    # Restore unstaged changes
    unstaged = changes.get("unstaged", {})

    # Unstaged modified files
    for item in unstaged.get("modified", []):
        path = item["path"]
        diff_content = item.get("diff")
        if not diff_content:
            warnings.append(f"Skipping unstaged modified file (no diff): {path}")
            continue
        if _apply_diff(repo_root, path, diff_content):
            logger.info(f"Restored unstaged modified: {path}")
        else:
            warnings.append(f"Failed to apply diff for unstaged file: {path}")

    # Unstaged deleted files
    for path in unstaged.get("deleted", []):
        try:
            full_path = repo_root / path
            if full_path.exists():
                full_path.unlink()
            logger.info(f"Restored unstaged deleted: {path}")
        except Exception as e:
            warnings.append(f"Failed to delete unstaged file {path}: {e}")

    # Create untracked files
    for item in changes.get("untracked", []):
        path = item["path"]
        if item.get("binary") or item.get("truncated") or item.get("content") is None:
            warnings.append(f"Skipping untracked file (binary/truncated/missing): {path}")
            continue
        if _write_file(repo_root, path, item["content"]):
            logger.info(f"Restored untracked: {path}")

    # Print summary
    if warnings:
        logger.warning("Restoration completed with warnings:")
        for w in warnings:
            logger.warning(f"  - {w}")

    logger.info(f"Successfully restored to commit {short_hash}")
    if git_info.get("branch"):
        logger.info(f"Original branch was: {git_info['branch']}")
    logger.info("Note: You are now in detached HEAD state. To create a branch:")
    logger.info(f"  git checkout -b <branch-name>")
