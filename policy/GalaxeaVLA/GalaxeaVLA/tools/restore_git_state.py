#!/usr/bin/env python3
"""Command-line tool to restore git state from git_info.json.

Usage:
    python tools/restore_git_state.py /path/to/output_dir/git_info.json
    python tools/restore_git_state.py /path/to/git_info.json --repo /path/to/repo

This tool restores the exact git state (commit + local modifications) that was
saved during a training run, allowing you to reproduce the exact code state.

Prerequisites:
    - The target repository must be completely clean (no uncommitted changes)
    - The saved commit must exist in the repository

Example workflow:
    1. Run training: bash scripts/run/finetune.sh 1 real/r1lite_g0plus_finetune
    2. Git info is automatically saved to output_dir/git_info.json
    3. Later, to restore the exact code state:
       git stash  # or commit your current changes
       python tools/restore_git_state.py /path/to/output_dir/git_info.json
    4. Verify with: git status && git diff
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Restore git state from git_info.json saved during training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "git_info_path",
        type=Path,
        help="Path to the git_info.json file",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to the target repository (default: current directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input path
    if not args.git_info_path.exists():
        logger.error(f"File not found: {args.git_info_path}")
        sys.exit(1)

    if not args.git_info_path.name.endswith('.json'):
        logger.warning(f"File does not have .json extension: {args.git_info_path}")

    # Import here to allow help without dependencies
    try:
        from galaxea_fm.utils.git_info import restore_git_state, GitInfoError, check_repo_clean
    except ImportError:
        # Try relative import if not installed
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from galaxea_fm.utils.git_info import restore_git_state, GitInfoError, check_repo_clean

    # Dry run mode - just show info
    if args.dry_run:
        import json
        with open(args.git_info_path, 'r') as f:
            git_info = json.load(f)

        print("\n=== Git Info Summary ===")
        print(f"Version: {git_info.get('version', 'unknown')}")
        print(f"Timestamp: {git_info.get('timestamp', 'unknown')}")
        print(f"Branch: {git_info.get('branch', 'detached HEAD')}")
        print(f"Commit: {git_info['commit']['short_hash']} - {git_info['commit']['message'][:50]}...")
        print(f"Is Dirty: {git_info.get('is_dirty', False)}")

        changes = git_info.get("changes", {})
        staged = changes.get("staged", {})
        unstaged = changes.get("unstaged", {})
        untracked = changes.get("untracked", [])

        print("\n=== Changes to Restore ===")
        print(f"Staged added: {len(staged.get('added', []))} files")
        print(f"Staged modified: {len(staged.get('modified', []))} files")
        print(f"Staged deleted: {len(staged.get('deleted', []))} files")
        print(f"Unstaged modified: {len(unstaged.get('modified', []))} files")
        print(f"Unstaged deleted: {len(unstaged.get('deleted', []))} files")
        print(f"Untracked: {len(untracked)} files")

        # Check if repo is clean
        try:
            is_clean = check_repo_clean(args.repo)
            print(f"\nRepository clean: {is_clean}")
            if not is_clean:
                print("WARNING: Repository has uncommitted changes. Clean it before restoring.")
        except GitInfoError as e:
            print(f"\nRepository check failed: {e}")

        print("\nDry run complete. Use without --dry-run to perform restoration.")
        sys.exit(0)

    # Perform restoration
    try:
        logger.info(f"Restoring git state from: {args.git_info_path}")
        restore_git_state(args.git_info_path, repo_path=args.repo)
        logger.info("Git state restoration completed successfully!")
        print("\nTo verify, run:")
        print("  git status")
        print("  git diff")
        print("  git diff --cached")
    except GitInfoError as e:
        logger.error(f"Restoration failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
