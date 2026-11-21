"""Freeze the harness source for a campaign, and record where running code came from.

  python -m v2.freeze --name NAME

copies v2/ and embodied_agent/ (without caches) into <root>/embodied-harness-v2-<name>/, where
<root> is the directory holding the harness checkout, and writes FROZEN.json there: the source
identity (content hash of the files an episode runs) and the branch, commit and changed paths of the
harness and root repositories the copy was made from. The root repository is the RoboDojo checkout the
simulator runs: by default the directory holding the harness checkout; a harness kept elsewhere names it
with --robodojo (python -m v2.freeze --name C --robodojo /path/to/RoboDojo). A campaign runs from its frozen copy, so later
edits to the working tree never reach it; every run records this provenance in run.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from v2.runtime.client import HARNESS_ROOT

COPIED = ('v2', 'embodied_agent')
# RoboDojo code behind the simulated tasks and their success checks.
ROBODOJO_PATHS = ('task', 'env', 'env_cfg', 'utils')
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}')  # a campaign name fits
MAX_CHANGES = 200
KEY_LIKE = re.compile(r'sk-[A-Za-z0-9_-]{8,}')


def launch_command():
    """The command line of this process, with anything shaped like an API key removed."""
    return [KEY_LIKE.sub('[redacted]', part) for part in (sys.executable, *sys.argv)]


def source_identity(root=None):
    """Hash of the harness code an episode runs: v2/ and the reused v1 modules (tests excluded)."""
    root = Path(root or HARNESS_ROOT)
    files = {}
    for top in COPIED:
        for path in sorted((root / top).rglob('*')):
            parts = path.relative_to(root).parts
            if (path.is_file() and path.suffix in ('.py', '.md', '.json') and not path.name.startswith('._') and
                    not any(part in ('tests', '__pycache__', 'skills') for part in parts)):
                files['/'.join(parts)] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': len(files)}


def git_state(directory, paths=()):
    """Branch, commit and changes of the repository whose top level is `directory`.

    A directory that is not itself a repository top level (a frozen copy inside another checkout)
    is reported as such instead of borrowing the enclosing repository's commit. With `paths`, the
    changed and untracked files under those paths are listed and count as dirty. A failing Git is
    reported in git_error rather than as "not a repository".
    """
    directory = Path(directory).resolve()
    errors = []
    environment = {**os.environ, 'LC_ALL': 'C'}  # untranslated messages, e.g. "not a git repository"

    def run(*args):
        try:
            result = subprocess.run(['git', *args], cwd=directory, capture_output=True, text=True, timeout=30,
                                    env=environment)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f'git {args[0]}: {type(exc).__name__}: {exc}'[:300])
            return None
        if result.returncode != 0:
            errors.append(f'git {args[0]}: {result.stderr.strip()}'[:300])
            return None
        return result.stdout.rstrip('\n')
    top = run('rev-parse', '--show-toplevel')
    if top is None:
        state = {'path': str(directory), 'repository': False}
        if 'not a git repository' not in errors[0]:
            state['git_error'] = errors[0]
        return state
    if Path(top).resolve() != directory:
        return {'path': str(directory), 'repository': False, 'inside': top}
    tracked = run('status', '--porcelain', '--untracked-files=no')
    state = {'path': str(directory), 'repository': True, 'branch': run('branch', '--show-current') or None,
             'commit': run('rev-parse', 'HEAD'), 'dirty': None if tracked is None else bool(tracked)}
    if paths:
        changes = run('status', '--porcelain', '--untracked-files=all', '--', *paths)
        lines = None if changes is None else [line for line in changes.split('\n') if line]
        state['changes'] = None if lines is None else lines[:MAX_CHANGES]
        state['changes_total'] = None if lines is None else len(lines)
        if lines:
            state['dirty'] = True
            state['changes_sha256'] = _changes_digest(run, directory, paths, lines)
    if errors:
        state['git_error'] = errors
    return state


def _changes_digest(run, directory, paths, lines):
    """What the changes contain, not only which files they touch: sha256 of the diff of tracked files
    against HEAD and of the path and bytes of every untracked file (None if Git fails). Bytes throughout,
    so unusual file names and non-UTF-8 content are hashed rather than skipped or fatal."""
    def git_bytes(*args):
        try:
            return subprocess.run(['git', *args], cwd=directory, capture_output=True, timeout=60, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    diff = git_bytes('diff', 'HEAD', '--binary', '--', *paths)
    status = git_bytes('status', '--porcelain', '-z', '--untracked-files=all', '--', *paths)
    if diff is None or status is None:
        return None
    digest = hashlib.sha256(diff)
    for entry in sorted(item for item in status.split(b'\0') if item.startswith(b'?? ')):
        name = entry[3:]
        digest.update(name + b'\0')
        path = directory / os.fsdecode(name)
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def provenance(repo=None):
    """Where the running code came from: a frozen copy's record, or the checkouts themselves.

    `repo` is the RoboDojo checkout the simulator runs (task, reward and evaluator code); its live
    state is recorded as well, and compared with the state recorded when the copy was frozen.
    """
    frozen = HARNESS_ROOT / 'FROZEN.json'
    live = git_state(repo, ROBODOJO_PATHS) if repo else None
    if frozen.is_file():
        record = json.loads(frozen.read_text())
        state = {'frozen_copy': str(HARNESS_ROOT), 'frozen': record,
                 'frozen_source_matches': record.get('source_identity') == source_identity()}
        if live is not None:
            then = (record.get('git') or {}).get('root') or {}
            state['robodojo'] = live
            state['robodojo_same_as_frozen'] = (Path(then.get('path', '')).resolve() == Path(repo).resolve() and
                                                (then.get('commit'), then.get('changes'), then.get('changes_sha256')) ==
                                                (live.get('commit'), live.get('changes'), live.get('changes_sha256')))
        return state
    state = {'harness': git_state(HARNESS_ROOT, COPIED), 'root': git_state(HARNESS_ROOT.parent, ROBODOJO_PATHS)}
    if live is not None:
        state['robodojo'] = live
    return state


def freeze(name, root=None, harness=None, robodojo=None):
    """Copy the harness source to <root>/embodied-harness-v2-<name>/ and return its FROZEN.json record.

    `robodojo` is the RoboDojo checkout recorded as the root repository (default: the harness's parent).
    """
    harness = Path(harness or HARNESS_ROOT).resolve()
    robodojo = Path(robodojo).resolve() if robodojo else harness.parent
    if not NAME.fullmatch(name):
        raise ValueError('the name uses letters, digits, underscores or hyphens (at most 48)')
    if (harness / 'FROZEN.json').exists():
        raise ValueError(f'{harness} is itself a frozen copy; freeze from the harness checkout')
    root = Path(root).resolve() if root else harness.parent
    target = root / f'embodied-harness-v2-{name}'
    if target.exists():
        raise ValueError(f'{target} already exists')
    staging = root / f'.embodied-harness-v2-{name}.staging'
    if staging.exists():
        shutil.rmtree(staging)
    identity = source_identity(harness)
    try:
        for top in COPIED:
            shutil.copytree(harness / top, staging / top, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.DS_Store', '._*'))
        copied = source_identity(staging)
        if copied != identity or source_identity(harness) != identity:
            raise RuntimeError('the harness source changed while it was being copied; freeze again')
        record = {'schema': 'v2_frozen/1', 'name': name,
                  'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'source': str(harness), 'copied': list(COPIED), 'source_identity': identity,
                  'git': {'harness': git_state(harness, COPIED), 'root': git_state(robodojo, ROBODOJO_PATHS)},
                  'command': launch_command(), 'python': sys.version.split()[0]}
        (staging / 'FROZEN.json').write_text(json.dumps(record, indent=2) + '\n')
        for path in staging.rglob('*'):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)
        os.replace(staging, target)
    finally:
        if staging.exists():
            for path in staging.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    path.chmod(0o644)
            shutil.rmtree(staging)
    return {'path': str(target), **record}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m v2.freeze', description=__doc__.split('\n\n')[0])
    parser.add_argument('--name', required=True, help='campaign name')
    parser.add_argument('--root', help='where to put the copy (default: next to the harness checkout)')
    parser.add_argument('--robodojo', help='the RoboDojo checkout the simulator runs (default: the directory '
                                                'holding the harness checkout)')
    args = parser.parse_args(argv)
    record = freeze(args.name, args.root, robodojo=args.robodojo)
    print(json.dumps({key: record[key] for key in ('path', 'source_identity', 'git')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
