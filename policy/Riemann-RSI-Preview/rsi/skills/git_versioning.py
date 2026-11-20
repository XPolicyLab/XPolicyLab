import os
from pathlib import Path
import re
import subprocess


def git(root, *arguments):
    environment = dict(os.environ)
    for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR',
                 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES'):
        environment.pop(name, None)
    result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
                            env=environment, timeout=120, check=False)
    if result.returncode:
        raise ValueError('Git operation failed: ' + result.stderr.decode('utf-8', 'replace')[-2000:])
    return result.stdout


def repository_state(root):
    root = Path(root).resolve()
    try:
        top = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    except (ValueError, FileNotFoundError) as error:
        raise ValueError('Candidate preparation requires a Git repository; source-copy fallback is disabled') from error
    if top != root:
        raise ValueError('Use the exact Git worktree root, not a nested task directory')
    commit = git(root, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit) is None:
        raise ValueError('Use an existing committed source version')
    branch = git(root, 'rev-parse', '--abbrev-ref', 'HEAD').decode().strip()
    common = Path(git(root, 'rev-parse', '--git-common-dir').decode().strip())
    if not common.is_absolute():
        common = root / common
    status = git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=normal')
    return {'schema': 'riemann-git-version-v1', 'commit': commit, 'branch': branch,
            'worktree': str(root), 'common_git_directory': str(common.resolve()),
            'clean': not status, 'official_performance_claimed': False}


def runtime_source_identity(root):
    """Record source provenance for a worktree, PR policy folder or source archive."""
    root = Path(root).resolve()
    try:
        commit = git(root, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
        dirty = bool(git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=normal', '--', '.'))
        return {'source_versioning': 'git', 'source_commit': commit, 'source_dirty': dirty}
    except (ValueError, FileNotFoundError):
        import json
        manifest = root / 'package-manifest.json'
        data = json.loads(manifest.read_text()) if manifest.is_file() else {}
        return {'source_versioning': 'exported_source', 'source_commit': data.get('source_commit'),
                'source_dirty': None}



def require_clean(root):
    state = repository_state(root)
    if not state['clean']:
        raise ValueError('Commit or explicitly resolve current changes before evaluating a version')
    return state
