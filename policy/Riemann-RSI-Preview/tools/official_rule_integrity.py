"""Verify external official SDK, protocol and resource identities."""
from pathlib import Path
import hashlib
import json
import subprocess
from rsi.resource_locations import relocate_path
from rsi.rule_configuration import settings

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(settings()['official_root'])
PROTOCOL_RUNTIME_PATHS = (
    '__init__.py', 'model_template.py', 'pyproject.toml', 'client_server', 'utils',
    'policy/__init__.py', 'policy/demo_policy',
)


def _verify_protocol_checkout(root, baseline):
    """Permit adapter submissions while retaining the pinned shared runtime."""
    command = ['git', '-c', 'safe.directory=' + str(root), '-C', str(root)]
    actual = subprocess.check_output([*command, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != baseline:
        ancestry = subprocess.run([*command, 'merge-base', '--is-ancestor', baseline, actual],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ancestry.returncode:
            raise ValueError('Installed official policy protocol is not based on the pinned checkout')
        # Compare shared runtime trees, not unrelated policies, docs or data tools.
        # Protected file hashes below still cover the actual SDK/protocol inputs.
        changed = subprocess.check_output([*command, 'diff', '--name-only', '--no-renames',
                                           '-z', baseline, actual, '--',
                                           *PROTOCOL_RUNTIME_PATHS])
        if changed:
            raise ValueError('Installed official policy shared runtime changed')
    # These small shared paths also cover helpers beyond the protected file list.
    # Submitted policy files remain Git-managed without becoming startup seals.
    dirty = subprocess.check_output([*command, 'diff', '--name-only', '--no-renames', '-z',
                                      'HEAD', '--', *PROTOCOL_RUNTIME_PATHS])
    if dirty:
        raise ValueError('Installed official policy shared runtime has uncommitted changes')
    return actual


def verify_policy(*, sdk=None):
    runtime = Path(sdk) if sdk is not None else SDK
    data = json.loads((ROOT / 'configs/official-runtime-inputs.json').read_text())
    groups = [('official_files', runtime)]
    for group, root in groups:
        for name, expected in data[group].items():
            relative = Path(name); path = root / relative
            if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                    or not path.is_file() or not path.resolve().is_relative_to(root)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError('Official policy closure changed: ' + str(path))
    head = subprocess.check_output(['git', '-c', 'safe.directory=' + str(runtime), '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip()
    if head != data['official_commit']:
        raise ValueError('Official SDK commit changed')
    _verify_protocol_checkout(runtime / 'XPolicyLab', data['actual_XPolicyLab_commit'])
    for name, expected in data.get('external_files', {}).items():
        path = Path(relocate_path(name))
        if not path.is_absolute() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Declared policy protocol dependency changed: ' + name)
    for name, target in data.get('external_symlinks', {}).items():
        path = Path(relocate_path(name))
        if not path.is_symlink() or str(path.resolve()) != str(Path(relocate_path(target)).resolve()) or target not in data['external_files']:
            raise ValueError('Declared external dependency link changed: ' + name)
    return len(data['official_files'])
