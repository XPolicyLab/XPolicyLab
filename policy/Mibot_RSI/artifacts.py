"""Mibot-RSI artifacts: the harness runtime (runtime/) and the knowledge snapshots (README, Installation, Model Assets).

  python artifacts.py deps [--install]        check (or install) the client-side Python dependencies
  python artifacts.py knowledge --ckpt NAME [--seeds 1] [--tasks a,b]
                                              install the knowledge release NAME pinned in release.json
  python artifacts.py check [--model-env]     what an evaluation needs: dependencies, the runtime (every file of
                                              runtime/ against runtime/EXPORT.json) and, with --model-env, the
                                              model endpoint and key variables (names only)

The runtime ships in runtime/: an eval-only export of one Mibot-RSI commit. Every knowledge archive is a file
of the Hugging Face dataset named in release.json, downloaded from the dataset commit it pins (dataset.revision) and
checked against its pinned sha256 and size before it is unpacked; unpacked files are made read-only.
MIBOT_RSI_ARTIFACTS_DIR=<dir> reads the same relative paths from a local copy of the dataset (still sha256-checked).
A private dataset uses HF_TOKEN or the cached `huggingface-cli login`.
Standard library only, plus huggingface_hub for downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS = POLICY_DIR / 'checkpoints'
RUNTIME = POLICY_DIR / 'runtime'
# The harness runtime's dependencies (its requirements.txt), checked in the evaluation client's Python.
DEPENDENCIES = (('numpy', 'numpy', '>=1.24,<3'), ('scipy', 'scipy', '>=1.10,<2'), ('PIL', 'Pillow', '>=10,<13'),
                ('cv2', 'opencv-python-headless', '>=4.8,<5'), ('yaml', 'pyyaml', '>=5'))
STANDARD_VARIANT_SUFFIX = '_random'


class ArtifactError(RuntimeError):
    pass


def release():
    path = POLICY_DIR / 'release.json'
    if not path.is_file():
        raise ArtifactError(f'{path} is missing')
    return json.loads(path.read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ fetching

def fetch(relative, pin):
    """A local path to the dataset file `relative`, checked against its pin ({sha256, bytes})."""
    if os.environ.get('MIBOT_RSI_ARTIFACTS_DIR'):
        path = Path(os.environ['MIBOT_RSI_ARTIFACTS_DIR']).expanduser() / relative
    else:
        dataset = release().get('dataset') or {}
        revision = dataset.get('revision')
        if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set('0123456789abcdef'):
            raise ArtifactError(f"release.json pins no commit of the dataset {dataset.get('repo')} (dataset.revision); "
                                'downloads come only from that commit')
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ArtifactError('huggingface_hub is not installed; pip install huggingface_hub, or set '
                                'MIBOT_RSI_ARTIFACTS_DIR to a local copy of the dataset') from None
        print(f"[artifacts] downloading {relative} from {dataset['repo']}@{revision[:12]}", flush=True)
        path = Path(hf_hub_download(repo_id=dataset['repo'], filename=relative, repo_type='dataset',
                                    revision=revision))
    if not path.is_file():
        raise ArtifactError(f'{path} is not a file')
    if pin.get('bytes') is not None and path.stat().st_size != pin['bytes']:
        raise ArtifactError(f'{relative}: {path.stat().st_size} bytes, the pin says {pin["bytes"]}')
    digest = sha256_file(path)
    if digest != pin['sha256']:
        raise ArtifactError(f'{relative}: sha256 {digest} does not match the pinned {pin["sha256"]}')
    return path


def unpack(archive, target, verify):
    """Unpack a checked archive into `target` (read-only), through a staging folder; `verify(folder)` raises on
    content that does not match its own manifest. An existing valid target is kept."""
    target = Path(target)
    if target.exists():
        try:
            verify(target)
            print(f'[artifacts] {target} is already installed and verified')
            return target
        except Exception as exc:
            print(f'[artifacts] replacing {target}: {exc}')
            remove(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=target.parent))
    try:
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle.getmembers():
                name = Path(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                    raise ArtifactError(f'{archive}: unsafe member {member.name}')
            try:
                bundle.extractall(staging, filter='data')
            except TypeError:  # Python without extraction filters; members were checked above
                bundle.extractall(staging)
        verify(staging)
        make_read_only(staging)
        os.replace(staging, target)
    except BaseException:
        remove(staging)
        raise
    return target


def make_read_only(folder):
    for path in sorted(Path(folder).rglob('*'), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    Path(folder).chmod(0o555)


def remove(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if not path.exists():
        return
    for item in [path, *path.rglob('*')]:
        if not item.is_symlink():
            item.chmod(0o755 if item.is_dir() else 0o644)
    shutil.rmtree(path)


# ------------------------------------------------------------------ runtime

def verify_runtime(folder):
    """Every file of the runtime against its EXPORT.json (deploy.py's runtime refuses to run otherwise too)."""
    folder = Path(folder)
    record = json.loads((folder / 'EXPORT.json').read_text())
    for relative, digest in record['files'].items():
        if sha256_file(folder / relative) != digest:
            raise ArtifactError(f'runtime file {relative} does not match EXPORT.json')
    extra = sorted(path.relative_to(folder).as_posix() for path in folder.rglob('*')
                   if (path.is_file() or path.is_symlink()) and '__pycache__' not in path.parts
                   and path.relative_to(folder).as_posix() not in record['files'] and path.name != 'EXPORT.json')
    if extra:
        raise ArtifactError('runtime/ holds files outside EXPORT.json: ' + ', '.join(extra[:5]))
    if not (folder / 'v2' / 'submission.py').is_file():
        raise ArtifactError(f'{folder} holds no v2/submission.py')
    return record


def resolve_runtime():
    if os.environ.get('MIBOT_RSI_HARNESS'):
        folder = Path(os.environ['MIBOT_RSI_HARNESS']).expanduser()
        if not (folder / 'v2' / 'submission.py').is_file():
            raise ArtifactError(f'MIBOT_RSI_HARNESS={folder} holds no v2/submission.py')
        return folder
    if not (RUNTIME / 'EXPORT.json').is_file():
        raise ArtifactError(f'{RUNTIME} holds no EXPORT.json')
    record = verify_runtime(RUNTIME)
    print(f"[artifacts] harness runtime {str(record.get('commit'))[:12]} ({len(record['files'])} files) verified")
    return RUNTIME


# ------------------------------------------------------------------ knowledge

def verify_snapshot(folder, pin=None):
    """Every file of a knowledge snapshot against its manifest.json (and no other file), and the manifest against
    the pin."""
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        if sha256_file(folder / relative) != digest:
            raise ArtifactError(f"snapshot {manifest['id']}: {relative} was modified")
    extra = sorted({path.relative_to(folder).as_posix() for path in folder.rglob('*')
                    if path.is_file() or path.is_symlink()} - set(manifest['files']) - {'manifest.json'})
    if extra:
        raise ArtifactError(f"snapshot {manifest['id']} holds files outside its manifest: " + ', '.join(extra[:5]))
    if pin is not None:
        if manifest['id'] != pin['snapshot_id']:
            raise ArtifactError(f"snapshot {manifest['id']} is not the pinned {pin['snapshot_id']}")
        if sha256_file(folder / 'manifest.json') != pin['manifest_sha256']:
            raise ArtifactError(f"the manifest of snapshot {manifest['id']} differs from the pinned one")
    return manifest


def knowledge_dir(ckpt, seed, env_cfg='arx_x5', action_type='joint', bench='RoboDojo'):
    return CHECKPOINTS / f'{bench}-{ckpt}-{env_cfg}-{action_type}-{seed}'


def knowledge_store(ckpt):
    """The one folder of a knowledge release. It names no seed: the agent is shown paths inside it."""
    return CHECKPOINTS / f'mibot-rsi-knowledge-{ckpt}'


def install_knowledge(ckpt, seeds, tasks=None):
    pins = release().get('knowledge') or {}
    if pins.get('ckpt_name') != ckpt:
        raise ArtifactError(f"release.json pins knowledge release {pins.get('ckpt_name')!r}, not {ckpt!r}")
    entries = pins.get('tasks') or {}
    wanted = sorted(entries) if not tasks else sorted({t[:-len(STANDARD_VARIANT_SUFFIX)] if
                                                      t.endswith(STANDARD_VARIANT_SUFFIX) else t for t in tasks})
    missing = [task for task in wanted if task not in entries]
    if missing or not wanted:
        raise ArtifactError('release.json pins no snapshot for ' + (', '.join(missing) or 'any task'))
    store = knowledge_store(ckpt)
    store.mkdir(parents=True, exist_ok=True)
    for task in wanted:
        pin = entries[task]
        archive = fetch(pin['archive'], pin)
        unpack(archive, store / task, lambda folder, pin=pin: verify_snapshot(folder, pin))
        print(f"[artifacts] {task}: snapshot {pin['snapshot_id']} verified")
    # Knowledge does not depend on the evaluation seed: each seed's checkpoint folder (XPolicyLab's layout) links to
    # the one seed-free folder, and the harness resolves the link, so no path the agent sees names the seed.
    for seed in seeds:
        link = knowledge_dir(ckpt, seed)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise ArtifactError(f'{link} exists and is not a link to {store.name}')
        link.symlink_to(store.name)
    print(f'[artifacts] knowledge {ckpt}: {len(wanted)} task(s) in {store}, linked for seeds '
          + ', '.join(map(str, seeds)))
    return store


# ------------------------------------------------------------------ checks

def check_deps(install=False):
    missing = []
    for module, package, spec in DEPENDENCIES:
        try:
            loaded = importlib.import_module(module)
            print(f"[artifacts] {package} {getattr(loaded, '__version__', '?')}")
        except ImportError:
            missing.append(package + spec)
    if missing and install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
        return check_deps(False)
    if missing:
        raise ArtifactError('missing in ' + sys.executable + ': ' + ', '.join(missing) +
                            ' (bash install.sh --install-deps ' + sys.executable + ')')


def check_model_env():
    import yaml
    config = yaml.safe_load((POLICY_DIR / 'deploy.yml').read_text()) or {}
    key_env = (config.get('model_settings') or {}).get('api_key_env') or 'MIBOT_RSI_API_KEY'
    missing = [name for name in ('MIBOT_RSI_BASE_URL', key_env) if not os.environ.get(name)]
    if missing:
        raise ArtifactError('set ' + ' and '.join(missing) + ' (README, Evaluation)')
    print(f'[artifacts] MIBOT_RSI_BASE_URL and {key_env} are set')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    deps = sub.add_parser('deps')
    deps.add_argument('--install', action='store_true')
    knowledge = sub.add_parser('knowledge')
    knowledge.add_argument('--ckpt', required=True)
    knowledge.add_argument('--seeds', default='1')
    knowledge.add_argument('--tasks')
    check = sub.add_parser('check')
    check.add_argument('--model-env', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'deps':
            check_deps(args.install)
        elif args.command == 'knowledge':
            seeds = [int(seed) for seed in args.seeds.split(',') if seed.strip()]
            install_knowledge(args.ckpt, seeds, args.tasks.split(',') if args.tasks else None)
        else:
            check_deps(False)
            print(f'[artifacts] harness runtime: {resolve_runtime()}')
            if args.model_env:
                check_model_env()
    except (ArtifactError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'[artifacts][ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
