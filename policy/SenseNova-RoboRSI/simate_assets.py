"""Pinned Simate source, installer inputs and checkpoint assets (no inference)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

POLICY_DIR = Path(__file__).resolve().parent
MANIFEST_SHA256 = '87db54a5945527ddbff6eeea865bd89fe1efd54755a4a62f11a966c78b1ad2d1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(policy_dir=POLICY_DIR):
    policy = Path(policy_dir)
    manifest_path = policy / 'simate/manifest.json'
    if manifest_path.is_symlink() or sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError('Pinned Simate installation manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported Simate installation manifest')
    for name, expected in manifest['resources'].items():
        path = policy / 'simate' / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError('Pinned Simate installer input changed: ' + name)
    return manifest


def validate_source(policy_dir=POLICY_DIR):
    policy = Path(policy_dir)
    load_metadata(policy)
    source = json.loads((policy / 'simate/source.json').read_text())
    root = policy.parent / 'Simate_beta'
    for name, record in source['files'].items():
        path = root / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record['bytes']
                or sha256_file(path) != record['sha256']):
            raise ValueError('Required official policy/Simate_beta source missing or changed: ' + name)
    return {'root': str(root), 'revision': source['revision'], 'files': len(source['files'])}


def checkpoint_assets(policy_dir=POLICY_DIR):
    policy = Path(policy_dir)
    load_metadata(policy)
    return json.loads((policy / 'simate/assets.json').read_text())['assets']


def verify_asset(path, record):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record['size'] or sha256_file(path) != record['sha256']:
        raise ValueError('Simate checkpoint size/SHA-256 differs: ' + path.name)


def install_assets(destination, policy_dir=POLICY_DIR):
    """Reuse complete local assets; otherwise fetch the fixed public revision."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = checkpoint_assets(policy_dir)
    for name, record in records.items():
        path = destination / name
        if path.exists() or path.is_symlink():
            verify_asset(path, record)
            continue
        cache = os.environ.get('ROBORSI_SIM_ASSET_CACHE')
        cached = Path(cache).expanduser() / name if cache else None
        temporary = destination / ('.' + name + '.download-' + str(os.getpid()))
        try:
            with temporary.open('xb') as output:
                if cached is not None and cached.is_file():
                    verify_asset(cached, record)
                    with cached.open('rb') as source:
                        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                else:
                    with urlopen(record['url'], timeout=120) as source:
                        remaining = record['size']
                        while remaining:
                            block = source.read(min(4 * 1024 * 1024, remaining))
                            if not block:
                                raise ValueError('Truncated Simate checkpoint: ' + name)
                            output.write(block)
                            remaining -= len(block)
                        if source.read(1):
                            raise ValueError('Simate checkpoint exceeds pinned size: ' + name)
            verify_asset(temporary, record)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {'directory': str(destination), 'assets': records, 'verification': 'every asset size and SHA-256'}
