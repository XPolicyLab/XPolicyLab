"""Download an immutable release revision and verify every released byte."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'checkpoints/c05')
    p.add_argument('--revision', required=True, help='Immutable 40-character HF commit from the submission PR')
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        p.error('Use an immutable full commit SHA, not main')
    from huggingface_hub import snapshot_download
    root = args.output.resolve()
    snapshot_download('Aaron-Tian/lingbot-vla-v2-c05-robotwin', revision=args.revision, local_dir=root)
    manifest = json.loads((root / 'release_manifest.json').read_text())
    for name, record in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe release path')
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Release file failed SHA/size check: ' + name)
    contract = json.loads((root / 'checkpoint_contract.json').read_text())
    for name, expected in contract['weights_sha256'].items():
        if manifest['files']['checkpoint/' + name]['sha256'] != expected:
            raise ValueError('Weight identity differs from contract')
    identity = hashlib.sha256(json.dumps(contract['weights_sha256'], sort_keys=True).encode()).hexdigest()
    if identity != contract['checkpoint_id']:
        raise ValueError('Checkpoint identity mismatch')
    (root / 'DOWNLOAD_VERIFIED.json').write_text(json.dumps({
        'revision': args.revision, 'checkpoint_id': identity, 'files': len(manifest['files']),
        'all_sha256_verified': True}, indent=2) + '\n')
    print('Verified C05 release:', root)


if __name__ == '__main__':
    main()
