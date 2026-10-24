"""Fetch pinned public vision weights; verify every file against the evaluated copy."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(path, expected):
    if not path.is_file() or path.stat().st_size != expected['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest() == expected['sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    from real_rsi_assets import model_path
    root = (args.output or Path(model_path())).expanduser().resolve()
    lock = json.loads((HERE / 'models.lock.json').read_text())
    errors = []
    for name, spec in lock.items():
        for filename, expected in spec['files'].items():
            target = root / name / filename
            if verify(target, expected):
                continue
            if not args.verify_only:
                from huggingface_hub import hf_hub_download
                hf_hub_download(spec['repo_id'], filename, revision=spec['revision'],
                                local_dir=root / name, force_download=target.exists())
            if not verify(target, expected):
                errors.append(str(target))
        print(f'Checked {name}', flush=True)
    if errors:
        raise SystemExit('Missing or mismatched files:\n' + '\n'.join(errors))
    print(f'All {len(lock)} model snapshots verified: {root}')


if __name__ == '__main__':
    main()
