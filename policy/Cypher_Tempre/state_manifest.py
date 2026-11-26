"""Record/copy existing state without changing its live chain or faculties."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
from .client import chain_tip


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skill-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--private-archive', type=Path)
    a = p.parse_args()
    root, output = a.skill_root.resolve(), a.output.resolve()
    if output.is_relative_to(root) or (a.private_archive and a.private_archive.resolve().is_relative_to(root)):
        p.error('Receipts/archive must be outside the live skill.')
    if output.exists() or (a.private_archive and a.private_archive.exists()):
        p.error('A provenance artifact already exists; use a new output name.')
    tip = chain_tip(root)
    entries = {}
    for path in sorted(root.rglob('*')):
        if any(part in {'.git', '__pycache__'} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Review linked state explicitly before archiving: ' + str(path))
        if path.is_file() and path.suffix != '.pyc':
            entries[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    archive = None
    if a.private_archive:
        archive = a.private_archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for name, info in entries.items():
                data = (root / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != info['sha256']:
                    raise RuntimeError('Live state changed during archive; preserve failed artifact and retry with a new name.')
                z.writestr(name, data)
    if tip != chain_tip(root) or any(digest(root / name) != info['sha256'] for name, info in entries.items()):
        raise RuntimeError('Live state changed during snapshot; no consistent receipt is claimed.')
    out = {'recorded_at': time.time(), 'purpose': 'Provenance only; the existing live agent continues learning.',
           'skill_root': str(root), 'chain_tip': tip, 'files': entries,
           'private_archive': str(archive) if archive else None,
           'archive_sha256': digest(archive) if archive else None,
           'live_state_modified': False, 'empty_state_substituted': False,
           'publication_status': 'Private accumulated state; not uploaded. Public release requires content/provenance review.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(entries), 'chain_tip': tip, 'archive_sha256': out['archive_sha256'], 'live_state_modified': False}, indent=2))


if __name__ == '__main__':
    main()
