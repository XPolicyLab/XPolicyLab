"""Verify declared external model members without deployment side effects."""
from pathlib import Path
import hashlib


def verify_records(root, records, deep=True):
    for relative, record in records.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Manifest paths must remain inside their declared root')
        target = root / relative
        if not target.is_file() or target.stat().st_size != record['bytes']:
            raise ValueError(f'Missing or incomplete policy dependency: {target}')
        if deep:
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != record['sha256']:
                raise ValueError(f'Policy dependency digest changed: {target}')
    return len(records)
