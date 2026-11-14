"""Reseal this policy's delivery manifest after intentional packaging edits."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
excluded = {'.git', '.venv', '__pycache__', 'checkpoints', 'weights'}
rows = {}
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if (not path.is_file() or excluded.intersection(relative.parts)
            or relative.as_posix() == 'CHECKSUMS.json' or path.suffix in {'.pyc', '.pyo', '.partial'}):
        continue
    if not path.resolve().is_relative_to(root):
        raise ValueError('External file in source package: ' + str(relative))
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    rows[relative.as_posix()] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
(root / 'CHECKSUMS.json').write_text(json.dumps({'algorithm': 'sha256', 'files': rows}, indent=2) + '\n')
print('Sealed', len(rows), 'files; the manifest does not include itself.')
