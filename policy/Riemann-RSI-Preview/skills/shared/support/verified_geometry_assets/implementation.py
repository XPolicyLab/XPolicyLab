"""Only exact external public v23 resources; no original policy or SDK is loaded."""
import hashlib
from rsi.resource_locations import load_manifest
from pathlib import Path





def resolve(root=None, *, declaration_path):
    declaration = load_manifest(declaration_path)
    folder = Path(root or declaration['default_root'])
    if folder.is_symlink():
        raise ValueError('Public resource root must not be a symlink')
    result = {}
    for member, record in declaration['resources'].items():
        relative = Path(member)
        if relative.parts[0] != 'skill' or '..' in relative.parts or relative.is_absolute():
            raise ValueError('Invalid declared public resource member')
        path = folder
        for part in relative.parts[1:]:
            path = path / part
            if path.is_symlink():
                raise ValueError('Declared public resource must not be a symlink: ' + member)
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Public resource escaped its declared root')
        content = path.read_bytes()
        if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['source_sha256']:
            raise ValueError('Public v23 resource hash mismatch: ' + member)
        result[member] = path
    return result
