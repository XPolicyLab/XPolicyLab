"""Portable JSON/NPZ representation of authenticated public static geometry."""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA = 'public-static-numeric-tree-v1'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(directory, trees, *, source_root, source_files, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source_root = str(Path(source_root))
    arrays = {}
    def encode(value):
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            if array.dtype.kind not in 'biufc' or not np.isfinite(array).all():
                raise ValueError('Static geometry requires finite numeric arrays')
            raw = np.ascontiguousarray(array).tobytes()
            name = 'a_' + hashlib.sha256(array.dtype.str.encode() + repr(array.shape).encode() + raw).hexdigest()
            arrays[name] = array.copy()
            return {'kind': 'scalar' if isinstance(value, np.generic) else 'array', 'name': name}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value): raise ValueError('Expected string model keys')
            return {'kind': 'dict', 'items': {key: encode(item) for key, item in value.items()}}
        if isinstance(value, (tuple, list)):
            return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'items': [encode(item) for item in value]}
        if isinstance(value, str) and value.startswith(source_root + '/'):
            return {'kind': 'source_path', 'relative': value[len(source_root) + 1:]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return {'kind': 'value', 'value': value}
        raise TypeError('Unsupported public model value: ' + type(value).__name__)
    encoded = {name: encode(tree) for name, tree in trees.items()}
    archive = directory / 'arrays.npz'
    np.savez_compressed(archive, **arrays)
    manifest = {'schema': SCHEMA, 'archive': 'arrays.npz', 'archive_sha256': digest(archive),
                'arrays': {name: {'dtype': value.dtype.str, 'shape': list(value.shape)} for name, value in arrays.items()},
                'trees': encoded, 'source_files': source_files, 'provenance': provenance,
                'public_static_data_only': True, 'scene_layout_or_observation_included': False}
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return path, digest(path)


def load(manifest_path, tree_name, *, expected_manifest_sha256, source_root=None):
    path = Path(manifest_path)
    if path.is_symlink() or digest(path) != expected_manifest_sha256:
        raise ValueError('Public static geometry manifest changed')
    manifest = json.loads(path.read_text())
    relative = Path(manifest['archive'])
    archive = path.parent / relative
    if (manifest['schema'] != SCHEMA or manifest.get('public_static_data_only') is not True
            or manifest.get('scene_layout_or_observation_included') is not False
            or relative.is_absolute() or '..' in relative.parts or archive.is_symlink()
            or not archive.resolve().is_relative_to(path.parent.resolve())
            or digest(archive) != manifest['archive_sha256']):
        raise ValueError('Public static geometry archive changed')
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != set(manifest['arrays']):
            raise ValueError('Public static geometry array membership changed')
        def decode(node):
            kind = node['kind']
            if kind in ('array', 'scalar'):
                value = data[node['name']]
                declared = manifest['arrays'][node['name']]
                if (value.dtype.kind not in 'biufc' or value.dtype.str != declared['dtype']
                        or list(value.shape) != declared['shape'] or not np.isfinite(value).all()):
                    raise ValueError('Public static geometry array format changed')
                return value[()] if kind == 'scalar' else value.copy()
            if kind == 'dict': return {key: decode(value) for key, value in node['items'].items()}
            if kind in ('tuple', 'list'):
                values = [decode(value) for value in node['items']]
                return tuple(values) if kind == 'tuple' else values
            if kind == 'source_path':
                relative = Path(node['relative'])
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid public source identity')
                return str(Path(source_root) / relative) if source_root is not None else 'public-cad:' + str(relative)
            if kind == 'value': return node['value']
            raise ValueError('Unknown public geometry encoding')
        return decode(manifest['trees'][tree_name])
