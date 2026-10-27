"""Read-only path relocation; immutable source manifests keep their original hashes.

ROBOCASA_PATH_MAP is opt-in and affects RoboCasa readers only. It is never a
permission to skip identity checks or to fall back to another data source.
"""
from functools import lru_cache
import json
import os
from pathlib import Path


@lru_cache(maxsize=8)
def _mapping(filename, mtime_ns, size):
    record = json.loads(Path(filename).read_text())
    if record.get('schema') != 'phoenix.robocasa.path_map.v1':
        raise ValueError('Unknown RoboCasa path map schema')
    pairs = record.get('mappings')
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError('RoboCasa path map must contain explicit mappings')
    result = []
    for original, relocated in pairs.items():
        a, b = Path(original), Path(relocated)
        if not a.is_absolute() or not b.is_absolute() or len(a.parts) < 4 or len(b.parts) < 4:
            raise ValueError('Refusing relative or broad path relocation')
        if '..' in a.parts or '..' in b.parts:
            raise ValueError('Path relocation may not contain parent traversal')
        result.append((a, b))
    return sorted(result, key=lambda pair: len(pair[0].parts), reverse=True)


def resolve_source(path):
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Source path may not contain parent traversal')
    filename = os.environ.get('ROBOCASA_PATH_MAP')
    if filename:
        stat = Path(filename).stat()
        for source, destination in _mapping(filename, stat.st_mtime_ns, stat.st_size):
            if path.is_relative_to(source):
                return destination / path.relative_to(source)
    return path


def require_migrated_cache(cache_root, tasks):
    """Check relocated cache availability without transfer receipts."""
    for task in tasks:
        path = Path(cache_root) / "episodes" / task
        if not path.is_dir():
            raise FileNotFoundError(path)
