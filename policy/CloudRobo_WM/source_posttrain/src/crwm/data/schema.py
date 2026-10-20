from __future__ import annotations
import hashlib
import json
from pathlib import Path
STATE_DIM, ACTION_DIM = (96, 90)
SCHEMA_VERSION = 'crwm.s96_a90.v1'
CODEC_VERSION = 'observed_anchor.direct_xyz.body_continuous_rotvec.v1'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
STATE = {'left.position': slice(0, 3), 'left.rotation': slice(3, 9), 'left.gripper': slice(9, 10), 'left.hand': slice(10, 34), 'right.position': slice(34, 37), 'right.rotation': slice(37, 43), 'right.gripper': slice(43, 44), 'right.hand': slice(44, 68), 'base': slice(68, 80), 'left.joints': slice(80, 88), 'right.joints': slice(88, 96)}
ACTION = {'left.position': slice(0, 3), 'left.rotation': slice(3, 6), 'left.gripper': slice(6, 7), 'left.hand': slice(7, 31), 'right.position': slice(31, 34), 'right.rotation': slice(34, 37), 'right.gripper': slice(37, 38), 'right.hand': slice(38, 62), 'base': slice(62, 74), 'left.joints': slice(74, 82), 'right.joints': slice(82, 90)}

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def within(root, relative) -> Path:
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root):
        raise ValueError(f'Manifest path escapes dataset: {relative}')
    return result

def atomic_json(path, payload):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
