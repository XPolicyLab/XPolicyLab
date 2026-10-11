"""Preserve RGB inputs without overlays/resizing; never evaluate task rewards."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')

def _json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)

def save_raw_observation(directory, prefix, obs, native_step, context=None):
    if Path(prefix).name != prefix or type(native_step) is not int or native_step < 0:
        raise ValueError('Invalid observation identity')
    arrays = {camera: np.asarray(obs['vision'][camera]['color']) for camera in CAMERAS}
    for image in arrays.values():
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype.hasobject:
            raise ValueError('Expected unmodified numeric RGB arrays')
    metadata = dict(native_step=native_step, context=context or {},
                    instruction=obs.get('instruction'), state=obs.get('state'),
                    calibration=obs.get('calibration'),
                    cameras={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                       pixel_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                       for name, value in arrays.items()})
    # Validate metadata before creating files; never stringify unsupported data.
    json.dumps(metadata, default=_json_value)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (prefix + '_raw.npz')
    sidecar = directory / (prefix + '_raw.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Observation source already preserved')
    temporary = output.with_suffix('.npz.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)
    metadata['archive_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    with sidecar.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, default=_json_value, indent=2)
    return dict(archive=str(output), metadata=str(sidecar), sha256=metadata['archive_sha256'])
