"""Opt-in capture of permitted camera observations for failed-geometry replay."""
import os
from pathlib import Path
import time
import numpy as np


def capture_failure(observation, error):
    destination = os.environ.get('PHYSICALRSI_GEOMETRY_EVIDENCE_DIR')
    if not destination:
        return
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {'error': str(error)}
    for name, camera in observation['vision'].items():
        for field in ('color','intrinsic_matrix','extrinsic_matrix'):
            arrays[f'vision/{name}/{field}'] = np.asarray(camera[field])
    np.savez_compressed(directory/f'failure-{time.time_ns()}.npz', **arrays)
