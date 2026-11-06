"""Load the exact public v23 templates; rebuilding USD silhouettes is not an actor capability."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

RASTER_SIZE = 192
RASTER_SCALE = 2400.0

class TemplateBank:
    def __init__(self, directory):
        directory = Path(directory)
        self.records = json.loads((directory/'digits.json').read_text())['records']
        self.masks = np.load(directory/'digits.npz')['masks']
        self._bank = {}

    def rotated(self, family, angles=None):
        angles = tuple(range(-60,61,3)) if angles is None else tuple(angles)
        key = (family,angles)
        if key not in self._bank:
            result = []
            for record in self.records:
                if family is not None and record['family'] != family: continue
                for angle in angles:
                    # Raster axes follow world XY, hence a positive OpenCV angle is negative physical yaw.
                    mask = cv2.warpAffine(self.masks[record['model_id']],
                        cv2.getRotationMatrix2D((RASTER_SIZE/2,RASTER_SIZE/2),-float(angle),1.),
                        (RASTER_SIZE,RASTER_SIZE),flags=cv2.INTER_NEAREST)
                    result.append((record, float(angle), mask))
            self._bank[key] = result
        return self._bank[key]


__all__ = ('TemplateBank', 'RASTER_SIZE', 'RASTER_SCALE')
