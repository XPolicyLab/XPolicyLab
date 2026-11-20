"""Isolate a public image region using its public reference background."""
import numpy as np


def run(image, background, mask):
    image,background,mask=(np.asarray(v) for v in (image,background,mask))
    if (image.dtype!=np.uint8 or background.dtype!=np.uint8 or image.ndim!=3
            or image.shape[-1]!=3 or background.shape!=image.shape
            or mask.dtype!=np.bool_ or mask.shape!=image.shape[:2]):
        raise ValueError('Matched uint8 public RGB and Boolean image mask required')
    result=background.copy()
    result[mask]=image[mask]
    return result
