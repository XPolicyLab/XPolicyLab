from typing import Any

import numpy as np
from PIL import Image


def to_pil_preserve(images: Any, scale_float: bool = True):
    """
    Convert (possibly nested) numpy image arrays back to PIL.Image WITHOUT changing spatial shape
    or nesting structure.

    Accepts:
      - np.ndarray with shape (H, W, C), C in {1,3,4}, dtype uint8 or float
      - PIL.Image.Image (returned as-is)
      - Nested list / tuple structures containing the above

    Guarantees:
      - No resize / pad / crop performed
      - Returns an object with the SAME nesting layout (list -> list, tuple -> tuple)
      - Only dtype (float -> uint8) and channel-mode adaptation may happen
        * float arrays assumed in [0,1] if scale_float=True (scaled *255 + clip)
    Args:
      images: input object / sequence
      scale_float: whether to scale float images in [0,1] to uint8
    Returns:
      Mirrored structure with all leaf nodes as PIL.Image.Image
    """

    def _convert(obj):
        # Nested containers
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)

        # PIL stays
        if isinstance(obj, Image.Image):
            return obj

        # numpy -> PIL
        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D array (H,W,C), got shape={arr.shape}")
            if arr.shape[2] not in (1, 3, 4):
                raise ValueError(f"Channel count must be 1/3/4, got {arr.shape[2]}")
            if np.issubdtype(arr.dtype, np.floating):
                if scale_float:
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    raise TypeError("Float array provided but scale_float=False")
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # Single channel -> 'L'
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
                return Image.fromarray(arr, mode="L")
            # 3 channels -> RGB, 4 -> RGBA
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            return Image.fromarray(arr, mode=mode)

        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)
