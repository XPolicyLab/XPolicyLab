"""RGB frames of RoboTwin and RoboDojo recordings, decoded by XPolicyLab.

Stored frames come in more than one byte format, and only XPolicyLab's
``decode_image_bit`` tells them apart. It is loaded from the benchmark checkout
that contains ``XPolicyLab/`` instead of copying its format logic.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np


@lru_cache(maxsize=4)
def xpolicylab_decoder(parent: str) -> Callable[[Any], np.ndarray]:
    package_parent = Path(parent).expanduser().resolve()
    process_data = package_parent / "XPolicyLab" / "utils" / "process_data.py"
    if not process_data.is_file():
        raise FileNotFoundError(
            f"RGB decoding requires XPolicyLab's shared helper; missing {process_data}."
        )
    parent_text = str(package_parent)
    if parent_text not in sys.path:
        sys.path.insert(0, parent_text)
    from XPolicyLab.utils.process_data import decode_image_bit

    return decode_image_bit


def decode_rgb_frame(encoded: Any, parent: str) -> np.ndarray:
    """Decode one stored frame to HWC uint8 RGB."""
    array = np.asarray(xpolicylab_decoder(parent)(encoded), dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"XPolicyLab decode_image_bit must return HWC RGB, got {array.shape}."
        )
    return array
