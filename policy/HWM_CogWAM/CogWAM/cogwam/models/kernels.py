"""Optional attention-kernel detection.

RynnBrain1.1 interleaves linear-attention layers with full attention, so the
available kernels change what the backbone can run. Detection is import-based
rather than version-based: a wheel that imports is a wheel that works, and the
package versions that matter are pinned in requirements.txt.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def has_flash_attn() -> bool:
    """Return True when a flash-attention backend is importable.

    Covers both the CUDA (``flash_attn``) and Ascend NPU (``torch_npu``)
    backends so the VLM can fall back to SDPA instead of failing to build.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    else:
        return True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True
