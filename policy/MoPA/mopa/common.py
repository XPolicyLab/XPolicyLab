"""Joint layout, RGB preprocessing and quantile normalization."""

from __future__ import annotations

import numpy as np
from PIL import Image

FORMAT_VERSION = "xpl-mopa-arm-v1"
DATASET_FORMAT = "xpl-mopa-npz-v1"
DEFAULT_CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def joint_fields(dim_info):
    arms = dim_info["arm_dim"]
    grippers = dim_info["ee_dim"]
    if len(arms) not in (1, 2) or len(arms) != len(grippers):
        raise ValueError("Expected matching dimensions for one or two arms/grippers")
    prefixes = [""] if len(arms) == 1 else ["left_", "right_"]
    fields = []
    for prefix, arm_dim, ee_dim in zip(prefixes, arms, grippers):
        for name, dim in (("arm_joint_state", arm_dim), ("ee_joint_state", ee_dim)):
            if int(dim) != dim or dim <= 0:
                raise ValueError(f"Invalid dimension for {prefix}{name}: {dim}")
            fields.append((prefix + name, int(dim)))
    return fields


def pack_joint(mapping, dim_info, plural=False):
    arrays = []
    for key, dim in joint_fields(dim_info):
        key = key + "s" if plural else key
        value = np.asarray(mapping[key], dtype=np.float32)
        if value.ndim != (2 if plural else 1) or value.shape[-1] != dim:
            raise ValueError(f"{key}: expected {'[T,' if plural else '['}{dim}], got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
        arrays.append(value)
    return np.concatenate(arrays, axis=-1)


def unpack_joint(vector, dim_info):
    vector = np.asarray(vector, dtype=np.float32)
    fields = joint_fields(dim_info)
    if vector.shape != (sum(dim for _, dim in fields),) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid joint action vector: shape={vector.shape}")
    result, offset = {}, 0
    for key, dim in fields:
        result[key] = vector[offset:offset + dim].copy()
        offset += dim
    return result


def validate_statistics(stats, dim):
    lower = np.asarray(stats["q01"], dtype=np.float32)
    upper = np.asarray(stats["q99"], dtype=np.float32)
    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValueError(f"Normalization statistics must have dimension {dim}")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
        raise ValueError("Normalization statistics must be finite with q99 >= q01")
    return lower, upper


def normalize(array, stats):
    array = np.asarray(array, dtype=np.float32)
    lower, upper = validate_statistics(stats, array.shape[-1])
    span = upper - lower
    normalized = 2 * (array - lower) / np.where(span > 1e-6, span, 1.0) - 1
    return np.where(span > 1e-6, np.clip(normalized, -1, 1), 0).astype(np.float32)


def denormalize(array, stats):
    array = np.asarray(array, dtype=np.float32)
    lower, upper = validate_statistics(stats, array.shape[-1])
    result = (np.clip(array, -1, 1) + 1) * (upper - lower) / 2 + lower
    return result.astype(np.float32)


def rgb_image(array, image_size):
    """Resize already decoded RGB, identically during conversion and eval."""
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Expected decoded uint8 RGB [H,W,3], got {array.shape}/{array.dtype}")
    if len(image_size) != 2 or any(int(v) <= 0 for v in image_size):
        raise ValueError("image_size must be [height, width] with positive sizes")
    height, width = map(int, image_size)
    return Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
