"""Joint layouts, RGB preprocessing and dataset normalization."""

from __future__ import annotations

import numpy as np
from PIL import Image

FORMAT_VERSION = "xpl-mopa-query-dmot-v2"
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
    # Prefer exact extrema when available; generic datasets use quantiles.
    lower_key, upper_key = (("min", "max") if "min" in stats and "max" in stats
                            else ("q01", "q99"))
    lower = np.asarray(stats[lower_key], dtype=np.float32)
    upper = np.asarray(stats[upper_key], dtype=np.float32)
    if lower.shape != (dim,) or upper.shape != (dim,):
        raise ValueError(f"Normalization statistics must have dimension {dim}")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper < lower):
        raise ValueError("Normalization statistics must be finite with q99 >= q01")
    return lower, upper


def _affine_parameters(stats, dim):
    lower, upper = validate_statistics(stats, dim)
    minmax = "min" in stats and "max" in stats
    if minmax:
        span = np.maximum(upper - lower, 1e-6)
        return lower + 0.5 * span, 0.5 * span, True
    span = upper - lower
    return lower, span, False


def normalize(array, stats):
    array = np.asarray(array, dtype=np.float32)
    offset, scale, minmax = _affine_parameters(stats, array.shape[-1])
    if minmax:
        # Keep the affine transform invertible for out-of-range predictions.
        return ((array - offset) / scale).astype(np.float32)
    normalized = 2 * (array - offset) / np.where(scale > 1e-6, scale, 1.0) - 1
    return np.where(scale > 1e-6, np.clip(normalized, -1, 1), 0).astype(np.float32)


def denormalize(array, stats):
    array = np.asarray(array, dtype=np.float32)
    offset, scale, minmax = _affine_parameters(stats, array.shape[-1])
    if minmax:
        return (array * scale + offset).astype(np.float32)
    return (np.clip(array, -1, 1) + 1) * scale / 2 + offset


def rgb_image(array, image_size):
    """Resize already decoded RGB, identically during conversion and eval."""
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Expected decoded uint8 RGB [H,W,3], got {array.shape}/{array.dtype}")
    if len(image_size) != 2 or any(int(v) <= 0 for v in image_size):
        raise ValueError("image_size must be [height, width] with positive sizes")
    height, width = map(int, image_size)
    return Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
