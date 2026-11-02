from __future__ import annotations

from typing import Sequence

import numpy as np


def shape_to_dim(shape: Sequence[int]) -> int:
    values = tuple(int(v) for v in shape)
    if len(values) == 0:
        return 1
    return int(np.prod(values, dtype=np.int64))


def to_float_2d_numpy(arr: np.ndarray, *, key: str, expected_dim: int) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if data.ndim != 2:
        raise ValueError(f"{key} 期望 2D 数据，实际 shape={data.shape}")

    if data.shape[1] != expected_dim:
        raise ValueError(f"{key} 维度不匹配: expected={expected_dim}, got={data.shape[1]}")

    return data.astype(np.float32, copy=False)


def normalize_image_size(
    image_size: int | Sequence[int] | None,
    *,
    default_image_size: Sequence[int] | None = None,
) -> tuple[int, int]:
    if image_size is None:
        image_size = default_image_size
    if image_size is None:
        image_size = (224, 224)

    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        values = list(image_size)
        if len(values) != 2:
            raise ValueError("image_size 列表长度必须为 2")
        height = int(values[0])
        width = int(values[1])
    else:
        size = int(image_size)
        height = size
        width = size

    if height <= 0 or width <= 0:
        raise ValueError("image_size 必须为正整数")
    return (height, width)


__all__ = ["normalize_image_size", "shape_to_dim", "to_float_2d_numpy"]
