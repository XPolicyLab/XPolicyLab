from __future__ import annotations

import numpy as np

from psi_policy.model.common.normalizer import SingleFieldLinearNormalizer


def get_normalizer_from_input_min_max(
    input_min,
    input_max,
    output_min=-1,
    output_max=1,
    range_eps=1e-7,
) -> SingleFieldLinearNormalizer:
    """Create a linear normalizer from explicit input limits."""
    input_min = np.asarray(input_min, dtype=np.float32)
    input_max = np.asarray(input_max, dtype=np.float32)
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range = input_range.copy()
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset = offset.copy()
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    stat = {
        "min": input_min,
        "max": input_max,
        "mean": (input_min + input_max) / 2,
        "std": (input_max - input_min) / np.sqrt(12),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )


def get_image_range_normalizer() -> SingleFieldLinearNormalizer:
    """将 [0,1] 范围图像映射到 [-1,1]。"""
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)
