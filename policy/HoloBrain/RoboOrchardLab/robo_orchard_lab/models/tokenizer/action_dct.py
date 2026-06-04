# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import warnings

import numpy as np
import torch
from scipy.fft import dct, idct

from robo_orchard_lab.models.codec import (
    ClassType,
    Codec as CodecMixin,
    CodecConfig,
    NormStatistics,
)

__all__ = [
    "ActionDCTCodec",
    "ActionDCTCodecConfig",
    "NormStatistics",
    "calculate_norm_stats",
]


class ActionDCTCodec(CodecMixin):
    """A codec for action data using DCT normalization.

    Action DCT is used in pi0-Fast by Physical-Intelligence
    (https://arxiv.org/pdf/2501.09747).

    This is an implementation based on:
    https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py


    """

    cfg: ActionDCTCodecConfig

    def __init__(self, cfg: ActionDCTCodecConfig):
        self.cfg = cfg

    @property
    def dct_coeff_min(self) -> float | int | None:
        """Get the minimum value for the DCT coefficients."""
        return self.cfg.dct_coeff_min

    @property
    def quantize_scale(self) -> int | None:
        """Get the quantization scale for the DCT coefficients."""
        return self.cfg.quantize_scale

    def encode(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode the action using DCT normalization.

        Args:
            data (np.ndarray|torch.Tensor): The action to encode. Should be
                in size of [B, T, D] or [T, D] where B is the batch size,
                T is the time window size, and D is the action dimension.
                The DCT always operates on the time dimension (T).

        Returns:
            np.ndarray: The encoded action.
        """

        action = data if isinstance(data, np.ndarray) else data.numpy()
        if action.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"Action dimension {action.shape[-1]} does not match "
                f"codec action dimension {self.cfg.action_dim}."
            )
        if (
            self.cfg.time_window_size is not None
            and action.shape[-2] != self.cfg.time_window_size
        ):
            raise ValueError(
                f"Action time window size {action.shape[-2]} does not match "
                f"codec time window size {self.cfg.time_window_size}."
            )

        dct_action: np.ndarray = dct(action, norm="ortho", axis=-2)  # type: ignore

        dct_coeff_min = self.cfg.dct_coeff_min

        if self.cfg.quantize_scale is not None:
            dct_action = np.round(dct_action * self.cfg.quantize_scale).astype(
                np.int32
            )
        if dct_coeff_min is not None:
            dct_action -= dct_coeff_min

        return torch.from_numpy(dct_action)

    def decode(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Decode the DCT normalized action back to the original action.

        Args:
            data (np.ndarray): The DCT normalized action to decode.

        Returns:
            np.ndarray: The decoded action.
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if data.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"DCT action dimension {data.shape[-1]} does not match "
                f"codec action dimension {self.cfg.action_dim}."
            )
        if (
            self.cfg.time_window_size is not None
            and data.shape[-2] != self.cfg.time_window_size
        ):
            raise ValueError(
                f"DCT action time window size {data.shape[-2]} does not match "
                f"codec time window size {self.cfg.time_window_size}."
            )

        if self.cfg.dct_coeff_min is not None:
            data = data + self.cfg.dct_coeff_min

        if self.cfg.quantize_scale is not None:
            data = data.astype(np.float32) / self.cfg.quantize_scale

        idct_action = idct(data, norm="ortho", axis=-2)

        return torch.asarray(idct_action)


class ActionDCTCodecConfig(CodecConfig[ActionDCTCodec]):
    class_type: ClassType[ActionDCTCodec] = ActionDCTCodec

    action_dim: int
    "The dimension of the action data."

    time_window_size: int | None = None
    """The size of the time window for the action.

    If None, the time window size will be inferred from the data.
    If specified, check the data shape to match this size.
    """

    quantize_scale: int | None = None
    """The scale to quantize the DCT coefficients.

    If specified, the coefficients will be rounded to the nearest multiple
    of `quantize_scale` and returned as an integer array."""

    dct_coeff_min: float | int | None = None
    """The minimum value for the DCT coefficients.

    If specified, the DCT coefficients will be shifted by this value
    after encoding and before decoding. If None, no shifting is applied.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.quantize_scale is not None and self.dct_coeff_min is not None:
            if isinstance(self.dct_coeff_min, float):
                warnings.warn(
                    "DCT coefficients are quantized to int32, "
                    "while dct_coeff_min is a float. "
                    "dct_coeff_min will be converted to int32."
                )
                self.dct_coeff_min = int(self.dct_coeff_min)


def calculate_norm_stats(
    arr: np.ndarray,
    quantize_scale=128,
    time_window_size: int | None = None,
) -> NormStatistics:
    """Calculate the min and max values  based on the given percentiles.

    Args:
        arr (np.array): The input array. Should be of shape
            (batch, timesteps, joints).
        q (tuple): A tuple containing the percentiles to calculate.
        quantize_scale (int): The scale to quantize the min and max values.
            When quantizing, the values are rounded to the nearest multiple
            of 1/quantize_scale.
        time_window_size (int, optional): If specified, the array will be
            sliced to this time window before calculating min and max.
            Default is None,

    Returns:
        tuple: The min and max values based on the specified percentiles.
    """
    if time_window_size is not None:
        arr = arr[:, :time_window_size, :]

    v_min, q01, q99, v_max = np.quantile(
        arr, q=(0, 0.01, 0.99, 1), axis=(0, 1)
    )
    # Quantize v_min to be a multiple of 1/quantize_scale
    v_min = torch.from_numpy(np.round(v_min * quantize_scale) / quantize_scale)
    q01 = torch.from_numpy(np.round(q01 * quantize_scale) / quantize_scale)
    q99 = torch.from_numpy(np.round(q99 * quantize_scale) / quantize_scale)
    v_max = torch.from_numpy(np.round(v_max * quantize_scale) / quantize_scale)
    # return v_min, v_max

    return NormStatistics(q01=q01, q99=q99, min=v_min, max=v_max)
