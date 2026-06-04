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
from dataclasses import dataclass
from io import BytesIO
from typing import TypedDict

import datasets as hg_datasets
import numpy as np
import pyarrow as pa
import torch

from robo_orchard_lab.dataset.datatypes.hg_features.base import (
    FeatureDecodeMixin,
    RODataFeature,
    RODictDataFeature,
    hg_dataset_feature,
)

__all__ = [
    "AnyTensorFeature",
    "TypedTensorFeature",
]


@hg_dataset_feature
@dataclass
class AnyTensorFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing tensors in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the tensor's shape and dtype.
    """

    decode: bool = True
    as_torch_tensor: bool = True

    @property
    def pa_type(self):
        """Return the pyarrow data type for this feature."""
        return pa.binary()

    def encode_example(self, value: np.ndarray | torch.Tensor) -> bytes:
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        buffer = BytesIO()
        np.save(buffer, value, allow_pickle=False)
        tensor_data = buffer.getvalue()
        return tensor_data

    def decode_example(
        self,
        value: bytes,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "TensorFeature(decode=True) instead."
            )
        buffer = BytesIO(value)
        decoded_tensor: np.ndarray = np.load(buffer, allow_pickle=False)
        if self.as_torch_tensor:
            return torch.from_numpy(decoded_tensor)
        return decoded_tensor


class TensorFeatureSerialized(TypedDict):
    data: np.ndarray
    shape: tuple[int, ...]


@hg_dataset_feature
@dataclass
class TypedTensorFeature(RODictDataFeature, FeatureDecodeMixin):
    """A feature for storing typed tensors in a dataset.

    The underlying data is stored as a flattened numpy array
    with shape information.
    """

    dtype: str
    decode: bool = True
    as_torch_tensor: bool = True

    def __post_init__(self):
        self._dict = {
            "data": hg_datasets.features.features.Sequence(
                hg_datasets.features.Value(self.dtype)
            ),
            "shape": hg_datasets.features.Sequence(
                hg_datasets.features.Value("int32")
            ),
        }
        self._warning_issued = False

    def encode_example(
        self, value: np.ndarray | torch.Tensor | None
    ) -> TensorFeatureSerialized | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            value = value.numpy()

        if value.dtype != self.dtype:
            # warn at most once per feature instance
            if not self._warning_issued:
                warnings.warn(
                    f"Input tensor dtype {value.dtype} does not match "
                    f"feature dtype {self.dtype}. Casting.",
                    UserWarning,
                )
                self._warning_issued = True
            value = value.astype(dtype=self.dtype)

        serialized = TensorFeatureSerialized(
            data=value.flatten(),
            shape=value.shape,
        )
        return serialized

    def decode_example(
        self,
        value: TensorFeatureSerialized | None,
        **kwargs,
    ) -> np.ndarray | torch.Tensor | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "TensorFeature(decode=True) instead."
            )
        if value is None:
            return None
        decoded_tensor: np.ndarray = np.array(value["data"], dtype=self.dtype)
        decoded_tensor = decoded_tensor.reshape(value["shape"])
        if self.as_torch_tensor:
            return torch.from_numpy(decoded_tensor)
        return decoded_tensor
