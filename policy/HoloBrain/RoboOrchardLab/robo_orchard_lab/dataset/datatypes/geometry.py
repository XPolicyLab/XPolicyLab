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
from dataclasses import Field, dataclass
from typing import Literal

import datasets as hg_datasets
from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
    BatchPose,
    BatchTransform3D,
)

from robo_orchard_lab.dataset.datatypes.hg_features import (
    PickleFeature,
    RODictDataFeature,
    TypedDictFeatureDecode,
    check_fields_consistency,
    hg_dataset_feature,
)
from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
    TypedTensorFeature,
)

__all__ = [
    "BatchTransform3DFeature",
    "BatchTransform3D",
    "BatchPoseFeature",
    "BatchPose",
    "BatchFrameTransformFeature",
    "BatchFrameTransform",
]


@classmethod
def _batch_transform_3d_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchTransform3DFeature | PickleFeature:
    """A class for 3D transforms with dataset feature support."""
    if use_pickle:
        return PickleFeature(class_type=BatchTransform3D)
    ret = BatchTransform3DFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchTransform3D.dataset_feature = (  # type: ignore
    _batch_transform_3d_dataset_feature
)


@hg_dataset_feature
@dataclass
class BatchTransform3DFeature(RODictDataFeature, TypedDictFeatureDecode):
    """A feature for storing batch frame transforms in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the frame transforms.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._decode_type = BatchTransform3D
        typed_tensor_feature = TypedTensorFeature(
            dtype=self.dtype, as_torch_tensor=True
        )
        state = {
            "xyz": typed_tensor_feature,
            "quat": typed_tensor_feature,
            "timestamps": hg_datasets.Sequence(hg_datasets.Value("int64")),
        }
        if not hasattr(self, "_dict"):
            self._dict = {}
        if isinstance(self._dict, Field):
            self._dict = {}

        self._dict.update(state)


@classmethod
def _batch_pose_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchPoseFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchPose)
    ret = BatchPoseFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchPose.dataset_feature = _batch_pose_dataset_feature  # type: ignore


@hg_dataset_feature
@dataclass
class BatchPoseFeature(BatchTransform3DFeature):
    """A feature for storing batch poses in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the poses.
    """

    def __post_init__(self):
        super().__post_init__()
        self._decode_type = BatchPose
        self._dict["frame_id"] = hg_datasets.Value("string")


@classmethod
def _batch_frame_transform_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchFrameTransformFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchFrameTransform)
    ret = BatchFrameTransformFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchFrameTransform.dataset_feature = (  # type: ignore
    _batch_frame_transform_dataset_feature
)


@hg_dataset_feature
@dataclass
class BatchFrameTransformFeature(BatchTransform3DFeature):
    """A feature for storing batch frame transforms in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the frame transforms.
    """

    def __post_init__(self):
        super().__post_init__()
        self._decode_type = BatchFrameTransform
        self._dict["parent_frame_id"] = hg_datasets.Value("string")
        self._dict["child_frame_id"] = hg_datasets.Value("string")
