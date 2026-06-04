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
from dataclasses import dataclass
from typing import Any, Literal

import datasets as hg_datasets
from robo_orchard_core.datatypes.tf_graph import (
    BatchFrameTransformGraph,
    BatchFrameTransformGraphState,
)

from robo_orchard_lab.dataset.datatypes.geometry import (
    BatchFrameTransformFeature,
)
from robo_orchard_lab.dataset.datatypes.hg_features import (
    PickleFeature,
    RODictDataFeature,
    TypedDictFeatureDecode,
    check_fields_consistency,
    hg_dataset_feature,
)

__all__ = [
    "BatchFrameTransformGraph",
    "BatchFrameTransformGraphFeature",
]


def _get(self, key: str, default: Any = None) -> Any:
    """Get the value of the feature by key."""
    return getattr(self, key, default)


@classmethod
def _batch_frame_transform_graph_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchFrameTransformGraphFeature | PickleFeature:
    """A class for frame transform graphs with dataset feature support."""
    if use_pickle:
        return PickleFeature(class_type=BatchFrameTransformGraphState)
    ret = BatchFrameTransformGraphFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchFrameTransformGraphState.dataset_feature = (  # type: ignore
    _batch_frame_transform_graph_dataset_feature
)


@hg_dataset_feature
@dataclass
class BatchFrameTransformGraphStateFeature(
    RODictDataFeature, TypedDictFeatureDecode
):
    """Dataset feature for BatchFrameTransformGraphState."""

    dtype: Literal["float32", "float64"] = "float32"

    decode: bool = True

    def __post_init__(self):
        self._decode_type = BatchFrameTransformGraphState
        self._dict = {
            "tf_list": hg_datasets.features.features.Sequence(
                BatchFrameTransformFeature(dtype=self.dtype)
            ),
            "bidirectional": hg_datasets.Value("bool"),
            "static_tf": hg_datasets.features.features.Sequence(
                hg_datasets.Value("bool")
            ),
        }


@classmethod
def _batch_frame_transform_graph_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchFrameTransformGraphFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchFrameTransformGraph)
    """A class for frame transform graphs with dataset feature support."""
    return BatchFrameTransformGraphFeature(dtype=dtype)


BatchFrameTransformGraph.get = _get  # type: ignore
BatchFrameTransformGraph.dataset_feature = (  # type: ignore
    _batch_frame_transform_graph_dataset_feature
)


@hg_dataset_feature
@dataclass
class BatchFrameTransformGraphFeature(BatchFrameTransformGraphStateFeature):
    """A feature for storing batch frame transform graph in a dataset.

    The underlying data is stored as BatchFrameTransformGraphState.

    """

    def encode_example(self, value: BatchFrameTransformGraph):
        state: BatchFrameTransformGraphState = BatchFrameTransformGraphState(
            **value.as_state().__dict__
        )
        return super().encode_example(state)

    def decode_example(self, value: Any, **kwargs) -> BatchFrameTransformGraph:
        state: BatchFrameTransformGraphState = super().decode_example(
            value, **kwargs
        )
        return BatchFrameTransformGraph.from_state(state)
