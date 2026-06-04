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
from typing import Any, Literal, Sequence

from pydantic import AliasChoices, Field

from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "PaddingList",
    "PaddingListConfig",
]


class PaddingList(DictTransform[dict[str, list[Any]]]):
    """A transform that applies padding to input lists.

    This transform supports padding input lists by replacing None values in the
    head and tail of the list with the first and last non-None values found in
    the list, respectively.

    """

    cfg: PaddingListConfig

    def __init__(self, cfg: PaddingListConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs: list[Any]) -> dict[str, list[Any]]:
        """Apply padding to the input columns.

        This method replaces None values in the head of input lists with the
        first non-None value found in the list for each column, and replaces
        None values in the tail of input lists with the last non-None value
        found in the list for each column.

        """
        ret = {}

        def replace_none(v: list):
            # find the first and last non-None values in the list
            first_not_none_idx = next(
                (i for i, x in enumerate(v) if x is not None), None
            )

            last_not_none_idx = next(
                (i for i, x in enumerate(reversed(v)) if x is not None), None
            )
            if last_not_none_idx is not None:
                last_not_none_idx = len(v) - 1 - last_not_none_idx
            # replace None values before the first non-None value
            # and after the last non-None value
            if first_not_none_idx is not None:
                for i in range(first_not_none_idx):
                    v[i] = v[first_not_none_idx]
            if last_not_none_idx is not None:
                for i in range(last_not_none_idx + 1, len(v)):
                    v[i] = v[last_not_none_idx]
            return v

        for k, v in kwargs.items():
            if not isinstance(v, list):
                raise TypeError(
                    f"Expected list for column '{k}', but got {type(v)}."
                )
            if len(v) == 0:
                raise ValueError(
                    f"Column '{k}' is empty. Padding cannot be applied."
                )
            ret[k] = replace_none(v)
        return ret


class PaddingListConfig(DictTransformConfig[PaddingList]):
    """Configuration class for Padding transform.

    This configuration defines the padding size and the padding value.
    It is used to create a Padding transform that pads the input tensors.

    """

    class_type: ClassType[PaddingList] = PaddingList

    input_columns: Sequence[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
        description="The input columns to apply padding to.",
    )

    pad_mode: Literal["replace_none"] = "replace_none"
    """The padding mode to use. Currently only 'replace_none' is supported.

    -replace_none: Replace None values in the head and tail of input lists
        with the first and last non-None values found in the list,
        respectively.

    """

    check_return_columns: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.input_columns is None:
            raise ValueError("input_columns must be provided.")
