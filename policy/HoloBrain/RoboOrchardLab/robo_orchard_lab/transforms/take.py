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
from typing import Any, Sequence

from pydantic import AliasChoices, Field

from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "TakeKeys",
    "TakeKeysConfig",
]


class TakeKeys(DictTransform[dict[str, Any]]):
    """A transform to take only the specified keys from the input dict."""

    cfg: TakeKeysConfig

    def __init__(self, cfg: TakeKeysConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs: Any) -> dict[str, Any]:
        return dict(**kwargs)


class TakeKeysConfig(DictTransformConfig[TakeKeys]):
    """Configuration class for TakeKeysTransform.

    User should specify the keys to take from the input dict by setting
    `input_columns`.

    """

    class_type: ClassType[TakeKeys] = TakeKeys

    input_columns: Sequence[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )
    """The keys to take from the input dict."""

    keep_input_columns: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.input_columns is None:
            raise ValueError("input_columns must be provided.")
        if self.keep_input_columns is True:
            raise ValueError(
                "keep_input_columns should be set to False "
                "for TakeKeysTransform."
            )
