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
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Sequence, TypeVar

import numpy as np
import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    load_from,
)

from robo_orchard_lab.utils.misc import stack_batch

__all__ = [
    "Collator",
    "CollatorConfig",
    "CollatorT_co",
    "collate_batch_dict",
]


def collate_batch_dict(batch: Sequence[dict]) -> dict[str, Any]:
    output = dict()
    assert all([isinstance(x, Dict) for x in batch])
    for key in batch[0].keys():
        elements = [x[key] for x in batch]
        if isinstance(elements[0], torch.Tensor):
            output[key] = stack_batch(elements)
        elif isinstance(elements[0], np.ndarray):
            output[key] = stack_batch([torch.from_numpy(x) for x in elements])
        elif isinstance(elements[0], float):
            output[key] = torch.tensor(elements, dtype=torch.float64)
        elif isinstance(elements[0], int):
            output[key] = torch.tensor(elements)
        elif isinstance(elements[0], list):
            if len(elements[0]) != 0 and isinstance(
                elements[0][0], np.ndarray
            ):
                output[key] = [
                    [torch.from_numpy(np.array(sample)) for sample in b]
                    for b in elements
                ]
            else:
                output[key] = elements
        elif isinstance(elements[0], Dict):
            output[key] = collate_batch_dict(elements)
        else:
            output[key] = elements
    return output


class Collator(metaclass=ABCMeta):
    """Base class for collators that process batches of data."""

    cfg: CollatorConfig
    InitFromConfig: bool = True

    @abstractmethod
    def __call__(self, batch: Sequence[dict]) -> dict[str, Any]:
        """Process a batch of data and return a collated dictionary."""
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def from_config(cls, config_path: str) -> Collator:
        """Load a Collator from a configuration file."""
        cfg = load_from(config_path, ensure_type=CollatorConfig)
        return cfg()


CollatorT_co = TypeVar("CollatorT_co", bound=Collator, covariant=True)


class CollatorConfig(ClassConfig[CollatorT_co]):
    """Configuration class for Collator."""

    class_type: ClassType[CollatorT_co]
