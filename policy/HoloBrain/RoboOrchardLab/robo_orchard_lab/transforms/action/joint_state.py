# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
from typing import Generic, Literal

import torch
from pydantic import AliasChoices, Field

from robo_orchard_lab.dataset.datatypes import BatchJointsState
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)
from robo_orchard_lab.transforms.noise import (
    GaussianNoiseConfig,
    NoiseConfigT_co,
    UniformNoiseConfig,
)

__all__ = [
    "AddNoiseToJointsState",
    "AddNoiseToJointsStateConfig",
    "UniformNoiseConfig",
    "GaussianNoiseConfig",
    "UpdateVelocity",
    "UpdateVelocityConfig",
    "ToDeltaJointsState",
    "ToDeltaJointsStateConfig",
]


class AddNoiseToJointsState(
    DictTransform[dict[str, BatchJointsState]], Generic[NoiseConfigT_co]
):
    cfg: AddNoiseToJointsStateConfig[NoiseConfigT_co]

    def __init__(self, cfg: AddNoiseToJointsStateConfig[NoiseConfigT_co]):
        super().__init__()
        self.cfg = cfg

    def transform(
        self, **kwargs: BatchJointsState
    ) -> dict[str, BatchJointsState]:
        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchJointsState):
                ret[key] = self.add_noise(value)
            else:
                raise TypeError(
                    f"Expected BatchJointsState, got {type(value)} for "
                    f"key '{key}'"
                )
        return ret

    def add_noise(self, joints_state: BatchJointsState) -> BatchJointsState:
        if self.cfg.inplace:
            out = joints_state
        else:
            out = joints_state.model_copy(deep=True)

        tensor: torch.Tensor = getattr(out, self.cfg.apply_to)
        tensor += self.cfg.noise.generate_noise(target_shape=tensor.shape)
        return out


class AddNoiseToJointsStateConfig(
    DictTransformConfig[AddNoiseToJointsState[NoiseConfigT_co]],
    Generic[NoiseConfigT_co],
):
    class_type: ClassType[AddNoiseToJointsState[NoiseConfigT_co]] = (
        AddNoiseToJointsState
    )

    noise: NoiseConfigT_co

    inplace: bool = True

    apply_to: Literal["position", "velocity"] = "position"

    def __post_init__(self):
        super().__post_init__()
        assert self.apply_to in BatchJointsState.model_fields, (
            f"apply_to must be one of {BatchJointsState.model_fields}, "
            f"got {self.apply_to}"
        )


class UpdateVelocity(DictTransform[dict[str, BatchJointsState]]):
    cfg: UpdateVelocityConfig

    def __init__(self, cfg: UpdateVelocityConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _update(self, joints_state: BatchJointsState) -> BatchJointsState:
        if self.cfg.inplace:
            out = joints_state
        else:
            out = joints_state.model_copy(deep=True)

        if out.position is not None:
            out.update_velocity(override=self.cfg.override_existing)

        return out

    def transform(
        self, **kwargs: BatchJointsState
    ) -> dict[str, BatchJointsState]:
        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchJointsState):
                ret[key] = self._update(value)
            else:
                raise TypeError(
                    f"Expected BatchJointsState, got {type(value)} "
                    f"for key '{key}'"
                )
        return ret


class UpdateVelocityConfig(DictTransformConfig[UpdateVelocity]):
    class_type: ClassType[UpdateVelocity] = UpdateVelocity

    input_columns: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )

    override_existing: bool = False
    inplace: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.input_columns is None:
            raise ValueError("input_columns must be provided.")


@dataclass
class ToDeltaJointsStateReturn:
    delta: BatchJointsState


class ToDeltaJointsState(DictTransform[ToDeltaJointsStateReturn]):
    cfg: ToDeltaJointsStateConfig

    def __init__(self, cfg: ToDeltaJointsStateConfig | None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ToDeltaJointsStateConfig()
        self.cfg = cfg

    def transform(
        self, current: BatchJointsState, future: BatchJointsState
    ) -> ToDeltaJointsStateReturn:
        ret = BatchJointsState(
            position=self._delta_tensor(current.position, future.position),
            velocity=self._delta_tensor(current.velocity, future.velocity),
            effort=self._delta_tensor(current.effort, future.effort),
            names=current.names,
            timestamps=self._delta_timestamps(
                current.timestamps, future.timestamps
            ),
        )
        return ToDeltaJointsStateReturn(ret)

    def _delta_tensor(
        self, current: torch.Tensor | None, future: torch.Tensor | None
    ) -> torch.Tensor | None:
        non_cnt = (current, future).count(None)

        if non_cnt == 2:
            return None
        elif non_cnt == 1:
            raise ValueError("Both current and future tensors cannot be None.")

        assert current is not None and future is not None, (
            "Both current and future tensors must be provided."
        )

        if current.shape[0] != future.shape[0] and current.shape[0] != 1:
            raise ValueError(
                "Current and future tensors must have the same batch size "
                f"or current must be a single sample. "
                f"Got {current.shape[0]} and {future.shape[0]}."
            )

        return future - current

    def _delta_timestamps(
        self, current: list[int] | None, future: list[int] | None
    ) -> list[int] | None:
        non_cnt = (current, future).count(None)

        if non_cnt == 2:
            return None
        elif non_cnt == 1:
            raise ValueError(
                "Both current and future timestamps cannot be None."
            )

        assert current is not None and future is not None, (
            "Both current and future timestamps must be provided."
        )

        if len(current) != len(future) and len(current) != 1:
            raise ValueError(
                "Current and future timestamps must have the same length "
                "or current must be a single sample."
            )

        if len(current) == 1:
            return [future[i] - current[0] for i in range(len(future))]

        return [f - c for f, c in zip(future, current, strict=True)]


class ToDeltaJointsStateConfig(DictTransformConfig[ToDeltaJointsState]):
    class_type: ClassType[ToDeltaJointsState] = ToDeltaJointsState

    input_columns: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )
    """Mapping of input columns.

    The values of mapping must be in ["current", "future"]."""
