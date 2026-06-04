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

import torch
from google.protobuf.timestamp import from_nanoseconds
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.torch_utils import dtype_str2torch
from robo_orchard_schemas.sensor_msgs.JointState_pb2 import (
    JointState as PbJointState,
    JointStateStamped as PbJointStateStamped,
    MultiJointStateStamped as PbMultiJointStateStamped,
)

from robo_orchard_lab.dataset.datatypes.joint_state import BatchJointsState
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
    TensorTargetConfigMixin,
)
from robo_orchard_lab.utils.protobuf import is_list_of_protobuf_msg_type

___all__ = [
    "ToBatchJointsState",
    "ToBatchJointsStateConfig",
    "FromBatchJointsState",
    "FromBatchJointsStateConfig",
]


def to_numpy(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return tensor.numpy(force=True)


ToBatchJointsState_SRC_TYPE = (
    PbJointStateStamped
    | PbMultiJointStateStamped
    | list[PbJointStateStamped]
    | list[PbMultiJointStateStamped]
)


class ToBatchJointsState(
    MessageConverterStateless[
        ToBatchJointsState_SRC_TYPE,
        BatchJointsState,
    ]
):
    """Convert to BatchJointsState.

    This class accepts either a single `JointStateStamped` or
    `MultiJointStateStamped`, or a list of either type.
    The output is a `BatchJointsState` object containing
    the joint states in a batch format.

    """

    def __init__(
        self,
        cfg: ToBatchJointsStateConfig,
    ):
        self._cfg = cfg
        self._dtype = dtype_str2torch(cfg.dtype)

    def _set_joint_states(
        self,
        data: list[PbJointState],
        row_id: int | None,
        target: BatchJointsState,
    ):
        if row_id is None:
            # batch mode
            for i, joint_state in enumerate(data):
                if target.position is not None:
                    target.position[i, 0] = (
                        joint_state.position
                        if joint_state.HasField("position")
                        else torch.nan
                    )
                if target.velocity is not None:
                    target.velocity[i, 0] = (
                        joint_state.velocity
                        if joint_state.HasField("velocity")
                        else torch.nan
                    )
                if target.effort is not None:
                    target.effort[i, 0] = (
                        joint_state.effort
                        if joint_state.HasField("effort")
                        else torch.nan
                    )
        else:
            for i, joint_state in enumerate(data):
                if target.position is not None:
                    target.position[row_id, i] = (
                        joint_state.position
                        if joint_state.HasField("position")
                        else torch.nan
                    )
                if target.velocity is not None:
                    target.velocity[row_id, i] = (
                        joint_state.velocity
                        if joint_state.HasField("velocity")
                        else torch.nan
                    )
                if target.effort is not None:
                    target.effort[row_id, i] = (
                        joint_state.effort
                        if joint_state.HasField("effort")
                        else torch.nan
                    )

    def convert(self, src: ToBatchJointsState_SRC_TYPE) -> BatchJointsState:
        src = self._format_input(src)
        batch_size = len(src)
        if batch_size == 0:
            raise ValueError("Input data must not be empty.")

        if is_list_of_protobuf_msg_type(src, PbJointStateStamped):
            # Convert list of JointStateStamped
            joint_num = 1
            batch_size = len(src)
            t = torch.zeros(size=(batch_size, joint_num), dtype=self._dtype)
            ret = BatchJointsState(
                position=t.clone()
                if src[0].state.HasField("position")
                else None,
                velocity=t.clone()
                if src[0].state.HasField("velocity")
                else None,
                effort=t.clone() if src[0].state.HasField("effort") else None,
                names=None,
                timestamps=[state.timestamp.ToNanoseconds() for state in src],
            )
            # set p, v, e
            self._set_joint_states(
                [joint_state.state for joint_state in src],
                row_id=None,
                target=ret,
            )
            ret.names = (
                [src[0].state.name] if src[0].state.HasField("name") else None
            )
        elif is_list_of_protobuf_msg_type(src, PbMultiJointStateStamped):
            # Convert list of MultiJointStateStamped
            joint_num = len(src[0].states)
            batch_size = len(src)
            t = torch.zeros(size=(batch_size, joint_num), dtype=self._dtype)
            ret = BatchJointsState(
                position=t.clone()
                if src[0].states[0].HasField("position")
                else None,
                velocity=t.clone()
                if src[0].states[0].HasField("velocity")
                else None,
                effort=t.clone()
                if src[0].states[0].HasField("effort")
                else None,
                names=None,
                timestamps=[state.timestamp.ToNanoseconds() for state in src],
            )
            # set p, v, e
            for i, state in enumerate(src):
                self._set_joint_states(
                    [joint_state for joint_state in state.states],
                    row_id=i,
                    target=ret,
                )
            ret.names = [src[0].states[i].name for i in range(joint_num)]
        else:
            raise TypeError(
                "Input data must be a list of JointStateStamped or "
                "MultiJointStateStamped."
            )
        return ret.to(device=self._cfg.device)

    def _format_input(
        self, data: ToBatchJointsState_SRC_TYPE
    ) -> list[PbMultiJointStateStamped] | list[PbJointStateStamped]:
        """Format the input data to a list of JointStateStamped."""
        if not isinstance(data, list):
            data = [data]  # type: ignore

        return data  # type: ignore


class FromBatchJointsState(
    MessageConverterStateless[BatchJointsState, list[PbMultiJointStateStamped]]
):
    """Convert from BatchJointsState to list of MultiJointStateStamped.

    The output is a list of `MultiJointStateStamped` messages, each
    representing the joint states at a specific timestamp.

    """

    def __init__(
        self,
        cfg: FromBatchJointsStateConfig,
    ):
        self._cfg = cfg

    def convert(self, src: BatchJointsState) -> list[PbMultiJointStateStamped]:
        joint_state = src
        if joint_state.timestamps is None:
            raise ValueError("timestamps is required")
        ret: list[PbMultiJointStateStamped] = []
        batch_size = joint_state.batch_size
        joint_num = joint_state.joint_num
        position = to_numpy(joint_state.position)
        velocity = to_numpy(joint_state.velocity)
        effort = to_numpy(joint_state.effort)
        names = (
            joint_state.names
            if joint_state.names is not None
            else [None] * joint_num
        )
        for i in range(batch_size):
            state_list = [
                PbJointState(
                    frame_id=names[j],
                    name=names[j],
                    position=position[i, j] if position is not None else None,
                    velocity=velocity[i, j] if velocity is not None else None,
                    effort=effort[i, j] if effort is not None else None,
                )
                for j in range(joint_num)
            ]
            ret.append(
                PbMultiJointStateStamped(
                    timestamp=from_nanoseconds(joint_state.timestamps[i]),
                    states=state_list,
                ),
            )
        return ret


class ToBatchJointsStateConfig(
    MessageConverterConfig[ToBatchJointsState],
    TensorTargetConfigMixin[ToBatchJointsState],
):
    class_type: type[ToBatchJointsState] = ToBatchJointsState


class FromBatchJointsStateConfig(MessageConverterConfig[FromBatchJointsState]):
    class_type: ClassType[FromBatchJointsState] = FromBatchJointsState
