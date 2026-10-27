# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed MeshContext dataclass, validation, and strategy map."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from awomo.awomo.training.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)

if TYPE_CHECKING:
    pass

    pass
    pass


STRATEGY_MAP: Dict[str, type] = {
    "fsdp2": FSDP2Config,
    "megatron_fsdp": MegatronFSDPConfig,
    "ddp": DDPConfig,
}


class MeshAxisName(str, Enum):
    """Canonical mesh-dimension names used by ``DeviceMesh`` and helpers."""

    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP_CP = "dp_cp"
    CP = "cp"
    TP = "tp"
    EP = "ep"
    EP_SHARD = "ep_shard"


_VALID_AXIS_NAMES: frozenset = frozenset(MeshAxisName)


@dataclass
class MeshContext:
    """Runtime distributed training context: configs + device meshes."""

    strategy_config: Optional[Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]] = None
    pipeline_config: Optional["PipelineConfig"] = None
    moe_config: Optional["MoEParallelizerConfig"] = None
    activation_checkpointing: bool = False

    device_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)
    moe_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_mesh_dim_names(self)
        _validate_distributed_setup(self)

    @property
    def pp_size(self) -> int:
        return _get_axis_size(self.device_mesh, MeshAxisName.PP)

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    def tp_size(self) -> int:
        return _get_axis_size(self.device_mesh, MeshAxisName.TP)

    @property
    def cp_size(self) -> int:
        return _get_axis_size(self.device_mesh, MeshAxisName.CP)

    @property
    def ep_size(self) -> int:
        return _get_axis_size(self.moe_mesh, MeshAxisName.EP)

    @property
    def dp_size(self) -> Optional[int]:
        return _get_axis_size(self.device_mesh, MeshAxisName.DP, default=None)

    @property
    def dp_replicate_size(self) -> Optional[int]:
        return _get_axis_size(self.device_mesh, MeshAxisName.DP_REPLICATE, default=None)

    @property
    def dp_shard_size(self) -> int:
        return _get_axis_size(self.device_mesh, MeshAxisName.DP_SHARD, default=1)

    def _dp_axis_names(self) -> Tuple[str, ...]:
        if self.device_mesh is not None:
            names = self.device_mesh.mesh_dim_names
            if MeshAxisName.DP_REPLICATE in names and MeshAxisName.DP_SHARD_CP in names:
                return (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD_CP)
        return (MeshAxisName.DP_SHARD_CP,)

    def pipeline_axis_kwargs(self) -> Dict[str, object]:
        return {
            "pp_axis_name": MeshAxisName.PP,
        } | self.parallelize_axis_kwargs()

    def parallelize_axis_kwargs(self) -> Dict[str, object]:
        return {
            "dp_axis_names": self._dp_axis_names(),
            "cp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.CP),
            "tp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.TP),
            "ep_axis_name": _optional_axis(self.moe_mesh, MeshAxisName.EP),
            "ep_shard_axis_names": (MeshAxisName.EP_SHARD,)
            if _optional_axis(self.moe_mesh, MeshAxisName.EP_SHARD)
            else None,
        }

    @classmethod
    def from_meshes(
        cls,
        device_mesh: Optional["DeviceMesh"],
        moe_mesh: Optional["DeviceMesh"] = None,
        *,
        strategy_config: Optional[Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]] = None,
        pipeline_config: Optional["PipelineConfig"] = None,
        moe_config: Optional["MoEParallelizerConfig"] = None,
        activation_checkpointing: bool = False,
    ) -> "MeshContext":
        return cls(
            strategy_config=strategy_config,
            pipeline_config=pipeline_config,
            moe_config=moe_config,
            activation_checkpointing=activation_checkpointing,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
        )


def _get_axis_size(mesh: Optional["DeviceMesh"], axis: MeshAxisName, default=1) -> Optional[int]:
    if mesh is None:
        return default
    if axis in mesh.mesh_dim_names:
        return mesh[axis].size()
    if hasattr(mesh, "_get_root_mesh"):
        root = mesh._get_root_mesh()
    else:
        root = mesh
    if hasattr(root, "_flatten_mapping") and axis in root._flatten_mapping:
        return root._flatten_mapping[axis].size()
    return default


def _optional_axis(mesh: Optional["DeviceMesh"], axis: MeshAxisName) -> Optional[str]:
    if mesh is not None and axis in mesh.mesh_dim_names:
        return axis
    return None


def _validate_mesh_dim_names(mesh_context: "MeshContext") -> None:
    for label in ("device_mesh", "moe_mesh"):
        mesh = getattr(mesh_context, label)
        if mesh is None:
            continue
        bad = {n for n in mesh.mesh_dim_names if n not in _VALID_AXIS_NAMES}
        if bad:
            raise ValueError(
                f"{label} contains unknown dimension names {bad}; allowed names are {sorted(_VALID_AXIS_NAMES)}"
            )


def _validate_distributed_setup(mesh_context: "MeshContext") -> None:
    if mesh_context.strategy_config is None:
        return

    if isinstance(mesh_context.strategy_config, MegatronFSDPConfig):
        if mesh_context.pp_size > 1:
            raise ValueError("megatron_fsdp does not support pipeline parallelism")
        if mesh_context.ep_size > 1:
            raise ValueError("megatron_fsdp does not support expert parallelism")
        if mesh_context.strategy_config.sequence_parallel:
            raise ValueError("megatron_fsdp does not yet support sequence_parallel")

    if isinstance(mesh_context.strategy_config, DDPConfig):
        if mesh_context.tp_size > 1:
            raise ValueError("ddp does not support tensor parallelism")
        if mesh_context.pp_size > 1:
            raise ValueError("ddp does not support pipeline parallelism")
        if mesh_context.cp_size > 1:
            raise ValueError("ddp does not support context parallelism")
        if mesh_context.ep_size > 1:
            raise ValueError("ddp does not support expert parallelism")
        if mesh_context.dp_replicate_size is not None and mesh_context.dp_replicate_size > 1:
            raise ValueError("ddp does not support HSDP (dp_replicate_size)")

    if mesh_context.pipeline_config is not None and mesh_context.pp_size <= 1:
        raise ValueError("pipeline config requires pp_size > 1")

    if mesh_context.moe_config is not None and mesh_context.ep_size <= 1:
        raise ValueError("moe config requires ep_size > 1")
