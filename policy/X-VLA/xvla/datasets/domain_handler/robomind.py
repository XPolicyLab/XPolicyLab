# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, Iterable
import numpy as np
import h5py

from ..utils import euler_to_rotate6d
from .base import BaseHDF5Handler


class RobomindHandler(BaseHDF5Handler):
    """
    Unified handler for 'robomind-*' datasets.

    Expected HDF5 structure (per variant):
      - robomind-franka / robomind-ur:
          /puppet/end_effector         [T, 6]  (xyz + euler_xyz)
          /puppet/joint_position       [T, ...]  (last elem = gripper 1/0)
      - robomind-agilex:
          /puppet/end_effector_left    [T, >=6] (xyz + euler_xyz + grip_raw)
          /puppet/end_effector_right   [T, >=6]
      - robomind-franka-dual:
          /puppet/end_effector         [T, 12]  (L_xyz L_euler R_xyz R_euler)
          /puppet/joint_position       [T, ...] (index 7 => L grip, last => R grip)
    """

    dataset_name = "robomind-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        ds_name: str = self.meta["dataset_name"]

        if ds_name in ("robomind-franka", "robomind-ur"):
            # Single arm; right is dummy zeros.
            freq, qdur = 30.0, 4.0
            ee = f["puppet"]["end_effector"][()]              # [T,6]
            grip = f["puppet"]["joint_position"][:, -1:]      # [T,1], 1->closed, 0->open
            left = np.concatenate([ee[:, :3], euler_to_rotate6d(ee[:, 3:6], "xyz"), grip], axis=-1)  # [T,10]
            right = np.zeros_like(left)
            return left, right, None, None, freq, qdur

        if ds_name == "robomind-agilex":
            # Dual arms; threshold raw gripper (>2.5 => closed).
            freq, qdur = 30.0, 4.0
            le = f["puppet"]["end_effector_left"][()]         # [..., >=6]
            re = f["puppet"]["end_effector_right"][()]
            l = np.concatenate([le[:, :3], euler_to_rotate6d(le[:, 3:6], "xyz"), (le[:, -1:] > 2.5)], axis=-1)
            r = np.concatenate([re[:, :3], euler_to_rotate6d(re[:, 3:6], "xyz"), (re[:, -1:] > 2.5)], axis=-1)
            return l, r, None, None, freq, qdur

        if ds_name == "robomind-franka-dual":
            # Packed dual ee; grippers from joint_position indices.
            freq, qdur = 30.0, 4.0
            ee = f["puppet"]["end_effector"][()]              # [T,12] L(xyz,euler) + R(xyz,euler)
            jp = f["puppet"]["joint_position"][()]            # [T, ...]
            l = np.concatenate([ee[:, 0:3],  euler_to_rotate6d(ee[:, 3:6],  "xyz"), jp[:, 7:8]],  axis=-1)
            r = np.concatenate([ee[:, 6:9],  euler_to_rotate6d(ee[:, 9:12], "xyz"), jp[:, -1:]],  axis=-1)
            return l, r, None, None, freq, qdur

        raise NotImplementedError(f"RobomindHandler: unsupported dataset '{ds_name}'")

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        # Leave margin for the future window; 30 frames at 30Hz ≈ 1s.
        return range(0, max(0, T_left - 30))
