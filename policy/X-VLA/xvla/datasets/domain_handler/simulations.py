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

from typing import Optional, Tuple, Iterable, Sequence, Any
import numpy as np
import h5py

from ..utils import euler_to_rotate6d, quat_to_rotate6d
from .base import BaseHDF5Handler


# ------------------------------- Calvin --------------------------------------
class CalvinHandler(BaseHDF5Handler):
    """Calvin (sim): proprio [T,7] -> xyz(3)+euler_xyz(3)+grip(1). Right is zeros."""
    dataset_name = "Calvin"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]  # [T,7]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:] < 0.],
            axis=-1,
        )  # [T,10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 20))


# --------------------------------- RT1 ---------------------------------------
class RT1Handler(BaseHDF5Handler):
    """RT1 (sim-like packaging): eef_quat_orientation [T,7], gripper [T,1]."""
    dataset_name = "RT1"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 3.0, 10.0
        eefq = f["eef_quat_orientation"][()]  # [T,7] pos3 + quat4
        grip = f["gripper"][()]               # [T,1] or [T]
        if grip.ndim == 1:
            grip = grip[:, None]
        left = np.concatenate([eefq[:, :3], quat_to_rotate6d(eefq[:, 3:]), grip], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 6))


# ------------------------------- Bridge --------------------------------------
class BridgeHandler(BaseHDF5Handler):
    """
    Bridge (sim). HDF5:
      /proprio [T, >=6] -> xyz(3) + euler_xyz(3) + ...
      /action  [T, ...] -> last channel is gripper (1=open), we convert to (1=closed)
    Output left/right: [T,10] = xyz(3)+rot6d(6)+grip(1). Single arm → right zeros.
    """
    dataset_name = "Bridge"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0
        proprio = f["proprio"][()]                     # [T, >=6]
        action  = f["action"][()]                      # [T, ...]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), 1 - action[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------- LIBERO --------------------------------------
class LiberoHandler(BaseHDF5Handler):
    """
    LIBERO (sim). HDF5:
      /abs_action_6d [T,10] = xyz(3)+rot6d(6)+grip_raw(1). Single arm.
    Also drops first frame for images (matches original pipeline behavior).
    """
    dataset_name = "libero"

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys = self.meta["observation_key"]
        images = [f[k] for k in keys]
        # Drop the first frame (image desync quirk in original data)
        return [img[1:] for img in images]

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        a = f["abs_action_6d"][()]                             # [T,10]
        left = np.concatenate([a[:, :9], (a[:, 9:] > 0.0)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------ VLABench -------------------------------------
class VLABenchHandler(BaseHDF5Handler):
    """
    VLABench (sim). HDF5:
      /proprio [T, >=7] -> xyz(3) + euler_xyz(3) + grip(1).
    Single arm → right zeros.
    """
    dataset_name = "VLABench"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))


# ------------------------------ RobotWin2 ------------------------------------
class RobotWin2Handler(BaseHDF5Handler):
    """
    robotwin2_abs_ee / robotwin2_clean (sim). HDF5:
      /endpose/left_endpose   [T,7]  xyz(3)+quat(4)
      /endpose/right_endpose  [T,7]
      /endpose/left_gripper   [T]    1=open  -> convert to 1=closed
      /endpose/right_gripper  [T]
    Output both arms. freq≈30Hz, qdur=1s.
    """
    dataset_name = "robotwin2-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        l = f["endpose/left_endpose"][()]                      # [T,7]
        r = f["endpose/right_endpose"][()]                     # [T,7]
        lg = (1 - f["endpose/left_gripper"][()][:, None])      # [T,1] 1=closed
        rg = (1 - f["endpose/right_gripper"][()][:, None])
        left  = np.concatenate([l[:, :3], quat_to_rotate6d(l[:, 3:]), lg], axis=-1)
        right = np.concatenate([r[:, :3], quat_to_rotate6d(r[:, 3:]), rg], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ---------------------------- Robocasa-Human ---------------------------------
class RobocasaHumanHandler(BaseHDF5Handler):
    """
    robocasa-human (teleop in sim). HDF5:
      /action_dict/abs_pos     [T,3]
      /action_dict/abs_rot_6d  [T,6]
      /action_dict/gripper     [T,1]  ( >0 => closed )
    Single arm → right zeros.
    """
    dataset_name = "robocasa-human"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [
                f["action_dict/abs_pos"][()],
                f["action_dict/abs_rot_6d"][()],
                (f["action_dict/gripper"][()] > 0.0).astype(np.float32),
            ],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
