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


class DroidHandler(BaseHDF5Handler):
    """
    Unified handler for 'Droid-*' datasets.

    Expected HDF5 layout:
      /observation/cartesian_position   [T, 6]   xyz(3) + euler_xyz(3)
      /observation/gripper_position     [T, 1]   1=closed, 0=open (or continuous)

    Output left/right format: [T,10] = xyz(3) + rot6d(6) + grip(1).
    Right arm is dummy zeros (single-arm dataset).
    """

    dataset_name = "Droid-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        # Data rate ~15Hz, use 4s future window to match your original code.
        freq, qdur = 15.0, 4.0

        cart = f["observation"]["cartesian_position"][()]   # [T,6] (xyz + euler_xyz)
        grip = f["observation"]["gripper_position"][()]     # [T,1] or [T]

        if grip.ndim == 1:
            grip = grip[:, None]

        left = np.concatenate(
            [cart[:, :3], euler_to_rotate6d(cart[:, 3:6], "xyz"), grip.astype(np.float32)],
            axis=-1,
        )  # [T,10]
        right = np.zeros_like(left)

        # Use uniform time by freq (return None to let base class construct it).
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        # Leave margin for future window (~30 frames at 15Hz ≈ 2s); match your prior -30.
        return range(0, max(0, T_left - 30))
