# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
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

"""HDF5-backed RoboDojo dataset loader for Hy-VLA training.

The dataset reads RoboDojo-format episode HDF5 files organised as::

    {hdf5_dir}/{task_name}/{robot_type}/data/episode_XXXXXXX.hdf5

Each episode HDF5 contains:

- ``state/left_ee_poses``, ``state/right_ee_poses``: (T, 7) xyz+quat_wxyz
- ``state/left_ee_joint_states``, ``state/right_ee_joint_states``: (T, 1) gripper
- ``action/left_arm_joint_states``, ``action/right_arm_joint_states``: (T, 6)
- ``action/left_ee_joint_states``, ``action/right_ee_joint_states``: (T, 1)
- ``vision/{cam_head,cam_left_wrist,cam_right_wrist}/colors``: (T,) vlen bytes (JPEG)
- ``instruction``: bytes (single instruction per episode)
- ``additional_info/frequency``: int (e.g. 25 Hz)
"""

import os
import glob
import json
import pickle
import random
import h5py
import cv2
import numpy as np
from hy_vla.utils.transform_utils import (
    convert_PosQuat2PosRotationMatrix_batch,
    dual_arm_poses_to_relative,
    convert_frame_robo_to_umi,
)
from scipy.spatial.transform import Rotation as R


def _scan_episodes(hdf5_dir: str) -> list[dict]:
    """Walk the RoboDojo directory tree and collect all episode HDF5 paths.

    Directory layout::

        hdf5_dir/
          task_A/
            robot_X/
              data/
                episode_0000000.hdf5
                episode_0000001.hdf5
                ...

    Returns a list of episode descriptors::

        {
            "task_name": str,
            "robot_type": str,
            "hdf5_path": str (absolute),
            "episode_id": int,
        }
    """
    episodes: list[dict] = []
    for task_dir in sorted(os.listdir(hdf5_dir)):
        task_path = os.path.join(hdf5_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        for robot_dir in sorted(os.listdir(task_path)):
            robot_path = os.path.join(task_path, robot_dir)
            if not os.path.isdir(robot_path):
                continue
            data_dir = os.path.join(robot_path, "data")
            if not os.path.isdir(data_dir):
                continue
            pattern = os.path.join(data_dir, "episode_*.hdf5")
            for h5_path in sorted(glob.glob(pattern)):
                basename = os.path.basename(h5_path)
                ep_id_str = basename.replace("episode_", "").replace(".hdf5", "")
                try:
                    ep_id = int(ep_id_str)
                except ValueError:
                    continue
                episodes.append({
                    "task_name": task_dir,
                    "robot_type": robot_dir,
                    "hdf5_path": h5_path,
                    "episode_id": ep_id,
                })
    return episodes


def pad_vector(vector, new_dim):
    """Can be (sequence_length x features_dimension)."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype)
    new_vector[..., :current_dim] = vector
    return new_vector


def get_history_indices(step_id, history_size, interval, random_sample=True):
    """Compute history frame indices for the K-frame image stack.

    Returns K indices (history + current). Slot K-1 is forced to
    ``step_id``; for slot k, the target end index is
    ``end = step_id - (K - 1 - k) * interval``.

      * ``random_sample=True``  (train): uniformly sample in
        ``[end - interval + 1, end]``, clipped to ``>= 0``.
      * ``random_sample=False`` (eval): take ``max(end, 0)``.

    Out-of-range indices collapse to 0 (the downstream code treats the
    duplicated frame 0 as padding).

    Note: ``history_size`` here is the TOTAL count (history + current),
    matching how the dataset pipeline consumes ``img_history_size``
    end-to-end.
    """
    assert history_size >= 1
    assert interval >= 1
    indices = []
    for k in range(history_size):
        end = step_id - (history_size - 1 - k) * interval
        if random_sample:
            start = max(end - interval + 1, 0)
            end_clamped = max(end, 0)
            if end_clamped < start:
                idx = start
            else:
                idx = random.randint(start, end_clamped)
        else:
            idx = max(end, 0)
        indices.append(idx)
    indices[-1] = step_id
    return indices


def _decode_jpeg(raw):
    """Decode a JPEG byte string into an RGB uint8 image.

    Already-decoded numpy arrays are returned as-is.
    """
    if raw is None:
        return None
    if isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.shape[-1] == 3 \
            and raw.dtype == np.uint8:
        return raw
    if hasattr(raw, "tobytes"):
        raw = raw.tobytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # NOTE: RoboDojo HDF5 JPEGs decode to RGB directly (verified by visual
    # inspection, LEFT=BGR came out natural, meaning no cvtColor needed).
    return img


class RoboDojoVLADataset:
    """Dataset wrapper for the RoboDojo HDF5 format.

    Provides the same ``get_item`` interface as
    :class:`hy_vla.data.robotwin_dataset.RoboTwinVLADataset` so it can be plugged
    into :class:`hy_vla.data.vla_dataset.VLADataset` without changes.
    """

    def __init__(self, config) -> None:
        HDF5_DIR = config.dataset.hdf5_dir
        self.HDF5_DIR = HDF5_DIR

        print(f"[robodojo_dataset] scanning {HDF5_DIR} ...")
        all_episodes = _scan_episodes(HDF5_DIR)
        if not all_episodes:
            raise ValueError(
                f"No episode HDF5 files found under {HDF5_DIR}. "
                f"Expected layout: {{hdf5_dir}}/{{task}}/{{robot}}/data/episode_*.hdf5"
            )

        # Optional task filter.
        task_filter = getattr(config.dataset, "task_filter", None)
        if task_filter is not None:
            if isinstance(task_filter, str):
                task_filter = [task_filter]
            task_filter = set(task_filter)
            all_episodes = [ep for ep in all_episodes if ep["task_name"] in task_filter]
            print(f"[robodojo_dataset] task_filter: {sorted(task_filter)} -> "
                  f"{len(all_episodes)} episodes kept")

        # Optional episode limit per task.
        max_episodes_per_task = getattr(config.dataset, "max_episodes_per_task", None)
        if max_episodes_per_task is not None:
            task_counts: dict[str, int] = {}
            filtered = []
            for ep in all_episodes:
                tn = ep["task_name"]
                cnt = task_counts.get(tn, 0)
                if cnt < max_episodes_per_task:
                    filtered.append(ep)
                    task_counts[tn] = cnt + 1
            all_episodes = filtered
            print(f"[robodojo_dataset] max_episodes_per_task: {max_episodes_per_task} -> "
                  f"{len(all_episodes)} episodes kept")

        # Group by task to keep enumeration order stable.
        eps_by_task: dict[str, list] = {}
        for ep in all_episodes:
            eps_by_task.setdefault(ep["task_name"], []).append(ep)

        # Flat episode pool: task-grouped, sorted by task name, internal order preserved.
        self.episodes = []
        for task_name in sorted(eps_by_task.keys()):
            self.episodes.extend(eps_by_task[task_name])

        print(f"[robodojo_dataset] num_tasks: {len(eps_by_task)}, "
              f"num_episodes: {len(self.episodes)}")

        # Video encoder.
        self.use_video_encoder = bool(getattr(config.dataset, "use_video_encoder", False))
        print(f"[robodojo_dataset] use_video_encoder: {self.use_video_encoder}")

        # Action type.
        self.action_type = config.dataset.act_type
        print(f"[robodojo_dataset] action_type: {self.action_type}")

        # Optional: convert EEF poses to UMI coordinate frame.
        self.umi_coord_frame = bool(getattr(config.dataset, "umi_coord_frame", False))
        if self.umi_coord_frame:
            print(
                "[robodojo_dataset] umi_coord_frame=True: "
                "applying world rotation W + local column permutation P "
                "(W @ R_rd @ P, pos → W @ pos) "
                "and converting quaternion wxyz→xyzw. "
                "WARNING: normalization stats must be regenerated "
                "with the same coord frame!"
            )
        self.umi_gripper_space = bool(getattr(config.dataset, "umi_gripper_space", False))
        if self.umi_coord_frame and self.umi_gripper_space:
            print(
                "[robodojo_dataset] umi_gripper_space=True: "
                "gripper values will also be mapped RoboTwin [1,0] → UMI [0,90]."
            )

        self.downsample_rate = config.dataset.downsample_rate
        print(f"[robodojo_dataset] downsample_rate: {self.downsample_rate}")

        # Norm-stats pickle.
        def _load_mean_std(path, chunk_slice=None, with_abs=False):
            if not os.path.exists(path):
                raise ValueError(f"File does not exist: {path}")
            with open(path, "rb") as fp:
                info = pickle.load(fp)
            qm = np.array(info["qpos_mean"], dtype=np.float32)
            qs = np.array(info["qpos_std"], dtype=np.float32)
            am = np.array(info["action_mean"], dtype=np.float32)
            as_ = np.array(info["action_std"], dtype=np.float32)
            am_absolute = None
            as_absolute = None
            if with_abs:
                if "action_mean_abs" not in info or "action_std_abs" not in info:
                    raise KeyError(
                        f"act_type contains 'with_absolute' but the norm pkl at "
                        f"{path} does not carry 'action_mean_abs' / "
                        f"'action_std_abs'. Re-generate it with "
                        f"utils/normalize_robodojo.py."
                    )
                am_absolute = np.array(info["action_mean_abs"], dtype=np.float32)
                as_absolute = np.array(info["action_std_abs"], dtype=np.float32)
                if am_absolute.shape != am.shape:
                    raise ValueError(
                        f"action_mean_abs shape {am_absolute.shape} must match "
                        f"action_mean shape {am.shape} in {path}"
                    )
            if chunk_slice is not None:
                n = int(chunk_slice)
                assert 0 < n <= am.shape[0], (
                    f"chunk_slice={n} out of range [1, {am.shape[0]}] for {path}"
                )
                print(f"[robodojo_dataset] mean_std slice: {am.shape[0]} -> {n} ({path})")
                am = am[:n]
                as_ = as_[:n]
                if am_absolute is not None:
                    am_absolute = am_absolute[:n]
                    as_absolute = as_absolute[:n]
            return qm, qs, am, as_, am_absolute, as_absolute

        if not hasattr(config.dataset, "mean_std_path"):
            raise ValueError(
                "dataset.mean_std_path is required: the dataset always "
                "normalizes states/actions with the loaded pkl."
            )
        mean_std_path = config.dataset.mean_std_path
        print(f"[robodojo_dataset] mean_std_path: {mean_std_path}")
        with_abs = "with_absolute" in self.action_type
        (
            self.qpos_mean,
            self.qpos_std,
            self.act_mean,
            self.act_std,
            _act_mean_abs,
            _act_std_abs,
        ) = _load_mean_std(
            mean_std_path,
            chunk_slice=getattr(config.dataset, "mean_std_chunk_slice", None),
            with_abs=with_abs,
        )

        # For ``_with_absolute``: cat along axis 0 so ``self.act_mean``
        # becomes (2*chunk, 20), aligning row-wise with the doubled-time
        # actions tensor produced by ``parse_hdf5_file``
        # (rows [0..chunk-1] = RT_relative, rows [chunk..2*chunk-1] =
        # absolute PosRotMat over the SAME chunk frames).
        if with_abs:
            self.act_mean = np.concatenate([self.act_mean, _act_mean_abs], axis=0)
            self.act_std = np.concatenate([self.act_std, _act_std_abs], axis=0)
            print(
                f"[robodojo_dataset] with_absolute: act_mean/std cat along time axis -> "
                f"{self.act_mean.shape}"
            )

        self.CHUNK_SIZE = config["dataset"]["action_chunk_size"]

        # Image history.
        raw_img_history_size = int(config["dataset"]["img_history_size"])
        if self.use_video_encoder:
            self.IMG_HISORY_SIZE = raw_img_history_size
        else:
            if raw_img_history_size != 1:
                print(
                    f"[WARN] use_video_encoder=False but img_history_size="
                    f"{raw_img_history_size}; forcing img_history_size=1 "
                    f"for the single-frame pathway."
                )
            self.IMG_HISORY_SIZE = 1

        self.IMG_HISTORY_INTERVAL = int(
            config["dataset"].get("img_history_interval", 1)
        )
        assert self.IMG_HISTORY_INTERVAL >= 1, "img_history_interval must be >= 1"
        self.IMG_HISTORY_RANDOM_SAMPLE = bool(
            config["dataset"].get("img_history_random_sample", True)
        )
        print(
            f"[robodojo_dataset] img_history_size: {self.IMG_HISORY_SIZE}, "
            f"img_history_interval: {self.IMG_HISTORY_INTERVAL}, "
            f"img_history_random_sample: {self.IMG_HISTORY_RANDOM_SAMPLE}"
        )
        self.STATE_DIM = config["dataset"]["state_dim"]

        # Deterministic mode.
        self.deterministic = bool(getattr(config.dataset, "deterministic", False))
        self.deterministic_index = None
        if self.deterministic:
            pairs: list[tuple[int, int]] = []
            for ep_idx, ep in enumerate(self.episodes):
                with h5py.File(ep["hdf5_path"], "r") as f:
                    # Use state/left_ee_poses to determine num_frames.
                    n_raw = f["state"]["left_ee_poses"].shape[0]
                for t in range(n_raw):
                    pairs.append((ep_idx, t))
            self.deterministic_index = pairs
            print(
                f"[robodojo_dataset] deterministic=True: enumerated "
                f"{len(self.deterministic_index)} (episode, raw_step) pairs "
                f"across {len(self.episodes)} episodes"
            )
            assert len(self.deterministic_index) > 0, (
                "deterministic mode produced an empty index"
            )

    def __len__(self):
        if self.deterministic and self.deterministic_index is not None:
            return len(self.deterministic_index)
        return len(self.episodes)

    def get_item(self, index: int = None):
        """Get a training sample.

        Args:
            index (int, optional): the dataset-side index.

        Returns:
            sample (dict): a dictionary containing the training sample.
        """
        if self.deterministic:
            assert index is not None, (
                "deterministic=True requires an explicit dataset index"
            )
            N = len(self.deterministic_index)
            i = int(index) % N
            attempts = 0
            while True:
                ep_idx, raw_step = self.deterministic_index[i]
                ep = self.episodes[ep_idx]
                valid, sample = self.parse_hdf5_file(
                    ep,
                    forced_step_id=int(raw_step),
                )
                if valid:
                    return sample
                i = (i + 1) % N
                attempts += 1
                assert attempts < N, (
                    "deterministic mode: every (episode, step) pair was rejected"
                )

        # Random path.
        while True:
            if index is None:
                ep = self.episodes[np.random.randint(0, len(self.episodes))]
            else:
                ep = self.episodes[index % len(self.episodes)]
            valid, sample = self.parse_hdf5_file(ep)
            if valid:
                return sample

    def parse_hdf5_file(self, ep, forced_step_id=None):
        """Parse a RoboDojo HDF5 file to generate a training sample.

        Args:
            ep (dict): an episode descriptor with keys
                ``hdf5_path``, ``task_name``, ``robot_type``, ``episode_id``.
            forced_step_id (int, optional): when not None, use this raw
                frame index as the current step.

        Returns:
            tuple: (valid, sample_dict)
        """
        file_path = ep["hdf5_path"]
        with h5py.File(file_path, "r") as f:
            # --- State: concatenate left/right EE poses + grippers ---
            state_group = f["state"]
            left_ee = state_group["left_ee_poses"][:]    # (T, 7) xyz+qxyzw
            right_ee = state_group["right_ee_poses"][:]  # (T, 7)
            left_gripper = state_group["left_ee_joint_states"][:]   # (T, 1)
            right_gripper = state_group["right_ee_joint_states"][:]  # (T, 1)

            # HDF5 stores quaternions in wxyz order [qw,qx,qy,qz].
            # Keep as-is — the downstream pipeline expects wxyz format.
            left_ee_converted = left_ee.copy()     # already [x,y,z, qw,qx,qy,qz]
            right_ee_converted = right_ee.copy()

            # Build full state: (T, 16) = [left_xyz(3), left_quat(4), left_grip(1),
            #                               right_xyz(3), right_quat(4), right_grip(1)]
            qpos = np.concatenate([
                left_ee_converted, left_gripper,
                right_ee_converted, right_gripper,
            ], axis=1)  # (T, 16)

            # HDF5 stores quaternions in wxyz; convert to xyzw (scipy convention).
            qpos[:, [3, 4, 5, 6]] = qpos[:, [4, 5, 6, 3]]     # left  wxyz→xyzw
            qpos[:, [11, 12, 13, 14]] = qpos[:, [12, 13, 14, 11]]  # right wxyz→xyzw

            # Optional: convert to UMI coordinate frame.
            if self.umi_coord_frame:
                qpos = convert_frame_robo_to_umi(qpos, convert_gripper=self.umi_gripper_space)

            num_steps = qpos.shape[0]

            # --- Step selection ---
            first_idx = 0
            if forced_step_id is not None:
                t = int(forced_step_id)
                if t < first_idx or t >= num_steps:
                    return False, None
                step_id = t
            else:
                step_id = np.random.randint(first_idx, num_steps)
            c_id = step_id
            M = num_steps

            # --- Instruction ---
            raw_instr = f["instruction"][()]
            if isinstance(raw_instr, bytes):
                instruction = raw_instr.decode("utf-8")
            else:
                instruction = str(raw_instr)

            meta = {"#steps": num_steps, "step_id": step_id, "instruction": instruction}

            # --- State: current frame ---
            state = qpos[c_id : c_id + 1]  # (1, 16) PosQuat, already xyzw (scipy convention)
            state = convert_PosQuat2PosRotationMatrix_batch(state, quat_order="xyzw")  # (1, 20)

            sample_ds = self.downsample_rate

            # --- Actions: UMI-style EE pose chunk → RT-relative ---
            # Build chunk from the downsampled qpos timeline: slot k
            # takes the EE pose at index c_id + k*sample_ds (clamped to M-1).
            # Then convert to wrist-frame-relative deltas (dual_arm_poses_to_relative).
            chunk_offsets = np.arange(self.CHUNK_SIZE, dtype=np.int64) * sample_ds
            chunk_compressed = np.minimum(c_id + chunk_offsets, M - 1)
            action_chunk_16d = qpos[chunk_compressed].copy()  # (CHUNK_SIZE, 16)

            # For ``_with_absolute``: also compute the absolute PosRotMat
            # target on the SAME chunk frames and cat along axis 0, so
            # the final actions tensor is (2*chunk, 20):
            # rows [0..chunk-1] = RT_relative, rows [chunk..2*chunk-1] =
            # absolute PosRotMat. ``self.act_mean / self.act_std`` was
            # already cat'd to (2*chunk, 20) in __init__.
            if "with_absolute" in self.action_type:
                actions_abs = convert_PosQuat2PosRotationMatrix_batch(
                    action_chunk_16d, quat_order="xyzw"
                )
            else:
                actions_abs = None

            actions = dual_arm_poses_to_relative(action_chunk_16d)  # (CHUNK_SIZE, 20)

            if actions_abs is not None:
                # Cat BEFORE normalize so act_mean/std aligns row-wise.
                actions = np.concatenate([actions, actions_abs], axis=0)

            # --- Normalize ---
            state = (state - self.qpos_mean) / (self.qpos_std + 1e-8)
            actions = (actions - self.act_mean) / (self.act_std + 1e-8)

            state = pad_vector(state, self.STATE_DIM)
            state_indicator = pad_vector(np.ones(qpos.shape[-1]), self.STATE_DIM)
            actions = pad_vector(actions, self.STATE_DIM)

            # --- Vision ---
            vision_group = f["vision"]

            def parse_img(cam_key):
                """Read JPEG-encoded frames from ``vision/{cam_key}/colors``."""
                if cam_key not in vision_group:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                cam = vision_group[cam_key]
                if "colors" not in cam:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                colors_ds = cam["colors"]
                raw_interval = self.IMG_HISTORY_INTERVAL * sample_ds
                planned_indices = np.asarray(
                    get_history_indices(
                        c_id,
                        self.IMG_HISORY_SIZE,
                        raw_interval,
                        random_sample=self.IMG_HISTORY_RANDOM_SAMPLE,
                    ),
                    dtype=int,
                )
                planned_indices = np.clip(planned_indices, 0, len(colors_ds) - 1)

                unique_sorted = np.unique(planned_indices)
                # Read unique frames once.
                raw_batch = [colors_ds[int(idx)] for idx in unique_sorted]
                index_to_pos = {int(idx): pos for pos, idx in enumerate(unique_sorted)}

                imgs = []
                for idx in planned_indices:
                    raw = raw_batch[index_to_pos[int(idx)]]
                    img = _decode_jpeg(raw)
                    if img is not None:
                        imgs.append(img)

                if len(imgs) == 0:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)

                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Left-pad with the first image.
                    imgs = np.concatenate(
                        [np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs],
                        axis=0,
                    )
                return imgs

            cam_high = parse_img("cam_head")
            cam_left_wrist = parse_img("cam_left_wrist")
            cam_right_wrist = parse_img("cam_right_wrist")

            # Mask: slot k is valid iff ``c_id - (K-1-k) * S >= first_idx``.
            K = self.IMG_HISORY_SIZE
            S = self.IMG_HISTORY_INTERVAL * sample_ds
            cam_mask = np.array(
                [(c_id - (K - 1 - k) * S) >= first_idx for k in range(K)],
                dtype=bool,
            )

            sample: dict[str, np.ndarray] = {
                "meta": meta,
                "state": state,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_mask.copy(),
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_mask.copy(),
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_mask.copy(),
            }
            return True, sample
