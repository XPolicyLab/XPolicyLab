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

import copy

import cv2
import numpy as np
import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
from scipy.spatial.transform import Rotation

__all__ = [
    "AddItems",
    "ConvertDataType",
    "IdentityTransform",
    "ItemSelection",
    "SimpleStateSampling",
    "Resize",
    "ToTensor",
    "MultiArmKinematics",
    "GetProjectionMat",
    "UnsqueezeBatch",
    "JointStateNoise",
    "ExtrinsicNoise",
    "RandomCropPaddingResize",
]


class MoveEgoToCam:
    def __init__(self, cam_idx=-1):
        self.cam_idx = cam_idx

    def __call__(self, data):
        if isinstance(self.cam_idx, str):
            cam_idx = data["cam_names"].index(self.cam_idx)
        else:
            cam_idx = self.cam_idx
        data["T_base2ego"] = data["T_world2cam"][cam_idx] @ data.get(
            "T_base2world", np.eye(4)
        )
        return data


class IdentityTransform:
    def __call__(self, data):
        return data


class AddItems:
    def __init__(self, to_numpy=True, **kwargs):
        self.items = copy.deepcopy(kwargs)
        for k, v in self.items.items():
            if to_numpy and not isinstance(v, np.ndarray):
                self.items[k] = np.array(v)

    def __call__(self, data):
        for k, v in self.items.items():
            data[k] = copy.deepcopy(v)
        return data


class JointStateNoise:
    def __init__(self, noise_range, add_to_pred=False):
        self.range = np.array(noise_range)
        self.add_to_pred = add_to_pred

    def __call__(self, data):
        assert "hist_robot_state" not in data
        num_steps, num_joints = data["hist_joint_state"].shape
        if self.add_to_pred:
            num_steps = 1
        noise = np.random.uniform(
            self.range[..., 0],
            self.range[..., 1],
            size=[num_steps, num_joints],
        )
        noise = torch.from_numpy(noise).to(data["hist_joint_state"])
        data["hist_joint_state"] = data["hist_joint_state"] + noise
        if self.add_to_pred:
            data["pred_joint_state"] = data["pred_joint_state"] + noise
        return data


class SimpleStateSampling:
    def __init__(
        self,
        hist_steps,
        pred_steps,
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=None,
        limitation=3.14,
        static_threshold=1e-3,
        only_hist=False,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.use_master_gripper = use_master_gripper
        self.use_master_joint = use_master_joint
        if use_master_joint ^ use_master_gripper:
            assert gripper_indices is not None
        self.gripper_indices = gripper_indices
        self.limitation = limitation
        self.static_threshold = static_threshold
        self.only_hist = only_hist

    def __call__(self, data):
        if "joint_state" not in data and "hist_joint_state" in data:
            data["hist_joint_state"] = np.clip(
                data["hist_joint_state"], -self.limitation, self.limitation
            )
            return data

        joint_state = copy.deepcopy(data["joint_state"])  # N x num_joint
        mask = np.all(
            (joint_state > -self.limitation) & (joint_state < self.limitation),
            axis=-1,
        )
        joint_state = np.clip(joint_state, -self.limitation, self.limitation)

        if "step_index_in_shard" in data:
            step_index = data["step_index_in_shard"]
        elif "step_index_in_chunk" in data:
            step_index = data["step_index_in_chunk"]
        else:
            step_index = data["step_index"]
        hist_steps = self.hist_steps
        pred_steps = self.pred_steps

        if "ee_state" in data:
            ee_state = data["ee_state"]  # N x [num_gripper*[xyzqxqyqzqw]]
            state = np.concatenate(
                [joint_state, ee_state.reshape(joint_state.shape[0], -1)],
                axis=1,
            )
        else:
            state = joint_state
        num_joint = joint_state.shape[1]

        if mask[step_index]:
            hist_state = state[
                max(0, step_index + 1 - hist_steps) : step_index + 1
            ]
        else:
            idx = step_index
            while idx > 0 and not mask[idx]:
                idx -= 1
            if idx < 0:
                idx = step_index + 1
                while idx < len(mask) and not mask[idx]:
                    idx += 1
            hist_state = state[max(0, idx - hist_steps) : idx]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(state[:1], (hist_steps - hist_state.shape[0], 1))
            hist_state = np.concatenate([padding, hist_state], axis=0)
        hist_state = np.copy(hist_state)
        hist_joint_state = hist_state[:, :num_joint]
        data["hist_joint_state"] = hist_joint_state
        if "ee_state" in data:
            hist_ee_state = hist_state[:, num_joint:]
            data["hist_ee_state"] = hist_ee_state
        if self.only_hist:
            return data

        if "master_joint_state" in data:
            if self.use_master_gripper and self.use_master_joint:
                joint_state = data["master_joint_state"]
            elif self.use_master_gripper:
                joint_state[:, self.gripper_indices] = data[
                    "master_joint_state"
                ][:, self.gripper_indices]
            elif self.use_master_joint:
                master_joint_state = copy.deepcopy(data["master_joint_state"])
                master_joint_state[:, self.gripper_indices] = joint_state[
                    :, self.gripper_indices
                ]
                joint_state = master_joint_state
            state[:, : joint_state.shape[1]] = joint_state

        idx = step_index + 1
        if idx < len(joint_state) - 1 and self.static_threshold > 0:
            static_mask = np.any(
                np.abs(joint_state[idx:] - hist_joint_state[-1])
                > self.static_threshold,
                axis=-1,
            )
            idx += np.argmax(static_mask)

        pred_state = state[idx : idx + pred_steps]
        pred_mask = mask[idx : idx + pred_steps]
        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                state[-1:], (pred_steps - pred_state.shape[0], 1)
            )
            pred_mask = np.concatenate(
                [pred_mask, np.zeros(padding.shape[0], dtype=bool)], axis=0
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)
        pred_joint_state = pred_state[:, :num_joint]
        if "ee_state" in data:
            pred_ee_state = pred_state[:, num_joint:]

        data.update(
            pred_joint_state=pred_joint_state,
            pred_mask=pred_mask,
        )
        if "ee_state" in data:
            data.update(
                pred_ee_state=pred_ee_state,
            )
        return data


class UpSampleJointState:
    def __init__(self, pred_steps, hist_steps=None):
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps

    def __call__(self, data):
        joint_state = torch.cat(
            [data["hist_joint_state"][-1:], data["pred_joint_state"]]
        )  # steps x num_joint
        pred_mask = torch.cat([data["pred_mask"][:1], data["pred_mask"]])[
            :, None
        ]
        joint_state = torch.cat(
            [joint_state, pred_mask.to(joint_state)], dim=-1
        )
        joint_state = joint_state.T[None]  # 1 x num_joint x steps

        joint_state = torch.nn.functional.interpolate(
            joint_state, self.pred_steps + 1, mode="linear", align_corners=True
        )
        data["pred_joint_state"] = joint_state[0].T[1:, :-1]
        data["pred_mask"] = joint_state[0].T[1:, -1].to(dtype=torch.bool)
        if (
            self.hist_steps is not None
            and data["hist_joint_state"].shape[0] != self.hist_steps
        ):
            data["hist_joint_state"] = torch.nn.functional.interpolate(
                data["hist_joint_state"].T[None],
                self.hist_steps,
                mode="linear",
                align_corners=True,
            )[0].T
        return data


class Resize:
    def __init__(self, dst_wh, dst_intrinsic=None):
        self.dst_wh = dst_wh
        if isinstance(dst_intrinsic, (list, tuple)):
            dst_intrinsic = np.array(dst_intrinsic)

        if dst_intrinsic is not None:
            _tmp = np.eye(4)
            _tmp[:3, :3] = dst_intrinsic[:3, :3]
            self.dst_intrinsic = _tmp
            u, v = np.arange(dst_wh[0]), np.arange(dst_wh[1])
            u = np.repeat(u[None], dst_wh[1], 0)
            v = np.repeat(v[:, None], dst_wh[0], 1)
            uv = np.stack([u, v, np.ones_like(u)], axis=-1)
            self.dst_pts = uv @ np.linalg.inv(self.dst_intrinsic[:3, :3]).T
        else:
            self.dst_intrinsic = None

    def __call__(self, data):
        if "imgs" in data:
            imgs = data["imgs"]
            resized_imgs = []
        else:
            imgs = None
        if "depths" in data:
            depths = data["depths"]
            resized_depths = []
        else:
            depths = None

        for i in range(data["intrinsic"].shape[0]):
            intrinsic = data["intrinsic"][i]
            inputs = []
            if imgs is not None:
                inputs.append(imgs[i])
            if depths is not None:
                inputs.append(depths[i])
            results, intrinsic = self.resize(inputs, intrinsic)
            data["intrinsic"][i] = intrinsic
            if imgs is not None:
                resized_imgs.append(results[0])
            if depths is not None:
                resized_depths.append(results[-1])
        if imgs is not None:
            data["imgs"] = np.stack(resized_imgs)
        if depths is not None:
            data["depths"] = np.stack(resized_depths)
        data["image_wh"] = np.array(data["imgs"].shape[1:3][::-1])
        return data

    def resize(self, inputs, intrinsic=None):
        if self.dst_intrinsic is not None:
            src_intrinsic = intrinsic[:3, :3]
            src_uv = self.dst_pts @ src_intrinsic.T
            src_uv = src_uv.astype(np.float32)
            for i, x in enumerate(inputs):
                inputs[i] = cv2.remap(
                    x,
                    src_uv[..., 0],
                    src_uv[..., 1],
                    cv2.INTER_LINEAR,
                )
            intrinsic = self.dst_intrinsic
        elif self.dst_wh is not None:
            origin_wh = inputs[0].shape[:2][::-1]
            trans_mat = np.eye(4)
            trans_mat[0, 0] = self.dst_wh[0] / origin_wh[0]
            trans_mat[1, 1] = self.dst_wh[1] / origin_wh[1]
            intrinsic = trans_mat @ intrinsic
            for i, x in enumerate(inputs):
                inputs[i] = cv2.resize(x, self.dst_wh)
        return inputs, intrinsic


class ToTensor:
    def __call__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = self.__call__(v)
            elif isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
            elif isinstance(v, (list, tuple)) and all(
                [isinstance(x, np.ndarray) for x in v]
            ):
                data[k] = type(v)([torch.from_numpy(x) for x in v])
        return data


class ConvertDataType:
    def __init__(self, convert_map, strict=False):
        self.convert_map = convert_map
        self.strict = strict

    def __call__(self, data):
        for data_name, dtype in self.convert_map.items():
            if data_name not in data and not self.strict:
                continue
            if isinstance(data[data_name], list):
                data[data_name] = torch.tensor(data[data_name])
            if isinstance(data[data_name], np.ndarray):
                data[data_name] = data[data_name].astype(dtype)
            elif isinstance(data[data_name], torch.Tensor):
                if isinstance(dtype, str):
                    dtype = getattr(torch, dtype)
                data[data_name] = data[data_name].to(dtype)
            else:
                raise TypeError(
                    f"Unsupport convert {data_name}'s "
                    f"type {type(data[data_name])} to {dtype}"
                )
        return data


class ItemSelection:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for k in list(data.keys()):
            if k not in self.keys:
                data.pop(k)
        return data


class MultiArmKinematics:
    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id=None,
        finger_keys=None,
        ee_to_gripper=None,
    ):
        super().__init__()
        self.urdf = urdf
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain_gpu = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain.to(dtype=torch.float32)
        self.chain_gpu.to(dtype=torch.float32)

        self.arm_link_keys = arm_link_keys
        self.arm_joint_id = arm_joint_id
        for keys, ids in zip(
            self.arm_link_keys, self.arm_joint_id, strict=False
        ):
            assert len(keys) == len(ids)
        self.num_arms = len(self.arm_link_keys)
        if finger_keys is None:
            finger_keys = [[]] * len(arm_link_keys)
        else:
            assert len(finger_keys) == self.num_arms, (
                "Number of gripper should equal to number of arms"
            )
        self.finger_keys = finger_keys
        if ee_to_gripper is not None:
            assert len(ee_to_gripper) == self.num_arms
            for i in range(len(ee_to_gripper)):
                if ee_to_gripper[i] is not None:
                    assert len(self.finger_keys[i]) == 0
            self.ee_to_gripper = ee_to_gripper
        else:
            self.ee_to_gripper = [None] * self.num_arms

        self.num_joints = 0
        self.num_keys = 0
        for i, single_arm_link_keys in enumerate(self.arm_link_keys):
            self.num_joints += len(single_arm_link_keys)
            self.num_joints += (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )  # Gripper
            self.num_keys += len(single_arm_link_keys)
            self.num_keys += len(self.finger_keys[i])
            self.num_keys += self.ee_to_gripper[i] is not None
        self.get_joint_relative_pos()

    def get_joint_relative_pos(self):
        joint_relative_pos = []
        for i, single_arm_joint_id_a in enumerate(self.arm_joint_id):
            joint_ids_a = torch.arange(
                len(single_arm_joint_id_a)
                + (
                    len(self.finger_keys[i]) > 0
                    or self.ee_to_gripper[i] is not None
                )
            )
            joint_relative_pos_per_arm = []
            for j, single_arm_joint_id_b in enumerate(self.arm_joint_id):
                if j == i:
                    joint_ids_b = joint_ids_a
                else:
                    joint_ids_b = torch.arange(
                        -1,
                        -(
                            len(single_arm_joint_id_b)
                            + 1
                            + (len(self.finger_keys[j]) > 0)
                        ),
                        -1,
                    )
                joint_relative_pos_per_arm.append(
                    torch.abs(joint_ids_a[:, None] - joint_ids_b)
                )
            joint_relative_pos_per_arm = torch.cat(
                joint_relative_pos_per_arm, dim=1
            )
            joint_relative_pos.append(joint_relative_pos_per_arm)
        self._joint_relative_pos = torch.cat(joint_relative_pos, dim=0)
        assert self._joint_relative_pos.shape[0] == self.num_joints
        assert self._joint_relative_pos.shape[1] == self.num_joints

    def __eq__(self, other):
        if isinstance(other, MultiArmKinematics):
            return self.urdf == other.urdf
        return False

    @property
    def joint_relative_pos(self):
        return torch.clone(self._joint_relative_pos)

    def __call__(self, data):
        joint_states = []
        valid_keys = []
        for key in ["pred_joint_state", "hist_joint_state"]:
            if key in data:
                joint_states.append(data[key])
                valid_keys.append(key)
        if len(joint_states) == 0 and "joint_state" in data:
            joint_states.append(data["joint_state"])
            valid_keys.append("joint_state")
        joint_states = torch.cat(joint_states, dim=0)
        robot_states = self.joint_state_to_robot_state(
            joint_states, data.get("embodiedment_mat")
        )
        start_idx = 0
        for key in valid_keys:
            steps = data[key].shape[0]
            data[key.replace("joint", "robot")] = robot_states[
                start_idx : start_idx + steps
            ]
            start_idx += steps

        data["joint_relative_pos"] = self.joint_relative_pos
        data["kinematics"] = self
        return data

    def joint_state_to_robot_state(
        self, joint_state, embodiedment_mat=None, return_matrix=False
    ):
        input_shape = joint_state.shape
        joint_state = joint_state.to(torch.float32)

        if joint_state.device.type == "cpu":
            chain = self.chain
        else:
            if self.chain_gpu.device != joint_state.device:
                self.chain_gpu.to(device=joint_state.device)
            chain = self.chain_gpu

        all_joint_state = torch.zeros(
            [*input_shape[:-1], len(chain.get_joints())]
        ).to(joint_state)

        start = 0
        for i, single_arm_joint_id in enumerate(self.arm_joint_id):
            num_joint = len(single_arm_joint_id)
            all_joint_state[..., single_arm_joint_id] = joint_state[
                ..., start : start + num_joint
            ]
            start += num_joint + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )
        all_joint_state = all_joint_state.flatten(end_dim=-2)
        link_poses_dict = chain.forward_kinematics(all_joint_state)

        link_poses = []
        split_size = []
        for i, (single_arm_link_keys, single_finger_keys) in enumerate(
            zip(self.arm_link_keys, self.finger_keys, strict=False)
        ):
            for key in single_arm_link_keys + single_finger_keys:
                link_poses.append(link_poses_dict[key].get_matrix())
            num_finger_keys = len(single_finger_keys)
            if self.ee_to_gripper[i] is not None:
                ee_pose = link_poses[-1]
                gripper_pose = ee_pose @ ee_pose.new_tensor(
                    self.ee_to_gripper[i]
                )
                link_poses.append(gripper_pose)
                num_finger_keys += 1

            split_size.extend([len(single_arm_link_keys), num_finger_keys])
        link_poses = torch.cat(link_poses)

        if embodiedment_mat is not None:
            if embodiedment_mat.ndim != 2:
                embodiedment_mat = embodiedment_mat.flatten(0, -3)  # [x, 4, 4]
                embodiedment_mat = embodiedment_mat.repeat(self.num_keys, 1, 1)
            link_poses = embodiedment_mat @ link_poses

        if return_matrix:
            link_poses = link_poses.unflatten(0, (self.num_keys, -1))
            link_poses = link_poses.transpose(0, 1)
            link_poses = link_poses.unflatten(0, input_shape[:-1])
            return link_poses

        robot_states = torch.cat(
            [
                link_poses[..., :3, 3],
                matrix_to_quaternion(link_poses[..., :3, :3]),
            ],
            dim=-1,
        )
        robot_states = robot_states.reshape(self.num_keys, -1, 7)

        results = list(robot_states.split(split_size))
        for i in range(self.num_arms):
            if results[i * 2 + 1].shape[0] > 1:
                results[i * 2 + 1] = results[i * 2 + 1].mean(
                    dim=0, keepdim=True
                )

        robot_states = torch.cat(results, dim=0)
        robot_states = robot_states.permute(1, 0, 2)
        robot_states = robot_states.reshape(*input_shape[:-1], -1, 7)
        robot_states = torch.cat(
            [joint_state[..., None], robot_states], dim=-1
        )
        return robot_states


class CalibrationToExtrinsic(MultiArmKinematics):
    def __init__(
        self,
        urdf,
        calibration=None,
        cam_ee_joint_indices: dict = None,
        cam_names=None,
        **kwargs,
    ):
        super().__init__(urdf, **kwargs)
        if calibration is not None:
            self.calibration = self.calibration_handler(calibration)
        else:
            self.calibration = None
        if cam_ee_joint_indices is None:
            cam_ee_joint_indices = dict(left=5, right=12)
        self.cam_ee_joint_indices = cam_ee_joint_indices
        self.cam_names = cam_names

    def calibration_handler(self, calibration):
        calibration = copy.deepcopy(calibration)
        for k, v in calibration.items():
            if isinstance(v, dict):
                v = torch.from_numpy(self._pose_to_mat(v))
            elif isinstance(v, (list, tuple)):
                v = torch.Tensor(v)
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            calibration[k] = torch.linalg.inv(v)
        return calibration

    def __call__(self, data):
        if "calibration" in data:
            calibrations = self.calibration_handler(data["calibration"])
        else:
            calibrations = self.calibration
        if calibrations is None:
            return data
        current_joint_pose = self.joint_state_to_robot_state(
            data["hist_joint_state"][-1][None], return_matrix=True
        )[0]
        cam_names = data.get("cam_names", self.cam_names)
        t_base2cam_list = []
        for cam in cam_names:
            calibration = torch.clone(calibrations[cam])
            if cam not in self.cam_ee_joint_indices:
                t_base2cam = calibration
            else:
                idx = self.cam_ee_joint_indices[cam]
                t_ee2cam = calibration
                t_ee2base = torch.eye(4)
                t_ee2base = current_joint_pose[idx]
                t_base2cam = t_ee2cam @ torch.linalg.inv(t_ee2base).to(
                    t_ee2cam
                )
            t_base2cam_list.append(t_base2cam)
        t_base2cam = torch.stack(t_base2cam_list)
        if "T_base2world" in data:
            t_world2cam = torch.linalg.solve(
                data["T_base2world"], t_base2cam, left=False
            )
        else:
            t_world2cam = t_base2cam
        data["T_world2cam"] = t_world2cam
        return data

    def _pose_to_mat(self, pose):
        if "position" in pose:
            x, y, z = pose["position"]
        else:
            x, y, z = pose["translation"]

        if "orientation" in pose:
            qx, qy, qz, w = pose["orientation"]
        else:
            qx, qy, qz, w = pose["rotation_xyzw"]
        trans = np.array([x, y, z])
        rot = Rotation.from_quat(
            [qx, qy, qz, w], scalar_first=False
        ).as_matrix()
        ret = np.eye(4)
        ret[:3, 3] = trans
        ret[:3, :3] = rot
        return ret


class GetProjectionMat:
    def __init__(self, target_coordinate="ego"):
        assert target_coordinate in ["base", "world", "ego"]
        self.target_coordinate = target_coordinate

    def __call__(self, data):
        intrinsic = data["intrinsic"]
        if self.target_coordinate == "world":
            projection_mat = intrinsic @ data["T_world2cam"]
            embodiedment_mat = data["T_base2world"]
        elif self.target_coordinate == "base":
            projection_mat = (
                intrinsic @ data["T_world2cam"] @ data["T_base2world"]
            )
            embodiedment_mat = torch.eye(4).to(projection_mat)
        elif self.target_coordinate == "ego":
            projection_mat = (
                intrinsic
                @ data["T_world2cam"]
                @ data["T_base2world"]
                @ torch.linalg.inv(data["T_base2ego"])
            )
            embodiedment_mat = data["T_base2ego"]
        data["projection_mat"] = projection_mat
        data["embodiedment_mat"] = embodiedment_mat
        return data


class UnsqueezeBatch:
    def __call__(self, data):
        for k, v in data.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                data[k] = v[None]
            else:
                data[k] = [v]
        return data


class RandomCropPaddingResize:
    def __init__(
        self,
        range_w=(-10, 10),
        range_h=(-10, 10),
        range_scale=None,
    ):
        self.range_w = range_w
        self.range_h = range_h
        self.range_scale = range_scale

    def __call__(self, data):
        if "imgs" in data:
            imgs = data["imgs"]
            aug_imgs = []
        else:
            imgs = None
        if "depths" in data:
            depths = data["depths"]
            aug_depths = []
        else:
            depths = None

        for i in range(data["intrinsic"].shape[0]):
            crop_w = int(np.random.uniform(self.range_w[0], self.range_w[1]))
            crop_h = int(np.random.uniform(self.range_h[0], self.range_h[1]))

            pad_w = int(np.random.uniform(self.range_w[0], self.range_w[1]))
            pad_h = int(np.random.uniform(self.range_h[0], self.range_h[1]))

            pad = (
                (
                    abs(pad_h) if pad_h > 0 else 0,
                    abs(pad_h) if pad_h < 0 else 0,
                ),
                (
                    abs(pad_w) if pad_w > 0 else 0,
                    abs(pad_w) if pad_w < 0 else 0,
                ),
                (0, 0),
            )

            trans_mat_crop_pad = np.eye(4)
            trans_mat_crop_pad[0, 2] = -max(crop_w, 0) + pad[1][0]
            trans_mat_crop_pad[1, 2] = -max(crop_h, 0) + pad[0][0]
            data["intrinsic"][i] = trans_mat_crop_pad @ data["intrinsic"][i]

            if self.range_scale is not None:
                scale = np.random.uniform(*self.range_scale)
                trans_mat_resize = np.eye(4)
                trans_mat_resize[0, 0] = scale
                trans_mat_resize[1, 1] = scale
                data["intrinsic"][i] = trans_mat_resize @ data["intrinsic"][i]

            if imgs is not None:
                aug_img = imgs[i][
                    max(crop_h, 0) : crop_h + imgs[i].shape[0],
                    max(crop_w, 0) : crop_w + imgs[i].shape[1],
                ]
                aug_img = np.pad(aug_img, pad)
                if self.range_scale is not None:
                    aug_img = cv2.resize(
                        aug_img,
                        (
                            int(aug_img.shape[1] * scale),
                            int(aug_img.shape[0] * scale),
                        ),
                    )
                aug_imgs.append(aug_img)

            if depths is not None:
                aug_depth = depths[i][
                    max(crop_h, 0) : crop_h + depths[i].shape[0],
                    max(crop_w, 0) : crop_w + depths[i].shape[1],
                ]
                aug_depth = np.pad(aug_depth, pad[: aug_depth.ndim])
                if self.range_scale is not None:
                    aug_depth = cv2.resize(
                        aug_depth,
                        (
                            int(aug_depth.shape[1] * scale),
                            int(aug_depth.shape[0] * scale),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                aug_depths.append(aug_depth)

        if imgs is not None:
            data["imgs"] = aug_imgs
        if depths is not None:
            data["depths"] = aug_depths
        return data


class ExtrinsicNoise:
    def __init__(self, noise_range: tuple):
        assert len(noise_range) == 6
        self.noise_range = np.array(noise_range)

    def __call__(self, data):
        num_cams = len(data["T_world2cam"])
        noise = np.random.uniform(
            -self.noise_range,
            self.noise_range,
            size=(num_cams, 6),
        )
        noise = torch.from_numpy(noise)
        noise_matrix = torch.eye(4)[None].repeat(num_cams, 1, 1)
        noise_matrix[:, :3, :3] = euler_angles_to_matrix(noise[:, :3], "XYZ")
        noise_matrix[:, :3, 3] = noise[:, 3:]
        noise_matrix = noise_matrix.to(data["T_world2cam"])
        data["T_world2cam"] = noise_matrix @ data["T_world2cam"]
        return data
