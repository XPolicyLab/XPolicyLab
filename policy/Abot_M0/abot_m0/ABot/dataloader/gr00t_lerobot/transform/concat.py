# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import math
import numpy as np
import torch
import os
from pydantic import Field, PrivateAttr
from typing import Any, List, Set, Literal, Optional, Union
from ..schema import DatasetMetadata, StateActionMetadata, RotationType
from .base import InvertibleModalityTransform
import pytorch3d.transforms as pt

DeltaMode = Literal["prev", "state"]
RotOut = Literal["matrix", "rotation_6d", "quaternion", "euler_angles", "axis_angle"]
RotNorm = Literal["none", "minmax_neg1_1"]

class ConcatTransform(InvertibleModalityTransform):
    """
    Concatenate the keys according to specified order.
    """

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )

    video_concat_order: list[str] = Field(
        ...,
        description="Concatenation order for each video modality. "
        "Format: ['video.ego_view_pad_res224_freq20', ...]",
    )

    state_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each state modality. "
        "Format: ['state.position', 'state.velocity', ...].",
    )

    action_concat_order: Optional[list[str]] = Field(
        default=None,
        description="Concatenation order for each action modality. "
        "Format: ['action.position', 'action.velocity', ...].",
    )

    action_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the action keys.",
    )
    state_dims: dict[str, int] = Field(
        default_factory=dict,
        description="The dimensions of the state keys.",
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "video_concat_order",
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except:  # noqa: E722
                ### Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        if "video" in grouped_keys:
            # Check if keys in video_concat_order, state_concat_order, action_concat_order are
            # ineed contained in the data. If not, then the keys are misspecified
            video_keys = grouped_keys["video"]
            assert self.video_concat_order is not None, f"{self.video_concat_order=}, {video_keys=}"
            assert all(
                item in video_keys for item in self.video_concat_order
            ), f"keys in video_concat_order are misspecified, \n{video_keys=}, \n{self.video_concat_order=}"

            # Process each video view
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_video = np.expand_dims(
                    video_data, axis=-4
                )  # [..., H, W, C] -> [..., 1, H, W, C]
                unsqueezed_videos.append(unsqueezed_video)
            # Concatenate along the new axis
            unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [..., V, H, W, C]

            # Video
            data["video"] = unsqueezed_video

        # "state"
        if "state" in grouped_keys:
            state_keys = grouped_keys["state"]
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            assert all(
                item in state_keys for item in self.state_concat_order
            ), f"keys in state_concat_order are misspecified, \n{state_keys=}, \n{self.state_concat_order=}"
            # Check the state dims
            for key in self.state_concat_order:
                target_shapes = [self.state_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(6)  # Allow for rotation_6d
                # if key in ["state.right_arm", "state.right_hand"]:
                target_shapes.append(self.state_dims[key] * 2)  # Allow for sin-cos transform
                assert (
                    data[key].shape[-1] in target_shapes
                ), f"State dim mismatch for {key=}, {data[key].shape[-1]=}, {target_shapes=}"
            # Concatenate the state keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["state"] = torch.cat(
                [data.pop(key) for key in self.state_concat_order], dim=-1
            )  # [T, D_state]

        if "action" in grouped_keys:
            action_keys = grouped_keys["action"]
            assert self.action_concat_order is not None, f"{self.action_concat_order=}"
            # Check if all keys in concat_order are present
            assert set(self.action_concat_order) == set(
                action_keys
            ), f"{set(self.action_concat_order)=}, {set(action_keys)=}"
            # Record the action dims
            for key in self.action_concat_order:
                target_shapes = [self.action_dims[key]]
                if self.is_rotation_key(key):
                    target_shapes.append(3)  # Allow for axis angle
                assert (
                    self.action_dims[key] == data[key].shape[-1]
                ), f"Action dim mismatch for {key=}, {self.action_dims[key]=}, {data[key].shape[-1]=}"
            # Concatenate the action keys
            # We'll have StateActionToTensor before this transform, so here we use torch.cat
            data["action"] = torch.cat(
                [data.pop(key) for key in self.action_concat_order], dim=-1
            )  # [T, D_action]

        return data

    def unapply(self, data: dict) -> dict:
        start_dim = 0
        assert "action" in data, f"{data.keys()=}"
        # For those dataset without actions (LAPA), we'll never run unapply
        assert self.action_concat_order is not None, f"{self.action_concat_order=}"
        action_tensor = data.pop("action")
        for key in self.action_concat_order:
            if key not in self.action_dims:
                raise ValueError(f"Action dim {key} not found in action_dims.")
            end_dim = start_dim + self.action_dims[key]
            data[key] = action_tensor[..., start_dim:end_dim]
            start_dim = end_dim
        if "state" in data:
            assert self.state_concat_order is not None, f"{self.state_concat_order=}"
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.state_dims[key]
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None, "Metadata not set"
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
        assert isinstance(
            modality_config[subkey], StateActionMetadata
        ), f"Expected {StateActionMetadata} for {subkey=}, got {type(modality_config[subkey])=}"
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        """Get the dimension of a state or action key from the dataset metadata."""
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1, f"{shape=}"
        return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        modality_config = self.get_modality_metadata(key)
        return modality_config.rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata and compute the dimensions of the state and action keys."""
        super().set_metadata(dataset_metadata)
        # Pre-compute the dimensions of the state and action keys
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)

class ConcatStateActionTransform(InvertibleModalityTransform):
    """
    Concatenate state/action features into unified `state` and `action` tensors.
    """
    
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    state_concat_order: Optional[list[str]] = Field(default=None)
    action_concat_order: Optional[list[str]] = Field(default=None)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {
                "state_concat_order",
                "action_concat_order",
            }
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        if self.state_concat_order:
            for key in self.state_concat_order:
                assert key in data, f"Missing state key: {key}"
                assert isinstance(data[key], torch.Tensor), f"{key} must be torch.Tensor, got {type(data[key])}"
            data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)

        if self.action_concat_order:
            for key in self.action_concat_order:
                assert key in data, f"Missing action key: {key}"
                assert isinstance(data[key], torch.Tensor), f"{key} must be torch.Tensor, got {type(data[key])}"
            data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)

        return data

    def unapply(self, data: dict) -> dict:
        raise NotImplementedError("unapply is not supported.")

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

class ConcatDeltaChunkTransform(InvertibleModalityTransform):
    """
    Compute chunk-wise delta for action features.
    - Drop-first alignment: output length = T-1 (t=1..T-1).
    - position_keys: a[1:] - a[:-1]
    - rotation_keys: controlled by rotation_delta_specs
    - no_delta_keys: keep abs
    """

    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    action_keys: List[str] = Field(...)

    match_by: Literal["suffix", "full"] = Field(default="suffix")
    position_keys: Set[str] = Field(default_factory=set)
    rotation_keys: Set[str] = Field(default_factory=set)
    no_delta_keys: Set[str] = Field(default_factory=set)

    # for no-delta keys: keep a[1:] or a[:-1]
    no_delta_align: Literal["drop_first", "drop_last"] = Field(default="drop_first")

    # supported: "euler_delta_sub", "euler_delta_rel", "rotation_6d_delta_rel", "quaternion_delta_rel",
    #            "axis_angle_delta_rel", "matrix_delta_rel"
    rotation_delta_specs: str = Field(default="euler_delta_sub")

    euler_convention: str = Field(default="XYZ")
    quat_order_in: str = Field(default="wxyz")
    quat_wxyz_out: bool = Field(default=True)
    canonicalize_quat: bool = Field(default=True)

    _rot_in_type_by_fullkey: dict[str, str] = PrivateAttr(default_factory=dict)
    _ak_kind: dict[str, str] = PrivateAttr(default_factory=dict)
    _quat_ident: Optional[torch.Tensor] = PrivateAttr(default=None)

    def _match_token(self, full_key: str) -> str:
        return full_key if self.match_by == "full" else full_key.split(".")[-1]

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim != 2:
            raise AssertionError(f"Expected 1D/2D tensor, got {x.ndim}D with shape {tuple(x.shape)}")
        return x

    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _as_quat_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        if self.quat_order_in == "wxyz":
            return q
        if self.quat_order_in == "xyzw":
            return q[:, [3, 0, 1, 2]]
        raise ValueError(f"Unknown quat_order_in={self.quat_order_in}, expected 'wxyz' or 'xyzw'")

    def _quat_safe_norm_wxyz(self, q_wxyz: torch.Tensor) -> torch.Tensor:
        q_wxyz = self._sanitize(q_wxyz)
        if q_wxyz.dtype not in (torch.float32, torch.float64):
            q_wxyz = q_wxyz.float()
        else:
            q_wxyz = q_wxyz.to(torch.float32)

        eps = 1e-12
        n = torch.linalg.norm(q_wxyz, dim=-1, keepdim=True)
        bad = n < eps
        q_wxyz = q_wxyz / n.clamp_min(eps)

        if bad.any():
            ident = self._quat_ident
            if ident is None or ident.device != q_wxyz.device or ident.dtype != q_wxyz.dtype:
                ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q_wxyz.device, dtype=q_wxyz.dtype)
                self._quat_ident = ident
            q_wxyz[bad.squeeze(-1)] = ident

        if self.canonicalize_quat:
            sign = torch.where(q_wxyz[:, :1] < 0, -1.0, 1.0)
            q_wxyz = q_wxyz * sign
        return q_wxyz

    @staticmethod
    def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    @staticmethod
    def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=-1,
        )

    def _quat_rel_wxyz(self, q_wxyz: torch.Tensor) -> torch.Tensor:
        q = self._quat_safe_norm_wxyz(q_wxyz)
        q_next = q[1:]
        q_prev = q[:-1]
        q_rel = self._quat_mul_wxyz(q_next, self._quat_conj_wxyz(q_prev))
        if self.canonicalize_quat:
            sign = torch.where(q_rel[:, :1] < 0, -1.0, 1.0)
            q_rel = q_rel * sign
        return self._sanitize(q_rel)

    def _rot_in_to_matrix(self, rot: torch.Tensor, from_rep: str) -> torch.Tensor:
        rot = self._sanitize(rot)
        if rot.dtype not in (torch.float32, torch.float64):
            rot = rot.float()
        else:
            rot = rot.to(torch.float32)

        fr = (from_rep or "").lower()

        if fr in ["euler_angles_rpy", "rpy", "euler_rpy", "euler"]:
            e = self._wrap_to_pi(rot)
            return self._sanitize(pt.euler_angles_to_matrix(e, self.euler_convention))

        if fr in ["quaternion", "quat"] or ("quat" in fr):
            q = self._as_quat_wxyz(rot)
            q = self._quat_safe_norm_wxyz(q)
            return self._sanitize(pt.quaternion_to_matrix(q))

        if fr in ["axis_angle", "axisangle"]:
            return self._sanitize(pt.axis_angle_to_matrix(rot))

        if fr in ["matrix", "rotmat", "rotation_matrix"]:
            if rot.ndim == 3 and rot.shape[-2:] == (3, 3):
                return self._sanitize(rot)
            if rot.ndim == 2 and rot.shape[1] == 9:
                return self._sanitize(rot.view(-1, 3, 3))
            raise ValueError(f"matrix input expects [T,9] or [T,3,3], got {tuple(rot.shape)}")

        if fr in ["rotation_6d", "rot6d", "6d"]:
            return self._sanitize(pt.rotation_6d_to_matrix(rot))

        raise ValueError(f"Unsupported rot input type: {from_rep}")

    def _rot_in_to_euler(self, rot: torch.Tensor, from_rep: str) -> torch.Tensor:
        rot = self._sanitize(rot)
        if rot.dtype not in (torch.float32, torch.float64):
            rot = rot.float()
        else:
            rot = rot.to(torch.float32)

        fr = (from_rep or "").lower()
        if fr in ["euler_angles_rpy", "rpy", "euler_rpy", "euler"]:
            return self._sanitize(self._wrap_to_pi(rot))

        R = self._rot_in_to_matrix(rot, from_rep)
        e = pt.matrix_to_euler_angles(R, self.euler_convention)
        return self._sanitize(self._wrap_to_pi(e))

    def _rotmat_to_out(self, R: torch.Tensor, out_type: str) -> torch.Tensor:
        out_type = (out_type or "").lower()
        R = self._sanitize(R)

        if out_type in ["matrix", "rotmat"]:
            out = R.reshape(-1, 9)
        elif out_type in ["rotation_6d", "rot6d", "6d"]:
            out = pt.matrix_to_rotation_6d(R)
        elif out_type in ["quaternion", "quat"]:
            q_wxyz = pt.matrix_to_quaternion(R)
            if self.canonicalize_quat:
                sign = torch.where(q_wxyz[:, :1] < 0, -1.0, 1.0)
                q_wxyz = q_wxyz * sign
            out = q_wxyz if self.quat_wxyz_out else q_wxyz[:, [1, 2, 3, 0]]
        elif out_type in ["axis_angle", "axisangle"]:
            out = pt.matrix_to_axis_angle(R)
        elif out_type in ["euler_angles_rpy", "rpy", "euler_rpy", "euler"]:
            out = self._wrap_to_pi(pt.matrix_to_euler_angles(R, self.euler_convention))
        else:
            raise ValueError(f"Unsupported output rotation type: {out_type}")

        return self._sanitize(out)

    def rotation_type_to_str(self, rt: Optional[Union[str, Any]]) -> Optional[str]:
        if rt is None:
            return None
        if isinstance(rt, str):
            return rt
        if hasattr(rt, "value"):
            return str(rt.value)
        return str(rt)

    def set_metadata(self, dataset_metadata: Any):
        modality_metadata = dataset_metadata.modalities
        assert hasattr(modality_metadata, "action"), "dataset_metadata.modalities.action missing"

        self._ak_kind.clear()
        for ak in self.action_keys:
            tok = self._match_token(ak)
            if tok in self.position_keys:
                self._ak_kind[ak] = "pos"
            elif tok in self.rotation_keys:
                self._ak_kind[ak] = "rot"
            elif tok in self.no_delta_keys:
                self._ak_kind[ak] = "nodelta"
            else:
                self._ak_kind[ak] = "linear"

        self._rot_in_type_by_fullkey.clear()
        for full_key in self.action_keys:
            if self._ak_kind.get(full_key) != "rot":
                continue

            _, subkey = full_key.split(".", 1)
            meta = modality_metadata.action[subkey]
            rt = self.rotation_type_to_str(getattr(meta, "rotation_type", None))

            assert rt in ["euler_angles_rpy", "quaternion", "axis_angle", "matrix", "rotation_6d"], \
                f"Unexpected rotation_type={getattr(meta,'rotation_type',None)} (normalized={rt}) for {full_key}"

            self._rot_in_type_by_fullkey[full_key] = rt

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        spec = (self.rotation_delta_specs or "").strip()

        for ak in self.action_keys:
            if ak not in data:
                continue

            a = self._ensure_2d(data[ak])
            T = a.shape[0]
            if T <= 1:
                data[ak] = a.new_zeros((0, a.shape[1]))
                continue

            kind = self._ak_kind.get(ak, "linear")

            if kind == "pos":
                data[ak] = a[1:] - a[:-1]
                continue

            if kind == "nodelta":
                if self.no_delta_align == "drop_first":
                    data[ak] = a[1:]
                elif self.no_delta_align == "drop_last":
                    data[ak] = a[:-1]
                else:
                    raise ValueError(f"Unknown no_delta_align: {self.no_delta_align}")
                continue

            if kind == "rot":
                in_type = self._rot_in_type_by_fullkey.get(ak, None)
                assert in_type is not None, f"rotation input type not set for {ak} (did set_metadata run?)"
                fr = (in_type or "").lower()

                if fr in ["quaternion", "quat"] or ("quat" in fr):
                    if a.shape[1] != 4:
                        raise ValueError(f"{ak}: quaternion input expects dim=4, got {a.shape}")

                    q_wxyz = self._as_quat_wxyz(a)
                    q_rel = self._quat_rel_wxyz(q_wxyz)

                    if spec == "euler_delta_sub":
                        Rrel = pt.quaternion_to_matrix(q_rel)
                        out = self._wrap_to_pi(pt.matrix_to_euler_angles(Rrel, self.euler_convention))
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    if spec == "quaternion_delta_rel":
                        out = q_rel if self.quat_wxyz_out else q_rel[:, [1, 2, 3, 0]]
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    if spec == "axis_angle_delta_rel":
                        out = pt.matrix_to_axis_angle(pt.quaternion_to_matrix(q_rel))
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    if spec == "rotation_6d_delta_rel":
                        out = pt.matrix_to_rotation_6d(pt.quaternion_to_matrix(q_rel))
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    if spec == "matrix_delta_rel":
                        out = pt.quaternion_to_matrix(q_rel).reshape(-1, 9)
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    if spec == "euler_delta_rel":
                        out = self._wrap_to_pi(
                            pt.matrix_to_euler_angles(pt.quaternion_to_matrix(q_rel), self.euler_convention)
                        )
                        data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                        continue

                    raise ValueError(f"Unknown rotation_delta_specs: {spec}")

                if spec == "euler_delta_sub":
                    e = self._rot_in_to_euler(a, in_type)
                    data[ak] = self._sanitize(self._wrap_to_pi(e[1:] - e[:-1])).to(dtype=a.dtype, device=a.device)
                    continue

                if spec == "euler_delta_rel":
                    out_type = "euler_angles_rpy"
                elif spec == "rotation_6d_delta_rel":
                    out_type = "rotation_6d"
                elif spec == "quaternion_delta_rel":
                    out_type = "quaternion"
                elif spec == "axis_angle_delta_rel":
                    out_type = "axis_angle"
                elif spec == "matrix_delta_rel":
                    out_type = "matrix"
                else:
                    raise ValueError(f"Unknown rotation_delta_specs: {spec}")

                R = self._rot_in_to_matrix(a, in_type)
                Rrel = torch.bmm(R[1:], R[:-1].transpose(1, 2))
                out = self._rotmat_to_out(Rrel, out_type)
                data[ak] = self._sanitize(out).to(dtype=a.dtype, device=a.device)
                continue

            data[ak] = a[1:] - a[:-1]

        return data

    def unapply(self, data: dict) -> dict:
        raise NotImplementedError("unapply is not supported.")

    def __call__(self, data: dict) -> dict:
        return self.apply(data)