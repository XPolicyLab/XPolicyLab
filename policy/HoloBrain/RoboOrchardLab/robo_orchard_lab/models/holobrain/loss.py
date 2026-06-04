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

import logging

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
from torch import nn

from robo_orchard_lab.models.holobrain.utils import recompute

logger = logging.getLogger(__name__)


class HoloBrainActionLoss(nn.Module):
    def __init__(
        self,
        default_state_loss_weight=None,
        default_fk_loss_weight=None,
        default_mobile_loss_weight=1.0,
        loss_mode="l2",
        smooth_l1_beta=1.0,
        with_wasserstein_distance=True,
        with_consistent_loss=False,
        timestep_loss_weight=None,
        parallel_loss_weight=None,
        **kwargs,
    ):
        super().__init__()
        if len(kwargs) != 0:
            logger.warning(f"Get unexpected arguments: {kwargs}")
        assert loss_mode in ["l1", "l2", "smooth_l1"]
        self.state_loss_weight = default_state_loss_weight
        self.fk_loss_weight = default_fk_loss_weight
        self.mobile_loss_weight = default_mobile_loss_weight
        self.loss_mode = loss_mode
        self.smooth_l1_beta = smooth_l1_beta
        self.with_wasserstein_distance = with_wasserstein_distance
        self.with_consistent_loss = with_consistent_loss
        self.timestep_loss_weight = timestep_loss_weight
        self.parallel_loss_weight = parallel_loss_weight

    def forward(self, model_outs, inputs, **kwargs):
        pred = model_outs["pred"]
        target = model_outs["target"]
        pred_mask = inputs.get("pred_mask")
        output = {}
        output.update(
            self.robot_state_loss(
                pred,
                target,
                inputs.get("state_loss_weights", self.state_loss_weight),
                pred_mask=pred_mask,
                timestep=model_outs["timesteps"],
                num_parallel=model_outs["num_parallel"],
            )
        )
        fk_loss_weight = inputs.get("fk_loss_weight", self.fk_loss_weight)
        if fk_loss_weight is not None:
            fk_pred = recompute(pred, inputs)
            output.update(
                self.robot_state_loss(
                    fk_pred,
                    target,
                    fk_loss_weight,
                    pred_mask=pred_mask,
                    timestep=model_outs["timesteps"],
                    num_parallel=model_outs["num_parallel"],
                    suffix="_fk",
                )
            )
            if self.with_consistent_loss:
                output.update(
                    self.robot_state_loss(
                        pred,
                        fk_pred.detach(),
                        fk_loss_weight,
                        pred_mask=None,
                        timestep=model_outs["timesteps"],
                        num_parallel=model_outs["num_parallel"],
                        suffix="_consistent",
                    )
                )
        if model_outs.get("pred_mobile_traj") is not None:
            output.update(
                self.mobile_trajectory_loss(
                    model_outs["pred_mobile_traj"],
                    model_outs.get("target_mobile_traj"),
                    inputs.get("mobile_loss_weight", self.mobile_loss_weight),
                    pred_mask=pred_mask,
                    timestep=model_outs["timesteps"],
                    num_parallel=model_outs["num_parallel"],
                )
            )
        return output

    def robot_state_loss(self, pred, target, weight, suffix="", **kwargs):
        rot_size = pred.shape[-1] - 4
        pred_angle, pred_xyz, pred_rot = pred.split([1, 3, rot_size], dim=-1)
        tgt_angle, tgt_xyz, tgt_rot = target.split([1, 3, rot_size], dim=-1)

        if weight is not None:
            w_rot_size = weight.shape[-1] - 4
            w_angle, w_xyz, w_rot = weight.split([1, 3, w_rot_size], dim=-1)
        else:
            w_angle = w_xyz = w_rot = None

        if self.with_wasserstein_distance:
            pred_rot = quaternion_to_matrix(pred_rot).flatten(-2)
            tgt_rot = quaternion_to_matrix(tgt_rot).flatten(-2)
            if w_rot is not None:
                w_rot = w_rot[..., :9]
                scale = w_rot.sum(dim=-1, keepdim=True) * w_rot.shape[-1] / 9
                w_rot = F.pad(
                    w_rot, (0, 9 - w_rot.shape[-1], 0, 0), "replicate"
                )
                w_rot = w_rot * scale

        loss_angle = self._loss_func(pred_angle, tgt_angle, w_angle, **kwargs)
        loss_xyz = self._loss_func(pred_xyz, tgt_xyz, w_xyz, **kwargs)
        loss_rot = self._loss_func(pred_rot, tgt_rot, w_rot, **kwargs)
        return {
            f"loss_angle{suffix}": loss_angle,
            f"loss_xyz{suffix}": loss_xyz,
            f"loss_rot{suffix}": loss_rot,
        }

    def mobile_trajectory_loss(self, pred, target, weight, **kwargs):
        if target is None:
            loss_mobile = self._fake_loss(pred)
        else:
            self._loss_func(pred, target, weight, **kwargs)
        return {"loss_mobile": loss_mobile}

    def _loss_func(
        self,
        pred,
        target,
        weight=None,
        pred_mask=None,
        timestep=None,
        num_parallel=None,
    ):
        if self.loss_mode == "l2":
            error = torch.square(pred - target)
        elif self.loss_mode == "l1":
            error = torch.abs(pred - target)
        elif self.loss_mode == "smooth_l1":
            error = torch.abs(pred - target)
            error = torch.where(
                error < self.smooth_l1_beta,
                0.5 * error * error / self.smooth_l1_beta,
                error - 0.5 * self.smooth_l1_beta,
            )

        if num_parallel is not None:
            error = error.unflatten(0, (-1, num_parallel)).transpose(0, 1)

        if weight is not None:
            if isinstance(weight, torch.Tensor):
                weight = weight.to(error)
            else:
                weight = error.new_tensor(weight)
            error = error * weight

        if timestep is not None and self.timestep_loss_weight is not None:
            if num_parallel is not None:
                timestep = timestep.reshape(-1, num_parallel).transpose(0, 1)
            timestep_weight = self.timestep_loss_weight / (timestep + 1)
            while timestep_weight.dim() < error.dim():
                timestep_weight = timestep_weight.unsqueeze(-1)
            error = error * timestep_weight

        if num_parallel is not None:
            if self.parallel_loss_weight is not None:
                min_idx = error.flatten(2).sum(dim=-1).argmin(dim=0)
                bs = error.shape[1]
                bs_idx = torch.arange(bs).to(min_idx)

                parallel_weight = error.new_full(
                    [num_parallel, bs], self.parallel_loss_weight
                )
                parallel_weight[min_idx, bs_idx] = 1
                while parallel_weight.dim() < error.dim():
                    parallel_weight = parallel_weight.unsqueeze(-1)
                error = (error * parallel_weight).sum(dim=0)
            else:
                error = error.mean(dim=0)

        if pred_mask is not None:
            error = error[pred_mask]
            if error.shape[0] == 0:
                return self._fake_loss(pred)
        loss = error.sum(dim=-1).mean()
        return loss

    def _fake_loss(self, error):
        return error.sum() * 0
