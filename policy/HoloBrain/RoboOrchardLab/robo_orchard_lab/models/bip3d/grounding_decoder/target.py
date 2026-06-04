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

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from robo_orchard_lab.models.bip3d.grounding_decoder.utils import (
    center_distance,
    get_positive_map,
    wasserstein_distance,
)

__all__ = ["Grounding3DTarget"]


class Grounding3DTarget(nn.Module):
    def __init__(
        self,
        cls_weight=1.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=1.0,
        num_dn=0,
        dn_noise_scale=0.5,
        add_neg_dn=True,
        with_dn_query=False,
        num_classes=None,
        embed_dims=256,
        label_noise_scale=0.5,
        cost_weight_wd=1.0,
        cost_weight_cd=0.8,
        use_ignore_mask=False,
    ):
        super(Grounding3DTarget, self).__init__()
        self.num_dn = num_dn

        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.dn_noise_scale = dn_noise_scale
        self.add_neg_dn = add_neg_dn
        self.with_dn_query = with_dn_query
        self.cost_weight_wd = cost_weight_wd
        self.cost_weight_cd = cost_weight_cd
        self.use_ignore_mask = use_ignore_mask
        if self.with_dn_query:
            self.num_classes = num_classes
            self.embed_dims = embed_dims
            self.label_noise_scale = label_noise_scale
            self.label_embedding = nn.Embedding(
                self.num_classes, self.embed_dims
            )

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            if not isinstance(box, torch.Tensor):
                box = torch.cat(
                    [box.gravity_center, box.tensor[..., 3:]], dim=-1
                )
            output = torch.cat(
                [box[..., :3], box[..., 3:6], box[..., 6:]],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    @torch.no_grad()
    def sample(
        self,
        cls_pred,
        box_pred,
        char_positives,
        box_target,
        text_dict,
        ignore_mask=None,
    ):
        bs, num_pred, num_cls = cls_pred.shape

        token_positive_maps = get_positive_map(char_positives, text_dict)
        token_positive_maps = [
            x.to(cls_pred).bool().float() for x in token_positive_maps
        ]
        cls_cost = self._cls_cost(
            cls_pred, token_positive_maps, text_dict["text_token_mask"]
        )
        box_target = self.encode_reg_target(box_target, box_pred.device)
        box_cost = self._box_cost(box_pred, box_target)

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])

        output_cls_target = torch.zeros_like(cls_pred)
        output_box_target = torch.zeros_like(box_pred)
        output_reg_weights = torch.ones_like(box_pred)
        if self.use_ignore_mask:
            output_ignore_mask = torch.zeros_like(
                cls_pred[..., 0], dtype=torch.bool
            )
            ignore_mask = [
                output_ignore_mask.new_tensor(x) for x in ignore_mask
            ]
        else:
            output_ignore_mask = None
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(box_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = token_positive_maps[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            if self.use_ignore_mask:
                output_ignore_mask[i, pred_idx] = ignore_mask[i][target_idx]
        self.indices = indices
        return (
            output_cls_target,
            output_box_target,
            output_reg_weights,
            output_ignore_mask,
        )

    def _cls_cost(self, cls_pred, token_positive_maps, text_token_mask):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(token_positive_maps[i]) > 0:
                pred = cls_pred[i][:, text_token_mask[i]]
                neg_cost = (
                    -(1 - pred + self.eps).log()
                    * (1 - self.alpha)
                    * pred.pow(self.gamma)
                )
                pos_cost = (
                    -(pred + self.eps).log()
                    * self.alpha
                    * (1 - pred).pow(self.gamma)
                )
                gt = token_positive_maps[i][:, text_token_mask[i]]
                cls_cost = torch.einsum(
                    "nc,mc->nm", pos_cost, gt
                ) + torch.einsum("nc,mc->nm", neg_cost, (1 - gt))
                cost.append(cls_cost)
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                pred = box_pred[i].unsqueeze(dim=-2)
                gt = box_target[i].unsqueeze(dim=-3)
                _cost = 0
                if self.cost_weight_wd > 0:
                    _cost += self.cost_weight_wd * wasserstein_distance(
                        pred, gt
                    )
                if self.cost_weight_cd > 0:
                    _cost += self.cost_weight_cd * center_distance(pred, gt)
                _cost *= self.box_weight
                cost.append(_cost)
            else:
                cost.append(None)
        return cost

    def get_dn_anchors(
        self,
        char_positives,
        box_target,
        text_dict,
        label=None,
        ignore_mask=None,
    ):
        if self.num_dn <= 0:
            return None

        char_positives = [x[: self.num_dn] for x in char_positives]
        box_target = [x[: self.num_dn] for x in box_target]

        max_dn_gt = max([len(x) for x in char_positives] + [1])
        token_positive_maps = get_positive_map(char_positives, text_dict)
        token_positive_maps = torch.stack(
            [
                F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]), value=-1)
                for x in token_positive_maps
            ]
        )
        box_target = self.encode_reg_target(box_target, box_target[0].device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        token_positive_maps = token_positive_maps.to(box_target)
        box_target = torch.where(
            (token_positive_maps == -1).all(dim=-1, keepdim=True),
            box_target.new_tensor(0),
            box_target,
        )

        bs, num_gt, state_dims = box_target.shape
        num_dn_groups = self.num_dn // max(num_gt, 1)

        if num_dn_groups > 1:
            token_positive_maps = token_positive_maps.tile(num_dn_groups, 1, 1)
            box_target = box_target.tile(num_dn_groups, 1, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        noise[..., :3] *= box_target[..., 3:6]
        noise[..., 3:6] *= box_target[..., 3:6]
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            noise_neg[..., :3] *= box_target[..., 3:6]
            noise_neg[..., 3:6] *= box_target[..., 3:6]
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(dn_anchor, box_target)
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_token_positive_maps = -torch.ones_like(token_positive_maps) * 3
        if self.add_neg_dn:
            dn_token_positive_maps = torch.cat(
                [dn_token_positive_maps, dn_token_positive_maps], dim=1
            )

        for i in range(dn_anchor.shape[0]):
            if box_cost[i] is None:
                continue
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_token_positive_maps[i, anchor_idx] = token_positive_maps[
                i, gt_idx
            ]
        dn_anchor = (
            dn_anchor.reshape(num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        text_length = dn_token_positive_maps.shape[-1]
        dn_token_positive_maps = (
            dn_token_positive_maps.reshape(
                num_dn_groups, bs, num_gt, text_length
            )
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )

        valid_mask = (dn_token_positive_maps >= 0).all(dim=-1)
        if self.add_neg_dn:
            token_positive_maps = (
                torch.cat([token_positive_maps, token_positive_maps], dim=1)
                .reshape(num_dn_groups, bs, num_gt, text_length)
                .permute(1, 0, 2, 3)
                .flatten(1, 2)
            )
            valid_mask = torch.logical_or(
                valid_mask,
                (
                    (token_positive_maps >= 0).all(dim=-1)
                    & (dn_token_positive_maps == -3).all(dim=-1)
                ),
            )  # valid denotes the items is not from pad.

        attn_mask = dn_box_target.new_ones(
            num_gt * num_dn_groups, num_gt * num_dn_groups
        )
        dn_token_positive_maps = torch.clamp(dn_token_positive_maps, min=0)
        for i in range(num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_anchor[..., 3:6] = torch.clamp(dn_anchor[..., 3:6], min=0.01).log()

        if label is not None and self.with_dn_query:
            label = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in label
                ]
            )
            label = label.tile(num_dn_groups, 1)
            if self.add_neg_dn:
                label = torch.cat([label, label], dim=1)
            label = (
                label.reshape(num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1, 2)
            )

            label_noise_mask = torch.logical_or(
                torch.rand_like(label.float()) < self.label_noise_scale * 0.5,
                label == -1,
            )
            label = torch.where(
                label_noise_mask,
                torch.randint_like(label, 0, self.num_classes),
                label,
            )
            dn_query = self.label_embedding(label)
        else:
            dn_query = None

        return (
            dn_anchor,
            dn_box_target,
            dn_token_positive_maps,
            attn_mask,
            valid_mask,
            dn_query,
        )
