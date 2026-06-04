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

import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from robo_orchard_lab.distributed.utils import reduce_mean
from robo_orchard_lab.models.bip3d.grounding_decoder.utils import (
    center_distance,
    convert_grounding_to_cls_scores,
    create_positive_map_label_to_token,
    decode_box,
    get_entities,
    get_positive_map,
    linear_act_ln,
    rotation_3d_in_euler,
    wasserstein_distance,
)
from robo_orchard_lab.models.bip3d.utils import deformable_format
from robo_orchard_lab.models.layers.transformer_layers import MLP, LayerScale
from robo_orchard_lab.utils.build import build

__all__ = [
    "BBox3DDecoder",
    "GroundingRefineClsHead",
    "DoF9BoxLoss",
    "GroundingBox3DPostProcess",
    "DoF9BoxEncoder",
    "SparseBox3DKeyPointsGenerator",
]


class BBox3DDecoder(nn.Module):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        text_cross_attn: Union[None, dict, nn.Module] = None,
        loss_cls: Union[None, dict, nn.Module] = None,
        loss_reg: Union[None, dict, nn.Module] = None,
        post_processor: Union[None, dict, nn.Module] = None,
        sampler: Union[None, dict, nn.Module] = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        task_prefix: str = "det",
        reg_weights: Optional[List] = None,
        operation_order: Optional[List[str]] = None,
        look_forward_twice: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.task_prefix = task_prefix
        self.look_forward_twice = look_forward_twice

        if reg_weights is None:
            self.reg_weights = [1.0] * 9
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "gnn",
                "norm",
                "text_cross_attn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        self.instance_bank = build(instance_bank)
        self.anchor_encoder = build(anchor_encoder)
        self.sampler = build(sampler)
        self.post_processor = build(post_processor)
        self.loss_cls = build(loss_cls)
        self.loss_reg = build(loss_reg)
        self.op_config_map = {
            "gnn": graph_model,
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "text_cross_attn": text_cross_attn,
            "refine": refine_layer,
        }
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            layer = build(self.op_config_map.get(op, None))
            assert isinstance(layer, torch.nn.Module) or layer is None
            if layer is None:
                layer = nn.Identity()
            self.layers.append(layer)
        self.embed_dims = self.instance_bank.embed_dims
        self.norm = nn.LayerNorm(self.embed_dims)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for i, op in enumerate(self.operation_order):
            if op == "refine":
                m = self.layers[i]
                nn.init.constant_(m.layers[-2].weight, 0.0)
                nn.init.constant_(m.layers[-2].bias, 0.0)
                nn.init.constant_(m.layers[-1].weight, 1)
                nn.init.constant_(m.layers[-2].bias.data[2:], 0.0)

    def save_metadata(self, directory, *args, **kwargs):
        self.instance_bank.save_anchor(directory)

    def forward(
        self,
        feature_maps,
        text_dict: Optional[dict] = None,
        inputs: Optional[dict] = None,
        depth_prob=None,
        **kwargs,
    ):
        batch_size = feature_maps[0].shape[0]
        feature_maps = list(deformable_format(feature_maps))

        # ========= get instance info ============
        (
            instance_feature,
            anchor,
        ) = self.instance_bank.get(batch_size, inputs)

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            assert (
                isinstance(inputs, dict)
                and (text_dict is not None)
                and (self.sampler is not None)
            )
            dn_metas = self.sampler.get_dn_anchors(
                inputs[self.gt_cls_key],
                inputs[self.gt_reg_key],
                text_dict=text_dict,
                label=inputs["gt_labels_3d"],
            )
        else:
            assert self.post_processor is not None
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_query,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            if dn_query is None:
                dn_query = instance_feature.new_zeros(
                    batch_size, num_dn_anchor, instance_feature.shape[-1]
                )
            instance_feature = torch.cat(
                [instance_feature, dn_query],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
        else:
            num_dn_anchor = None
            num_free_instance = None

        anchor_embed = self.anchor_encoder(anchor)

        # =================== forward the layers ====================
        prediction = []
        classification = []
        _anchor = None
        for i, (op, layer) in enumerate(
            zip(self.operation_order, self.layers, strict=False)
        ):
            if self.layers[i] is None:
                continue
            elif op == "gnn":
                instance_feature = layer(
                    query=instance_feature,
                    key=instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = layer(instance_feature)
            elif op == "deformable":
                instance_feature = layer(
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    inputs,
                    depth_prob=depth_prob,
                )
            elif op == "text_cross_attn":
                assert text_dict is not None, (
                    "text_dict should not be None when using text_cross_attn"
                )
                text_feature = text_dict["embedded"]
                instance_feature = layer(
                    query=instance_feature,
                    key=text_feature,
                    value=text_feature,
                    query_pos=anchor_embed,
                    key_padding_mask=~text_dict["text_token_mask"],
                    key_pos=0,
                )
            elif op == "refine":
                _instance_feature = self.norm(instance_feature)
                if self.look_forward_twice:
                    if _anchor is None:
                        _anchor = anchor.clone()
                    _anchor, cls = layer(
                        _instance_feature,
                        _anchor,
                        anchor_embed,
                        text_feature=text_feature,
                        text_token_mask=(
                            text_dict["text_token_mask"]
                            if text_dict is not None
                            else None
                        ),
                    )
                    prediction.append(_anchor)
                    anchor = layer(
                        instance_feature,
                        anchor,
                        anchor_embed,
                    )[0]
                    anchor_embed = self.anchor_encoder(anchor)
                    _anchor = anchor
                    anchor = anchor.detach()
                else:
                    anchor, cls = layer(
                        _instance_feature,
                        anchor,
                        anchor_embed,
                        text_feature=text_feature,
                        text_token_mask=text_dict["text_token_mask"],
                    )
                    anchor_embed = self.anchor_encoder(anchor)
                    prediction.append(anchor)
                classification.append(cls)

            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [x[:, -num_dn_anchor:] for x in classification]
            classification = [x[:, :-num_dn_anchor] for x in classification]
            dn_prediction = [x[:, -num_dn_anchor:] for x in prediction]
            prediction = [x[:, :-num_dn_anchor] for x in prediction]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            dn_anchor = anchor[:, -num_dn_anchor:]
            instance_feature = instance_feature[:, :-num_dn_anchor]
            anchor_embed = anchor_embed[:, :-num_dn_anchor]
            anchor = anchor[:, :-num_dn_anchor]
            cls = cls[:, :-num_dn_anchor]

        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "instance_feature": instance_feature,
                "anchor_embed": anchor_embed,
            }
        )
        return output

    def loss(self, model_outs, data, text_dict=None):
        assert (
            (self.loss_cls is not None)
            and (self.loss_reg is not None)
            and (self.sampler is not None)
        )
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds, strict=False)
        ):
            reg = reg[..., : len(self.reg_weights)]
            reg = decode_box(reg)
            cls_target, reg_target, reg_weights, ignore_mask = (
                self.sampler.sample(
                    cls,
                    reg,
                    data[self.gt_cls_key],
                    data[self.gt_reg_key],
                    text_dict=text_dict,
                    ignore_mask=data.get("ignore_mask"),
                )
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask = mask.reshape(-1)
            if ignore_mask is not None:
                ignore_mask = ~ignore_mask.reshape(-1)
                mask = torch.logical_and(mask, ignore_mask)
                ignore_mask = ignore_mask.tile(1, cls.shape[-1])

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            token_mask = torch.logical_not(cls.isinf())
            cls = cls[token_mask]
            cls_target = cls_target[token_mask]
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                prefix=f"{self.task_prefix}_",
                suffix=f"_{decoder_idx}",
            )

            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)

        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds, strict=False)
        ):
            cls = cls.flatten(end_dim=1)[dn_valid_mask]
            mask = torch.logical_not(cls.isinf())
            cls_loss = self.loss_cls(
                cls[mask],
                dn_cls_target[mask],
                avg_factor=num_dn_pos,
            )

            reg = reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                ..., : len(self.reg_weights)
            ]
            reg = decode_box(reg)
            reg_loss = self.loss_reg(
                reg,
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                prefix=f"{self.task_prefix}_",
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target.sum(dim=-1) > 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def post_process(
        self,
        model_outs,
        inputs,
        text_dict,
    ):
        assert self.post_processor is not None
        results = self.post_processor(
            model_outs["classification"],
            model_outs["prediction"],
            inputs=inputs,
            text_dict=text_dict,
        )
        return results


class GroundingRefineClsHead(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        output_dim=9,
        scale=None,
        cls_layers=False,
        cls_bias=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.refine_state = list(range(output_dim))
        self.scale = scale
        self.layers = nn.Sequential(
            *linear_act_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            LayerScale(dim=self.output_dim, scale=1.0),
        )
        if cls_layers:
            self.cls_layers = nn.Sequential(
                MLP(embed_dims, embed_dims, embed_dims, 2),
                nn.LayerNorm(self.embed_dims),
            )
        else:
            self.cls_layers = nn.Identity()
        if cls_bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True
            )
        else:
            self.bias = None

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor = None,
        anchor_embed: torch.Tensor = None,
        text_feature=None,
        text_token_mask=None,
        **kwargs,
    ):
        if anchor_embed is not None:
            feature = instance_feature + anchor_embed
        else:
            feature = instance_feature
        output = self.layers(feature)
        if self.scale is not None:
            output = output * output.new_tensor(self.scale)
        if anchor is not None:
            output = output + anchor

        if text_feature is not None:
            cls = self.cls_layers(instance_feature) @ text_feature.transpose(
                -1, -2
            )
            cls = cls / math.sqrt(instance_feature.shape[-1])
            if self.bias is not None:
                cls = cls + self.bias
            if text_token_mask is not None:
                cls.masked_fill_(~text_token_mask[:, None, :], float("-inf"))
        else:
            cls = None
        return output, cls


class DoF9BoxLoss(nn.Module):
    def __init__(
        self,
        loss_weight_wd=1.0,
        loss_weight_cd=0.8,
        decode_pred=False,
    ):
        super().__init__()
        self.loss_weight_wd = loss_weight_wd
        self.loss_weight_cd = loss_weight_cd
        self.decode_pred = decode_pred

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        **kwargs,
    ):
        if box_target.shape[0] == 0:
            loss = box.sum() * 0
            return {f"{prefix}loss_box{suffix}": loss}
        if self.decode_pred:
            box = decode_box(box)
        loss = 0
        if self.loss_weight_wd > 0:
            loss += self.loss_weight_wd * wasserstein_distance(box, box_target)
        if self.loss_weight_cd > 0:
            loss += self.loss_weight_cd * center_distance(box, box_target)

        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
        output = {f"{prefix}loss_box{suffix}": loss}
        return output


class FocalLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
    ):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        avg_factor=None,
    ):
        if pred.dim() != target.dim():
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]

        loss_cls = self.loss_weight * self.py_sigmoid_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            alpha=self.alpha,
            avg_factor=avg_factor,
        )
        return loss_cls

    def py_sigmoid_focal_loss(
        self,
        pred,
        target,
        gamma=2.0,
        alpha=0.25,
        avg_factor=None,
    ):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(
            gamma
        )
        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / (avg_factor + torch.finfo(torch.float32).eps)
        return loss


class GroundingBox3DPostProcess:
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(GroundingBox3DPostProcess, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def __call__(
        self,
        cls_scores,
        box_preds,
        text_dict: dict,
        inputs: dict,
        output_idx=-1,
    ):
        cls_scores = cls_scores[output_idx].sigmoid()
        if "tokens_positive" in inputs:
            tokens_positive_maps = get_positive_map(
                inputs["tokens_positive"],
                text_dict,
            )
            label_to_token = [
                create_positive_map_label_to_token(x, plus=1)
                for x in tokens_positive_maps
            ]
            cls_scores = convert_grounding_to_cls_scores(
                cls_scores, label_to_token
            )
            entities = get_entities(
                inputs["text"],
                inputs["tokens_positive"],
            )
        else:
            cls_scores, _ = cls_scores.max(dim=-1, keepdim=True)
            entities = inputs["text"]

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        num_output = min(self.num_output, num_pred * num_cls)
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            num_output, dim=1, sorted=self.sorted
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            box = decode_box(box, 0.1, 20)
            category_ids = category_ids.cpu()

            label_names = []
            for id in category_ids.tolist():
                if isinstance(entities[i], (tuple, list)):
                    label_names.append(entities[i][id])
                else:
                    label_names.append(entities[i])

            output.append(
                {
                    "bboxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids,
                    "target_scores_3d": scores.cpu(),
                    "label_names": label_names,
                }
            )
        return output


class DoF9BoxEncoder(nn.Module):
    def __init__(
        self,
        embed_dims,
        rot_dims=3,
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_act_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(rot_dims, embed_dims[2])
        self.rot_dims = rot_dims
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., :3])
        if box_3d.shape[-1] == 3:
            return pos_feat
        size_feat = self.size_fc(box_3d[..., 3:6])
        yaw_feat = self.yaw_fc(box_3d[..., 6 : 6 + self.rot_dims])
        output = pos_feat + size_feat + yaw_feat
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = torch.tensor(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(
                self.embed_dims, num_learnable_pts * 3
            )

    def forward(
        self,
        anchor,
        instance_feature=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, 3:6].exp()
        key_points = self.fix_scale.to(anchor) * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        key_points = rotation_3d_in_euler(
            key_points.flatten(0, 1),
            anchor[..., 6:9].flatten(0, 1),
        ).unflatten(0, (bs, num_anchor))
        key_points = key_points + anchor[..., None, :3]
        return key_points
