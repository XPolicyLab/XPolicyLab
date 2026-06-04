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

import torch
from torch import nn

from robo_orchard_lab.models.bip3d.utils import deformable_format
from robo_orchard_lab.models.sem_modules.layers import (
    ScalarEmbedder,
    linear_act_ln,
)
from robo_orchard_lab.utils import as_sequence, build


class SEMActionDecoder(nn.Module):
    """Spatial Enhanced Manipulation (SEM) Action Decoder.

    Decoder implementation from the paper https://arxiv.org/abs/2505.16196
    """

    def __init__(
        self,
        img_cross_attn,
        norm_layer,
        ffn,
        head,
        training_noise_scheduler=None,
        test_noise_scheduler=None,
        num_inference_timesteps=10,
        joint_self_attn=None,
        temp_cross_attn=None,
        robot_encoder=None,
        text_cross_attn=None,
        timestep_norm_layer=None,
        feature_level=1,
        state_dims=8,
        embed_dims=256,
        pred_steps=30,
        operation_order=None,
        num_decoder=6,
        act_cfg=None,
        num_test_traj=1,
        chunk_size=8,
        force_kinematics=False,
        state_loss_weights=None,
        fk_loss_weight=None,
        pre_norm=True,
    ):
        super().__init__()
        self.feature_level = as_sequence(feature_level)
        self.embed_dims = embed_dims
        self.pred_steps = pred_steps
        self.chunk_size = chunk_size
        self.num_test_traj = num_test_traj
        self.state_loss_weights = state_loss_weights
        self.fk_loss_weight = fk_loss_weight
        self.force_kinematics = force_kinematics
        self.pre_norm = pre_norm

        self.robot_encoder = build(robot_encoder)
        if operation_order is None:
            operation_order = [
                "joint_self_attn",
                "t_norm",
                "temp_cross_attn",
                "norm",
                "img_cross_attn",
                "norm",
                "text_cross_attn",
                "norm",
                "ffn",
                "norm",
            ] * num_decoder
        self.operation_order = operation_order

        self.op_config_map = {
            "joint_self_attn": joint_self_attn,
            "norm": norm_layer,
            "t_norm": timestep_norm_layer,
            "ffn": ffn,
            "text_cross_attn": text_cross_attn,
            "img_cross_attn": img_cross_attn,
            "temp_cross_attn": temp_cross_attn,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        self.input_layers = nn.Sequential(
            nn.Linear(chunk_size * state_dims, embed_dims),
            *linear_act_ln(embed_dims, 2, 2, act_cfg=act_cfg),
        )
        self.head = build(head)

        self.training_noise_scheduler = build(training_noise_scheduler)
        self.test_noise_scheduler = build(test_noise_scheduler)
        self.prediction_type = (
            self.training_noise_scheduler.config.prediction_type
        )
        self.num_train_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )
        self.num_inference_timesteps = num_inference_timesteps
        self.t_embed = ScalarEmbedder(
            timestep_norm_layer["condition_dims"], 256
        )

    def format_img_feature_maps(self, feature_maps):
        if isinstance(feature_maps, (list, tuple)):
            feature_maps = [feature_maps[i] for i in self.feature_level]
            img_feature = deformable_format(feature_maps)[0].flatten(1, 2)
        else:
            img_feature = feature_maps
        return img_feature

    def apply_scale_shift(
        self, robot_state, joint_scale_shift=None, inverse=False
    ):
        if joint_scale_shift is None:
            return robot_state
        scale = joint_scale_shift[:, None, :, 0:1]
        shift = joint_scale_shift[:, None, :, 1:2]
        if not inverse:
            robot_state = torch.cat(
                [(robot_state[..., :1] - shift) / scale, robot_state[..., 1:]],
                dim=-1,
            )
        else:
            robot_state = torch.cat(
                [robot_state[..., :1] * scale + shift, robot_state[..., 1:]],
                dim=-1,
            )
        return robot_state

    def forward_kinematics(self, joint_state, inputs):
        if joint_state.shape[-1] == 1:
            joint_state = joint_state.squeeze(-1)
        robot_state = []
        kinematics = inputs["kinematics"]
        embodiedment_mat = inputs.get(
            "embodiedment_mat", [None] * len(kinematics)
        )
        if len(kinematics) <= 1 or (
            all(x == kinematics[0] for x in kinematics[1:])
            and (
                embodiedment_mat[0] is None
                or (embodiedment_mat[0] - embodiedment_mat[1:] == 0).all()
            )
        ):
            robot_state = kinematics[0].joint_state_to_robot_state(
                joint_state, embodiedment_mat[0]
            )
        else:
            for i in range(len(joint_state)):
                robot_state.append(
                    inputs["kinematics"][i].joint_state_to_robot_state(
                        joint_state[i], embodiedment_mat[i]
                    )
                )
            robot_state = torch.stack(robot_state)
        return robot_state

    def recompute(self, robot_state, inputs):
        joint_state = self.apply_scale_shift(
            robot_state[..., :1],
            inputs.get("joint_scale_shift"),
            inverse=True,
        )
        robot_state = torch.cat(
            [
                robot_state[..., :1],
                self.forward_kinematics(joint_state, inputs)[..., 1:],
            ],
            dim=-1,
        )
        return robot_state

    def forward(self, feature_maps, inputs, text_dict=None, **kwargs):
        img_feature = self.format_img_feature_maps(feature_maps)

        if "hist_robot_state" not in inputs:
            hist_robot_state = self.joint_state_to_robot_state(
                inputs["hist_joint_state"], inputs
            )
        else:
            hist_robot_state = inputs["hist_robot_state"]

        joint_scale_shift = inputs.get("joint_scale_shift")
        hist_robot_state = self.apply_scale_shift(
            hist_robot_state, joint_scale_shift
        )
        bs, hist_steps, num_joint, state_dims = hist_robot_state.shape

        if "joint_relative_pos" in inputs:
            joint_relative_pos = inputs["joint_relative_pos"]
        else:
            joint_relative_pos = torch.stack(
                [k.joint_relative_pos for k in inputs["kinematics"]]
            )
        joint_relative_pos = joint_relative_pos.to(hist_robot_state)

        if self.robot_encoder is not None:
            robot_feature = self.robot_encoder(
                hist_robot_state, joint_relative_pos
            )
        else:
            robot_feature = None

        if self.training:
            pred_robot_state = self.apply_scale_shift(
                inputs["pred_robot_state"], joint_scale_shift
            )
            pred_steps = pred_robot_state.shape[1]
            noise = torch.randn([bs, pred_steps, num_joint, 1]).to(img_feature)
            timesteps = torch.randint(
                0, self.num_train_timesteps, (bs,), device=img_feature.device
            ).long()
            noisy_action = self.training_noise_scheduler.add_noise(
                pred_robot_state[..., :1], noise, timesteps
            )
            noisy_action = self.recompute(noisy_action, inputs)
            pred = self.forward_layers(
                noisy_action,
                img_feature,
                text_dict,
                robot_feature,
                timesteps,
                joint_relative_pos,
            )
            if self.prediction_type == "epsilon":
                target = torch.cat([noise, pred_robot_state[..., 1:]], dim=-1)
            elif self.prediction_type == "sample":
                target = pred_robot_state
            else:
                raise ValueError("Unsupported prediction type")
            return {"pred": pred, "target": target, "timesteps": timesteps}
        else:
            pred_actions = []
            for _ in range(self.num_test_traj):
                # if i == 0:
                #     noisy_action = torch.zeros(
                #         [bs, self.pred_steps, num_joint, 1],
                #     ).to(img_feature)
                # else:
                #     noisy_action = torch.randn(
                #         [bs, self.pred_steps, num_joint, 1],
                #     ).to(img_feature)
                noisy_action = torch.randn(
                    [bs, self.pred_steps, num_joint, 1],
                ).to(img_feature)
                self.test_noise_scheduler.set_timesteps(
                    self.num_inference_timesteps,
                    device=img_feature.device,
                )
                for t in self.test_noise_scheduler.timesteps:
                    noisy_action = self.recompute(noisy_action, inputs)
                    pred = self.forward_layers(
                        noisy_action,
                        img_feature,
                        text_dict,
                        robot_feature,
                        t.to(device=noisy_action.device).tile(bs),
                        joint_relative_pos,
                    )
                    noisy_action = self.test_noise_scheduler.step(
                        pred[..., :1], t, noisy_action[..., :1]
                    ).prev_sample
                    noisy_action = torch.cat(
                        [noisy_action, pred[..., 1:]], dim=-1
                    )
                pred_actions.append(noisy_action)
            return {"pred_actions": pred_actions}

    def forward_layers(
        self,
        noisy_action,
        img_feature,
        text_dict=None,
        robot_feature=None,
        timesteps=None,
        joint_relative_pos=None,
    ):
        bs, pred_steps, num_joint, state_dims = noisy_action.shape
        num_chunk = pred_steps // self.chunk_size
        noisy_action = noisy_action.permute(0, 2, 1, 3)
        noisy_action = noisy_action.reshape(
            bs, num_joint * num_chunk, self.chunk_size * state_dims
        )

        t_embed = self.t_embed(timesteps)
        x = self.input_layers(noisy_action)

        joint_relative_pos = joint_relative_pos.tile(num_chunk, 1, 1)

        if robot_feature is not None:
            num_hist_chunk = robot_feature.shape[2]
            robot_feature = robot_feature.flatten(0, 1)
            # bs*num_joint, num_hist_chunk, c
        else:
            num_hist_chunk = 0

        if "temp_cross_attn" in self.operation_order:
            temp_attn_mask = ~torch.tril(
                torch.ones(
                    num_chunk,
                    num_hist_chunk + num_chunk,
                    dtype=torch.bool,
                    device=x.device,
                ),
                num_hist_chunk,
            )
            temp_query_pos = (
                torch.arange(num_chunk)[None].tile(bs * num_joint, 1).to(x)
                + num_hist_chunk
            )
            temp_key_pos = (
                torch.arange(num_hist_chunk + num_chunk)
                .tile(bs * num_joint, 1)
                .to(x)
            )

        if text_dict is not None and "text_cross_attn" in self.operation_order:
            text_feature = text_dict["embedded"]
            num_text_token = text_feature.shape[1]
            tca_query_pos = torch.arange(num_chunk).to(x)[None, None]
            tca_query_pos = tca_query_pos.tile(bs, num_joint, 1).flatten(1, 2)
            tca_query_pos += num_text_token
            tca_key_pos = torch.arange(num_text_token).to(x)[None].tile(bs, 1)

        if "img_cross_attn" in self.operation_order:
            ica_query_pos = torch.arange(num_chunk).to(x)[None, None]
            ica_query_pos = ica_query_pos.tile(bs, num_joint, 1).flatten(1, 2)
            ica_query_pos += 1
            ica_key_pos = None

        if self.pre_norm:
            identity = x
        else:
            identity = None
        for i, (op, layer) in enumerate(
            zip(self.operation_order, self.layers, strict=False)
        ):
            if op == "joint_self_attn":
                x = (
                    x.reshape(bs, num_joint, num_chunk, -1)
                    .permute(0, 2, 1, 3)
                    .flatten(0, 1)
                )
                if identity is not None:
                    _identity = (
                        identity.reshape(bs, num_joint, num_chunk, -1)
                        .permute(0, 2, 1, 3)
                        .flatten(0, 1)
                    )
                else:
                    _identity = None
                x = layer(
                    query=x,
                    key=x,
                    value=x,
                    query_pos=joint_relative_pos,
                    identity=_identity,
                )
                x = (
                    x.reshape(bs, num_chunk, num_joint, -1)
                    .permute(0, 2, 1, 3)
                    .flatten(1, 2)
                )
            elif op == "temp_cross_attn":
                x = x.reshape(bs * num_joint, num_chunk, -1)
                if robot_feature is not None:
                    kv = torch.cat([robot_feature, x], dim=1)
                else:
                    kv = x
                if identity is not None:
                    _identity = identity.reshape(bs * num_joint, num_chunk, -1)
                else:
                    _identity = None
                x = layer(
                    query=x,
                    key=kv,
                    value=kv,
                    query_pos=temp_query_pos,
                    key_pos=temp_key_pos,
                    attn_mask=temp_attn_mask,
                    identity=_identity,
                )
                x = x.reshape(bs, num_joint * num_chunk, -1)
            elif op == "text_cross_attn":
                x = layer(
                    query=x,
                    key=text_feature,
                    value=text_feature,
                    key_padding_mask=~text_dict["text_token_mask"],
                    query_pos=tca_query_pos,
                    key_pos=tca_key_pos,
                    identity=identity,
                )
            elif op == "img_cross_attn":
                x = layer(
                    query=x,
                    key=img_feature,
                    value=img_feature,
                    query_pos=ica_query_pos,
                    key_pos=ica_key_pos,
                    identity=identity,
                )
            elif op == "ffn":
                x = layer(x, identity=identity)
            elif op == "norm":
                if self.pre_norm:
                    identity = x
                x = layer(x)
            elif op == "t_norm":
                if self.pre_norm:
                    identity = x
                x, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer(x, t_embed)
            elif op == "gate_msa":
                x = gate_msa * x
            elif op == "gate_mlp":
                x = gate_mlp * x
            elif op == "scale_shift":
                x = x * (1 + scale_mlp) + shift_mlp
            elif self.layers[i] is None:
                continue

        pred = self.head(x.reshape(bs, num_joint, num_chunk, -1))
        pred = pred.permute(0, 2, 1, 3)
        return pred

    def loss(self, model_outs, inputs, **kwargs):
        pred = model_outs["pred"]
        target = model_outs["target"]
        pred_mask = inputs.get("pred_mask")
        output = {}
        loss = self._loss_func(
            pred, target, pred_mask, self.state_loss_weights
        )
        output["loss_noise_mse"] = loss
        if self.fk_loss_weight is not None:
            fk_pred = self.recompute(pred, inputs)
            fk_loss = self._loss_func(
                fk_pred, target, pred_mask, self.fk_loss_weight
            )
            output["loss_fk_mse"] = fk_loss
        return output

    def _loss_func(self, pred, target, pred_mask, weight=None):
        error = torch.square(pred - target)
        if pred_mask is not None:
            error = error[pred_mask]
            if error.shape[0] == 0:
                return pred.sum() * 0
        if weight is not None:
            error = error * error.new_tensor(weight)
        loss = error.mean()
        return loss

    def post_process(self, model_outs, inputs, **kwargs):
        bs = model_outs["pred_actions"][0].shape[0]
        results = []
        for i in range(bs):
            pred_actions = torch.stack(
                [x[i] for x in model_outs["pred_actions"]]
            )
            if "joint_scale_shift" in inputs:
                pred_actions = self.apply_scale_shift(
                    pred_actions,
                    inputs["joint_scale_shift"][i][None],
                    inverse=True,
                )
            results.append(dict(pred_actions=pred_actions))
        return results
