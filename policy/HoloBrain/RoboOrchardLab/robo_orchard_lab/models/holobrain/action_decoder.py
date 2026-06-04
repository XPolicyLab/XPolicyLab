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
from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from robo_orchard_core.utils.config import Config
from torch import nn

from robo_orchard_lab.models.bip3d.utils import deformable_format
from robo_orchard_lab.models.holobrain.layers import (
    ScalarEmbedder,
    linear_act_ln,
)
from robo_orchard_lab.models.holobrain.utils import (
    apply_joint_mask,
    apply_scale_shift,
    recompute,
)
from robo_orchard_lab.models.mixin import TorchModuleCfgType_co
from robo_orchard_lab.utils import as_sequence, build
from robo_orchard_lab.utils.build import DelayInitDictType

logger = logging.getLogger(__name__)


MODULE_TYPE = TorchModuleCfgType_co | DelayInitDictType
VALID_NOISE_TYPES = (
    "global_joint",
    "local_joint",
    "global_joint_global_pose",
    "global_joint_local_pose",
    "local_joint_global_pose",
    "local_joint_local_pose",
)
VALID_PREDICTION_TYPES = (
    "absolute_joint_absolute_pose",
    "absolute_joint_relative_pose",
    "relative_joint_absolute_pose",
    "relative_joint_relative_pose",
)


class HoloBrainDecoderTransformerConfig(Config):
    """Transformer config of HoloBrain decoder.

    Args:
        img_cross_attn (MODULE_TYPE):
            Cross attention module between action and image features.
        norm_layer (MODULE_TYPE): Normalization layer.
        ffn (MODULE_TYPE): Feed-forward network.
        operation_order (Optional[List[str]]): Sequence of operations
            (attn/FFN/norm) in transformer decoder.
        joint_self_attn (Optional[MODULE_TYPE]): Joint dimension self-attention
            module.
        temp_cross_attn (Optional[MODULE_TYPE]): Causal temporal attention
            module across time steps.
        text_cross_attn (Optional[MODULE_TYPE]): Cross attention between action
            and text features.
        temp_joint_attn (Optional[MODULE_TYPE]): Joint-temporal attention
            module (causal in temporal dimension).
        timestep_norm_layer (Optional[MODULE_TYPE]): Normalization layer for
            encoding diffusion timesteps.
        pre_norm (bool): Use pre-normalization or post-normalization.
    """

    img_cross_attn: MODULE_TYPE
    norm_layer: MODULE_TYPE
    ffn: MODULE_TYPE
    operation_order: List[Union[str, None]]
    joint_self_attn: Optional[MODULE_TYPE] = None
    temp_cross_attn: Optional[MODULE_TYPE] = None
    text_cross_attn: Optional[MODULE_TYPE] = None
    temp_joint_attn: Optional[MODULE_TYPE] = None
    timestep_norm_layer: Optional[MODULE_TYPE] = None
    pre_norm: bool = True

    @property
    def op_config_map(self) -> dict:
        return {
            "img_cross_attn": self.img_cross_attn,
            "norm": self.norm_layer,
            "ffn": self.ffn,
            "joint_self_attn": self.joint_self_attn,
            "temp_joint_attn": self.temp_joint_attn,
            "t_norm": self.timestep_norm_layer,
            "text_cross_attn": self.text_cross_attn,
            "temp_cross_attn": self.temp_cross_attn,
        }


class HoloBrainDecoderBaseConfig(Config):
    """Base config of HoloBrain decoder.

    Args:
        training_noise_scheduler (Optional[Any]): Diffusion scheduler for
            training phase (supports "sample" prediction only).
        test_noise_scheduler (Optional[Any]): Diffusion scheduler for inference
            phase (supports "sample" prediction only).
        num_inference_timesteps (int): Num of denoising steps during inference.
        feature_level (Union[int, List[int]]): Image feature level(s) to use.
        state_dims (int): Dimension size of robot state. Defaults to 8, refer
            to [a, x, y, z, qw, qx, qy, qz].
        embed_dims (int): Embedding dimension size. Defaults to 256.
        pred_steps (int): Number of future steps to predict. Defaults to 30.
        act_cfg (Optional[MODULE_TYPE]): Activation function configuration.
        num_test_traj (int): Number of trajectories to sample during inference.
        chunk_size (int): Step size for chunking prediction horizon.
        force_kinematics (bool): Apply forward kinematics before state input.
        with_mobile (bool): Enable trajectory prediction branch.
        mobile_traj_state_dims (int): Trajectory state dimensions
            (default 2 for [x,y]).
        use_joint_mask (bool): Mask joint angle information. Defaults to False.
        noise_type (str): Noise sampling space identifier.
        pred_scaled_joint (bool): Predict joint angles scaled to [-1,1].
        prediction_type (str): Prediction space specification.
    """

    training_noise_scheduler: Optional[MODULE_TYPE] = None
    test_noise_scheduler: Optional[MODULE_TYPE] = None
    num_inference_timesteps: int = 10
    feature_level: int | List[int] = 1
    state_dims: int = 8
    embed_dims: int = 256
    pred_steps: int = 64
    act_cfg: Optional[MODULE_TYPE] = None
    num_test_traj: int = 1
    chunk_size: int = 8
    force_kinematics: bool = False
    with_mobile: bool = False
    mobile_traj_state_dims: int = 2
    use_joint_mask: bool = False
    noise_type: Literal[VALID_NOISE_TYPES] = "global_joint"
    pred_scaled_joint: bool = True
    prediction_type: Literal[VALID_PREDICTION_TYPES] = (
        "absolute_joint_absolute_pose"
    )


class HoloBrainTrainingConfig(Config):
    """HoloBrain training config.

    Args:
        loss (Optional[MODULE_TYPE]): Loss module.
        temporal_attn_drop (Optional[float]): Ratio of random dropout for
            current robot state during training.
        num_parallel_training_sample (Optional[int]): Number of trajectories
            sampled per training example.
        teacher_forcing_rate (Optional[float]): Probability of using teacher
            forcing during training.
        teacher_forcing_mean_steps (Optional[int]): Average steps for teacher
            forcing.
    """

    loss: Optional[MODULE_TYPE] = None
    temporal_attn_drop: Optional[float] = None
    num_parallel_training_sample: Optional[int] = None
    teacher_forcing_rate: Optional[float] = None
    teacher_forcing_mean_steps: Optional[int] = None


class HoloBrainActionDecoder(nn.Module):
    """Spatial Enhanced Manipulation (HoloBrain) Action Decoder.

    Decoder implementation from the paper https://arxiv.org/abs/2505.16196

    Args:
        transformer_cfg (HoloBrainDecoderTransformerConfig): Config of
            transformer layers.
        head (MODULE_TYPE): Output head for action prediction.
        base_cfg (HoloBrainDecoderBaseConfig): base config of holobrain.
        training_cfg (HoloBrainTrainingConfig): training config of holobrain
            decoder.
        robot_encoder (Optional[MODULE_TYPE]): Encoder for processing robot
            state inputs.
        mobile_head (Optional[MODULE_TYPE]): Head for trajectory prediction.
        async_inference_plugin (Optional[MODULE_TYPE]): Plugin for aggregating
            predicted/remaining actions for async inference.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        transformer_cfg: HoloBrainDecoderTransformerConfig,
        head: MODULE_TYPE,
        base_cfg: HoloBrainDecoderBaseConfig,
        training_cfg: Optional[HoloBrainTrainingConfig] = None,
        robot_encoder: Optional[MODULE_TYPE] = None,
        mobile_head: Optional[MODULE_TYPE] = None,
        async_inference_plugin: Optional[MODULE_TYPE] = None,
        **kwargs,
    ):
        super().__init__()
        if len(kwargs) != 0:
            logger.warning(f"Get unexpected arguments: {kwargs}")
        transformer_cfg = HoloBrainDecoderTransformerConfig.model_validate(
            transformer_cfg
        )
        base_cfg = HoloBrainDecoderBaseConfig.model_validate(base_cfg)

        # base config
        self.training_noise_scheduler = build(
            base_cfg.training_noise_scheduler
        )
        self.test_noise_scheduler = build(base_cfg.test_noise_scheduler)
        assert self.training_noise_scheduler.config.prediction_type == "sample"
        assert self.test_noise_scheduler.config.prediction_type == "sample"
        self.num_train_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )
        self.num_inference_timesteps = base_cfg.num_inference_timesteps
        self.feature_level = as_sequence(base_cfg.feature_level)
        self.pred_steps = base_cfg.pred_steps
        self.chunk_size = base_cfg.chunk_size
        self.num_test_traj = base_cfg.num_test_traj
        self.force_kinematics = base_cfg.force_kinematics
        self.with_mobile = base_cfg.with_mobile
        self.state_dims = base_cfg.state_dims
        self.embed_dims = base_cfg.embed_dims
        self.mobile_traj_state_dims = base_cfg.mobile_traj_state_dims
        self.use_joint_mask = base_cfg.use_joint_mask
        self.noise_type = base_cfg.noise_type
        self.pred_scaled_joint = base_cfg.pred_scaled_joint
        self.prediction_type = base_cfg.prediction_type

        # training config
        if training_cfg is not None:
            training_cfg = HoloBrainTrainingConfig.model_validate(training_cfg)
            self.training_cfg = training_cfg
            self.loss = build(training_cfg.loss)
            if (
                training_cfg.teacher_forcing_rate is not None
                and training_cfg.teacher_forcing_mean_steps is None
            ):
                logger.warning(
                    f"Use default teacher_forcing_mean_steps: "
                    f"{self.pred_steps // 4}"
                )
                training_cfg.teacher_forcing_mean_steps = self.pred_steps // 4

        # build modules
        self.robot_encoder = build(robot_encoder)
        self.operation_order = transformer_cfg.operation_order
        self.pre_norm = transformer_cfg.pre_norm
        self.layers = nn.ModuleList(
            [
                build(transformer_cfg.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        self.input_layers = nn.Sequential(
            nn.Linear(self.chunk_size * self.state_dims, self.embed_dims),
            *linear_act_ln(self.embed_dims, 2, 2, act_cfg=base_cfg.act_cfg),
        )
        self.head = build(head)
        if self.with_mobile:
            self.mobile_input_layers = nn.Sequential(
                nn.Linear(
                    self.chunk_size * self.mobile_traj_state_dims,
                    self.embed_dims,
                ),
                *linear_act_ln(
                    self.embed_dims, 2, 2, act_cfg=base_cfg.act_cfg
                ),
            )
            self.mobile_head = build(mobile_head)
        self.t_embed = ScalarEmbedder(
            transformer_cfg.timestep_norm_layer["condition_dims"], 256
        )
        self.async_inference_plugin = build(async_inference_plugin)

    def format_img_feature_maps(
        self, feature_maps: Union[list[torch.Tensor], torch.Tensor]
    ):
        """Formats multi-scale image feature maps.

        Args:
            feature_maps (list[torch.Tensor] | torch.Tensor): Multi-scale
                feature as list of [bs, c, h, w] tensors, or single tensor.

        Returns:
            torch.Tensor: Flattened features [bs, n, c] if input was list,
                otherwise original tensor.
        """
        if isinstance(feature_maps, (list, tuple)):
            feature_maps = [feature_maps[i] for i in self.feature_level]
            img_feature = deformable_format(feature_maps)[0].flatten(1, 2)
        else:
            img_feature = feature_maps
        return img_feature

    def sample_noise(
        self,
        noise_shape: List[int],
        hist_robot_state: torch.Tensor,
        noise_type: Literal[VALID_NOISE_TYPES],
    ):
        """Samples noise based on noise_shape and noise_type.

        Args:
            noise_shape (tuple[int, ...]): Shape of the output noise tensor
                (e.g., [bs, num_steps, num_joint, state_dims]).
            hist_robot_state (torch.Tensor): Historical robot state tensor,
                used when get "local" noise_type.
            noise_type (VALID_NOISE_TYPE): Type of noise to sample.

        Returns:
            torch.Tensor: Sampled noise tensor of shape noise_shape.
        """
        if not noise_type.endswith("pose"):
            noise = torch.randn([*noise_shape[:-1], 1])
        else:
            noise = torch.randn(noise_shape)
        noise = noise.to(hist_robot_state)
        if noise_type.startswith("local_joint"):
            noise[..., :1] = noise[..., :1] + hist_robot_state[:, -1:, :, :1]
        if noise_type.endswith("local_pose"):
            noise[..., 1:] = noise[..., 1:] + hist_robot_state[:, -1:, :, 1:]
        return noise

    def get_prediction(
        self,
        model_pred: torch.Tensor,
        hist_robot_state: torch.Tensor,
        joint_scale_shift: torch.Tensor,
        joint_mask: torch.Tensor,
    ):
        """Adjusts model_pred to absolute result based on prediction_type.

        Args:
            model_pred (torch.Tensor): Model output, shape [bs, num_steps,
                num_joint, c] (channel 0 = joint position).
            hist_robot_state (torch.Tensor): Historical robot state, shape
                [bs, num_hist_steps, num_joint, c].
            joint_scale_shift (torch.Tensor): Scale/shift params for
                joint position normalization, shape [bs, num_joint].
            joint_mask (torch.Tensor): Boolean mask for joints, shape
                [bs, num_joint].

        Returns:
            torch.Tensor: Absolute prediction tensor (normalized joint
                position), shape same as model_pred.
        """

        origin_model_pred = model_pred.clone()
        if not self.pred_scaled_joint:
            model_pred = apply_scale_shift(
                model_pred, joint_scale_shift, scale_only=True
            )

        if self.prediction_type == "absolute_joint_absolute_pose":
            pred = model_pred
        elif self.prediction_type == "absolute_joint_relative_pose":
            pred = torch.cat(
                [
                    model_pred[..., :1],
                    model_pred[..., 1:] + hist_robot_state[:, -1:, :, 1:],
                ],
                dim=-1,
            )
        elif self.prediction_type == "relative_joint_absolute_pose":
            pred = torch.cat(
                [
                    model_pred[..., :1] + hist_robot_state[:, -1:, :, :1],
                    model_pred[..., 1:],
                ],
                dim=-1,
            )
        elif self.prediction_type == "relative_joint_relative_pose":
            pred = model_pred + hist_robot_state

        if joint_mask is not None and self.prediction_type.startswith(
            "relative"
        ):
            pred_joint_state = torch.where(
                joint_mask[:, None, :, None],
                pred[..., :1],
                origin_model_pred[..., :1],
            )
            pred = torch.cat([pred_joint_state, pred[..., 1:]], dim=-1)
        return pred

    def _repeat(self, n: int, *inputs):
        """Repeats each input tensor along the batch dimension.

        Args:
            n (int): Number of repetitions along batch dimension.
            *inputs (torch.Tensor): Tensors to repeat, each of shape [bs, ...].

        Returns:
            tuple[torch.Tensor, ...]: Repeated tensors, each of shape
                [bs * n, ...].
        """

        output = []
        for x in inputs:
            if x is None:
                output.append(None)
            else:
                output.append(x.repeat_interleave(n, dim=0))
        if len(output) == 0:
            return output[0]
        return output

    def forward(self, feature_maps, inputs, text_dict=None, **kwargs):
        """Forward pass of HoloBrain decoder for trajectory prediction.

        Encodes historical robot state from inputs via self.robot_encoder, then
        processes through diffusion-based denoising network.
        Branches based on training mode:

        - Training: Applies single noise step, executes forward_layers once to
          predict  num_parallel_training_sample trajectories per sample.
        - Inference: Iteratively refines predictions over
          num_inference_timesteps steps, outputting num_test_traj
          trajectories per sample.
        """
        inputs = inputs.copy()
        text_dict = text_dict.copy() if text_dict is not None else {}
        img_feature = self.format_img_feature_maps(feature_maps)

        if "hist_robot_state" not in inputs:
            hist_robot_state = self.joint_state_to_robot_state(
                inputs["hist_joint_state"], inputs
            )
        else:
            hist_robot_state = inputs["hist_robot_state"]

        joint_scale_shift = inputs.get("joint_scale_shift")
        hist_robot_state = apply_scale_shift(
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

        if not self.use_joint_mask:
            joint_mask = None
        else:
            joint_mask = inputs.get("joint_mask")
            if joint_mask is None:
                logger.warning(f"Miss `joint_mask`, uuid:{inputs.get('uuid')}")

        if self.robot_encoder is not None:
            robot_feature = self.robot_encoder(
                hist_robot_state, joint_relative_pos, joint_mask
            )
        else:
            robot_feature = None

        if "noise_type" in inputs:
            noise_type = inputs["noise_type"][0]
            assert noise_type in VALID_NOISE_TYPES, noise_type
        else:
            noise_type = self.noise_type

        if self.training:
            pred_robot_state = apply_scale_shift(
                inputs["pred_robot_state"], joint_scale_shift
            )
            pred_steps = pred_robot_state.shape[1]
            timesteps = torch.randint(
                0, self.num_train_timesteps, (bs,), device=img_feature.device
            ).long()

            if (
                self.training_cfg.num_parallel_training_sample is not None
                and self.training_cfg.num_parallel_training_sample > 1
            ):
                bs = self.training_cfg.num_parallel_training_sample * bs
                (
                    pred_robot_state,
                    hist_robot_state,
                    img_feature,
                    robot_feature,
                    joint_relative_pos,
                    timesteps,
                    joint_scale_shift,
                    joint_mask,
                    inputs["mobile_traj"],
                    inputs["embodiedment_mat"],
                    text_dict["embedded"],
                    text_dict["text_token_mask"],
                ) = self._repeat(
                    self.training_cfg.num_parallel_training_sample,
                    pred_robot_state,
                    hist_robot_state,
                    img_feature,
                    robot_feature,
                    joint_relative_pos,
                    timesteps,
                    joint_scale_shift,
                    joint_mask,
                    inputs.get("mobile_traj"),
                    inputs["embodiedment_mat"],
                    text_dict.get("embedded"),
                    text_dict.get("text_token_mask"),
                )
                inputs["joint_scale_shift"] = joint_scale_shift

            noise = self.sample_noise(
                [bs, pred_steps, num_joint, state_dims],
                hist_robot_state,
                noise_type,
            )

            if not noise_type.endswith("pose"):
                noisy_action = self.training_noise_scheduler.add_noise(
                    pred_robot_state[..., :1], noise, timesteps
                )
                noisy_action = recompute(noisy_action, inputs)
            else:
                noisy_action = self.training_noise_scheduler.add_noise(
                    pred_robot_state, noise, timesteps
                )

            if (
                self.training_cfg.teacher_forcing_rate is not None
                and self.training_cfg.teacher_forcing_rate > 0
            ):
                mask = torch.logical_and(
                    torch.poisson(
                        noisy_action.new_full(
                            [bs, 1],
                            self.training_cfg.teacher_forcing_mean_steps,
                        )
                    )
                    > torch.arange(pred_steps).to(noisy_action),
                    torch.rand(bs)[:, None].to(noisy_action)
                    < self.training_cfg.teacher_forcing_rate,
                )[..., None, None]
                noisy_action = torch.where(
                    mask, pred_robot_state, noisy_action
                )

            if self.with_mobile:
                target_mobile_traj = inputs.get("mobile_traj")
                noisy_mobile_traj = torch.randn(
                    [bs, pred_steps, self.mobile_traj_state_dims]
                ).to(img_feature)
                if target_mobile_traj is not None:
                    noisy_mobile_traj = (
                        self.training_noise_scheduler.add_noise(
                            target_mobile_traj, noisy_mobile_traj, timesteps
                        )
                    )
            else:
                noisy_mobile_traj = None
            pred, pred_mobile_traj = self.forward_layers(
                noisy_action,
                img_feature,
                text_dict,
                robot_feature,
                timesteps,
                joint_relative_pos,
                noisy_mobile_traj,
                joint_mask,
            )
            pred = self.get_prediction(
                pred, hist_robot_state, joint_scale_shift, joint_mask
            )
            return {
                "pred": pred,
                "target": pred_robot_state,
                "pred_mobile_traj": pred_mobile_traj,
                "target_mobile_traj": inputs.get("mobile_traj"),
                "timesteps": timesteps,
                "num_parallel": self.training_cfg.num_parallel_training_sample,
            }
        else:  # inference
            if (
                self.async_inference_plugin is not None
                and "remaining_actions" in inputs
                and "delay_horizon" in inputs
            ):
                remaining_actions = (
                    inputs["remaining_actions"][0]
                    .to(img_feature)
                    .unsqueeze(-1)
                )
                remaining_actions = apply_scale_shift(
                    remaining_actions,
                    joint_scale_shift,
                )
                delay_horizon = inputs["delay_horizon"][0]
            else:
                remaining_actions = None

            if self.num_test_traj is not None and self.num_test_traj > 1:
                bs = self.num_test_traj * bs
                (
                    img_feature,
                    robot_feature,
                    joint_relative_pos,
                    hist_robot_state,
                    joint_scale_shift,
                    joint_mask,
                    remaining_actions,
                    inputs["embodiedment_mat"],
                    text_dict["embedded"],
                    text_dict["text_token_mask"],
                ) = self._repeat(
                    self.num_test_traj,
                    img_feature,
                    robot_feature,
                    joint_relative_pos,
                    hist_robot_state,
                    joint_scale_shift,
                    joint_mask,
                    remaining_actions,
                    inputs["embodiedment_mat"],
                    text_dict.get("embedded"),
                    text_dict.get("text_token_mask"),
                )
                inputs["joint_scale_shift"] = joint_scale_shift

            noisy_action = self.sample_noise(
                [bs, self.pred_steps, num_joint, state_dims],
                hist_robot_state,
                noise_type,
            )
            if self.with_mobile:
                noisy_mobile_traj = torch.randn(
                    [bs, self.pred_steps, self.mobile_traj_state_dims]
                ).to(img_feature)
            else:
                noisy_mobile_traj = None
            self.test_noise_scheduler.set_timesteps(
                self.num_inference_timesteps,
                device=img_feature.device,
            )

            for t in self.test_noise_scheduler.timesteps:
                if not noise_type.endswith("pose"):
                    noisy_action = recompute(noisy_action, inputs)
                pred, pred_mobile_traj = self.forward_layers(
                    noisy_action,
                    img_feature,
                    text_dict,
                    robot_feature,
                    t.to(device=noisy_action.device).tile(bs),
                    joint_relative_pos,
                    noisy_mobile_traj,
                    joint_mask,
                )
                pred = self.get_prediction(
                    pred, hist_robot_state, joint_scale_shift, joint_mask
                )
                if remaining_actions is not None:
                    pred = self.async_inference_plugin(
                        pred,
                        remaining_actions,
                        delay_horizon,
                    )
                if not noise_type.endswith("pose"):
                    noisy_action = self.test_noise_scheduler.step(
                        pred[..., :1], t, noisy_action[..., :1]
                    ).prev_sample
                    noisy_action = torch.cat(
                        [noisy_action, pred[..., 1:]], dim=-1
                    )
                else:
                    noisy_action = self.test_noise_scheduler.step(
                        pred, t, noisy_action
                    ).prev_sample
                if self.with_mobile:
                    noisy_mobile_traj = self.test_noise_scheduler.step(
                        pred_mobile_traj, t, noisy_mobile_traj
                    )

            pred_actions = noisy_action
            pred_mobile_trajs = noisy_mobile_traj
            if self.num_test_traj is not None and self.num_test_traj > 1:
                inputs["joint_scale_shift"] = joint_scale_shift.unflatten(
                    0, (-1, self.num_test_traj)
                )[:, 0]
                pred_actions = pred_actions.unflatten(
                    0, (-1, self.num_test_traj)
                )
                if self.with_mobile:
                    pred_mobile_trajs = pred_mobile_trajs.unflatten(
                        0, (-1, self.num_test_traj)
                    )
            else:  # only one trajectory for one sample
                pred_actions = pred_actions.unsqueeze(1)
                if self.with_mobile:
                    pred_mobile_trajs = pred_mobile_trajs.unsqueeze(1)

            return {
                "pred_actions": pred_actions,
                "pred_mobile_trajs": pred_mobile_trajs,
            }

    def forward_layers(
        self,
        noisy_action: torch.Tensor,
        img_feature,
        text_dict=None,
        robot_feature=None,
        timesteps=None,
        joint_relative_pos=None,
        noisy_mobile_traj=None,
        joint_mask=None,
    ):
        """Forward holobrain transformer decoder and head.

        Args:
            noisy_action (torch.Tensor): Noisy action [bs, steps, joints, c].
            img_feature (torch.Tensor): Flattened img feature [bs, n, c].
            text_dict (dict, optional): Dict with text embeddings [bs, m, c]
                and token_mask [bs, m].
            robot_feature (torch.Tensor, optional): Robot state features
                [bs, num_hist_chunk, c]. Defaults to None.
            timesteps (torch.Tensor, optional): Timestep indicators for
                diffusion, [bs].
            joint_relative_pos (torch.Tensor, optional): Relative joint
                positions [bs, joints, joints]. Defaults to None.
            noisy_mobile_traj (torch.Tensor, optional): Noisy mobile trajectory
                [bs, steps, mobile_traj_state_dims]. Defaults to None.
            joint_mask (torch.Tensor, optional): Boolean joint mask
                [bs, joints]. Defaults to None.

        Returns:
            pred (torch.Tensor): Predicted robot state
                [bs, steps, joints, state_dims].
            pred_mobile_traj (torch.Tensor): Predicted mobile trajectory
                [bs, steps, joints, mobile_traj_state_dims]. None
                when with_mobile is False
        """
        t_embed = self.t_embed(timesteps)

        bs, pred_steps, num_joint, state_dims = noisy_action.shape
        num_chunk = pred_steps // self.chunk_size
        noisy_action = noisy_action.permute(0, 2, 1, 3)

        if joint_mask is not None:
            noisy_action = apply_joint_mask(noisy_action, joint_mask)

        noisy_action = noisy_action.reshape(bs, num_joint, num_chunk, -1)
        x = self.input_layers(noisy_action)

        if self.with_mobile:
            noisy_mobile_traj = noisy_mobile_traj.reshape(bs, 1, num_chunk, -1)
            x_mobile = self.mobile_input_layers(noisy_mobile_traj)
            x = torch.cat([x, x_mobile], dim=1)
            joint_relative_pos = F.pad(joint_relative_pos, [0, 1, 0, 1])
            num_joint += 1

        x = x.reshape(bs, num_joint * num_chunk, -1)

        if robot_feature is not None:
            num_hist_chunk = robot_feature.shape[2]
            if robot_feature.shape[1] == num_joint - 1:
                robot_feature = torch.cat(
                    [robot_feature, torch.zeros_like(robot_feature[:, :1])],
                    dim=1,
                )
            robot_feature = robot_feature.flatten(0, 1)
            # bs*num_joint, num_hist_chunk, c
        else:
            num_hist_chunk = 0

        temp_attn_mask = ~torch.tril(
            torch.ones(
                num_chunk,
                num_hist_chunk + num_chunk,
                dtype=torch.bool,
                device=x.device,
            ),
            num_hist_chunk,
        )
        if self.training_cfg.temporal_attn_drop is not None and self.training:
            attn_drop = torch.rand(bs) < self.training_cfg.temporal_attn_drop
            attn_drop = attn_drop[:, None, None].to(x.device)
            temp_attn_mask = temp_attn_mask[None].repeat(bs, 1, 1)
            temp_attn_mask[..., :num_hist_chunk] = attn_drop

        if "temp_cross_attn" in self.operation_order:
            temp_query_pos = (
                torch.arange(num_chunk)[None].tile(bs * num_joint, 1).to(x)
                + num_hist_chunk
            )
            temp_key_pos = (
                torch.arange(num_hist_chunk + num_chunk)
                .tile(bs * num_joint, 1)
                .to(x)
            )

        if "text_cross_attn" in self.operation_order:
            text_feature = text_dict.get("embedded")
            assert text_feature is not None
            num_text_token = text_feature.shape[1]
            tca_query_pos = torch.arange(num_chunk).to(x)[None, None]
            tca_query_pos = tca_query_pos.tile(bs, num_joint, 1).flatten(1, 2)
            tca_query_pos += num_text_token
            tca_key_pos = torch.arange(num_text_token).to(x)[None].tile(bs, 1)
            text_key_padding_mask = text_dict.get("text_token_mask")
            if text_key_padding_mask is not None:
                text_key_padding_mask = ~text_key_padding_mask

        if "img_cross_attn" in self.operation_order:
            ica_query_pos = torch.arange(num_chunk).to(x)[None, None]
            ica_query_pos = ica_query_pos.tile(bs, num_joint, 1).flatten(1, 2)
            ica_query_pos += 1
            ica_key_pos = None

        if "temp_joint_attn" in self.operation_order:
            temp_query_pos_wojoint = (
                torch.arange(num_chunk)[None].tile(bs, 1).to(x)
                + num_hist_chunk
            )
            temp_key_pos_wojoint = (
                torch.arange(num_hist_chunk + num_chunk).tile(bs, 1).to(x)
            )
            joint_relative_pos_wochunk = joint_relative_pos

        joint_relative_pos = joint_relative_pos.tile(num_chunk, 1, 1)

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
            elif op == "temp_joint_attn":
                x = x.reshape(bs, num_joint, num_chunk, -1)
                kv = torch.cat(
                    [robot_feature.unflatten(0, (bs, num_joint)), x], dim=2
                )
                if identity is not None:
                    _identity = identity.reshape(bs, num_joint, num_chunk, -1)
                else:
                    _identity = None
                kwargs = dict(
                    query=x,
                    key=kv,
                    joint_distance=joint_relative_pos_wochunk,
                    temporal_pos_q=temp_query_pos_wojoint,
                    temporal_pos_k=temp_key_pos_wojoint,
                    temporal_attn_mask=temp_attn_mask,
                    identity=_identity,
                )
                x = layer(**kwargs)
                x = x.reshape(bs, num_joint * num_chunk, -1)
            elif op == "text_cross_attn":
                x = layer(
                    query=x,
                    key=text_feature,
                    value=text_feature,
                    key_padding_mask=text_key_padding_mask,
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

        x = x.reshape(bs, num_joint, num_chunk, -1)
        if self.with_mobile:
            x_mobile = x[:, -1:]
            x = x[:, :-1]
            pred_mobile_traj = self.mobile_head(x_mobile)[:, 0]
        else:
            pred_mobile_traj = None
        pred = self.head(x)
        pred = pred.permute(0, 2, 1, 3)
        return pred, pred_mobile_traj

    def post_process(self, model_outs, inputs, **kwargs):
        """Post-processes model inference outputs.

        Converts raw model outputs (dict) from inference mode into a list of
        dictionaries, each corresponding to a sample in the batch. Denormalize
        the joint position.

        Args:
            model_outs (dict): Model inference outputs (dict), containing
                raw predicted action and mobile trajectory.
            inputs (dict): Input data contains joint_scale_shift.

        Returns:
            list[dict]: Per-sample prediction results, length = batch size.
                Each dict contains:
                - pred_actions (torch.Tensor): Predicted robot action, shape
                [num_traj, num_steps, num_joint, state_dim].
                - pred_mobile_trajs (torch.Tensor): Predicted mobile
                trajectories, shape [num_traj, num_steps, num_joint,
                mobile_traj_state_dims].
        """

        bs = model_outs["pred_actions"].shape[0]
        results = []
        for i in range(bs):
            pred_actions = model_outs["pred_actions"][i]
            if "joint_scale_shift" in inputs:
                pred_actions = apply_scale_shift(
                    pred_actions,
                    inputs["joint_scale_shift"][i][None],
                    inverse=True,
                )
            results.append(dict(pred_actions=pred_actions))

            if model_outs.get("pred_mobile_trajs") is not None:
                results[-1]["pred_mobile_trajs"] = model_outs[
                    "pred_mobile_trajs"
                ][i]
        return results
