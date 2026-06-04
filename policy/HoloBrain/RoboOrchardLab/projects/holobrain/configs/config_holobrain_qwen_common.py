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


import os

# XPolicyLab integration: comma-separated dataset names override the
# default upstream training/deploy lists. Default behaviour unchanged
# when XPOLICY_HOLOBRAIN_DATASETS is not set.
_XP_DATASETS = os.environ.get("XPOLICY_HOLOBRAIN_DATASETS")
if _XP_DATASETS:
    _DATASETS = [d.strip() for d in _XP_DATASETS.split(",") if d.strip()]
else:
    _DATASETS = ["robotwin2_0", "robotwin2_0_ur5_wsg"]

_XP_BATCH_SIZE = int(os.environ.get("XPOLICY_HOLOBRAIN_BATCH_SIZE", 16))
_XP_NUM_WORKERS = int(os.environ.get("XPOLICY_HOLOBRAIN_NUM_WORKERS", 16))
_XP_MAX_STEP = int(os.environ.get("XPOLICY_HOLOBRAIN_MAX_STEP", int(1e5)))

config = dict(
    hist_steps=1,
    pred_steps=64,
    chunk_size=4,
    embed_dims=256,
    with_depth=True,
    with_depth_loss=True,
    min_depth=0.01,
    max_depth=1.2,
    num_depth=128,
    batch_size=_XP_BATCH_SIZE,
    max_step=_XP_MAX_STEP,
    step_log_freq=50,
    save_step_freq=5000,
    num_workers=_XP_NUM_WORKERS,
    lr=1e-4,
    training_with_subtask=False,
    with_cot=False,
    training_datasets=_DATASETS,
    # validation_datasets=["robotwin2_0"],
    deploy_datasets=_DATASETS,
    vlm_pretrain="Qwen/Qwen2.5-VL-3B-Instruct",
    num_vlm_layers=1,
    freeze_vlm=False,
    checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.0_Qwen/pretrain/model.safetensors",  # noqa: E501
)


def build_model(config):
    import copy

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.holobrain import (
        AdaRMSNorm,
        HoloBrain_Qwen2_5_VL,
        HoloBrain_Qwen2_5_VLConfig,
        HoloBrainActionDecoder,
        HoloBrainActionLoss,
        HoloBrainDecoderBaseConfig,
        HoloBrainDecoderTransformerConfig,
        HoloBrainEncoderBaseConfig,
        HoloBrainEncoderTransformerConfig,
        HoloBrainRobotStateEncoder,
        HoloBrainTrainingConfig,
        JointGraphAttention,
        RotaryAttention,
        TemporalJointGraphAttention,
        TextTemplate,
        UpsampleHead,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.layers.transformer_layers import FFN
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )

    patch_size = 28
    model_class = HoloBrain_Qwen2_5_VL
    model_config = HoloBrain_Qwen2_5_VLConfig

    embed_dims = config["embed_dims"]
    decoder_norm = nn.RMSNorm

    num_chunk = config["pred_steps"] // config["chunk_size"]
    state_dims = 8  # [joint_angle, x, y, z, qw, qx, qy, qz]
    head = dict(
        type=UpsampleHead,
        upsample_sizes=[num_chunk * 2, config["pred_steps"]],
        input_dim=embed_dims,
        dims=[128, 64],
        norm=dict(type=decoder_norm, normalized_shape=embed_dims),
        act=dict(type=nn.SiLU, inplace=True),
        norm_act_idx=[0, 1],
        num_output_layers=2,
        out_dim=state_dims,
    )
    with_mobile = config.get("with_mobile", False)
    if with_mobile:
        mobile_head = copy.deepcopy(head)
        mobile_head.update(out_dim=2)
    else:
        mobile_head = None

    decoder_operation_order = [
        "t_norm",
        "temp_joint_attn",
        "gate_msa",
        "norm",
        "img_cross_attn",
        "norm",
        "text_cross_attn",
        "norm",
        "scale_shift",
        "ffn",
        "gate_mlp",
    ] * config.get("decoder_layers", 6)

    model = model_class(
        cfg=model_config(
            with_cot=config["with_cot"],
            vlm_pretrain=config["vlm_pretrain"],
            num_vlm_layers=config.get("num_vlm_layers"),
            freeze_vlm=config.get("freeze_vlm", True),
            use_state_dict_with_vlm=not config.get("freeze_vlm", True),
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                # input image should in BGR convention, it will be converted to RGB here  # noqa: E501
                channel_flip=True,
                unsqueeze_depth_channel=True,
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config["min_depth"],
                        max_depth=config["max_depth"],
                        num_depth=config["num_depth"],
                        origin_stride=2,
                        valid_threshold=0.5,
                        stride=(patch_size,),
                    ),
                    dict(
                        type=TextTemplate,
                        with_subtask=config["training_with_subtask"],
                    ),
                ],
            ),
            backbone_3d=(
                dict(
                    type=SwinTransformer,
                    in_channels=1,
                    embed_dims=32,
                    depths=[2, 6, 2],
                    num_heads=[2, 4, 8],
                    window_size=8,
                    patch_size=patch_size // 4,
                    strides=[patch_size // 4, 2, 2],
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    out_indices=(2,),
                    with_cp=True,
                    convert_weights=False,
                )
                if config.get("with_depth")
                else None
            ),
            spatial_enhancer=dict(
                type=DepthFusionSpatialEnhancer,
                embed_dims=embed_dims,
                feature_3d_dim=128,
                num_depth_layers=2,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                num_depth=config["num_depth"],
                with_feature_3d=config.get("with_depth"),
                loss_depth_weight=(
                    config.get("loss_depth_weight", 1.0)
                    if config.get("with_depth_loss")
                    else -1
                ),
            ),
            decoder=dict(
                type=HoloBrainActionDecoder,
                head=head,
                mobile_head=mobile_head,
                transformer_cfg=HoloBrainDecoderTransformerConfig(
                    img_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    temp_joint_attn=dict(
                        type=TemporalJointGraphAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    norm_layer=dict(
                        type=decoder_norm,
                        normalized_shape=embed_dims,
                    ),
                    ffn=dict(
                        type=FFN,
                        embed_dims=embed_dims,
                        feedforward_channels=2048,
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                    ),
                    joint_self_attn=dict(
                        type=JointGraphAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                    ),
                    temp_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=32,
                    ),
                    text_cross_attn=dict(
                        type=RotaryAttention,
                        embed_dims=embed_dims,
                        num_heads=8,
                        max_position_embeddings=512,
                    ),
                    timestep_norm_layer=dict(
                        type=AdaRMSNorm,
                        normalized_shape=embed_dims,
                        condition_dims=256,
                        zero=True,
                    ),
                    operation_order=decoder_operation_order,
                ),
                base_cfg=HoloBrainDecoderBaseConfig(
                    chunk_size=config.get("chunk_size", 8),
                    use_joint_mask=True,
                    noise_type="local_joint",
                    pred_scaled_joint=False,
                    prediction_type="relative_joint_relative_pose",
                    pred_steps=config["pred_steps"],
                    state_dims=state_dims,
                    embed_dims=embed_dims,
                    with_mobile=with_mobile,
                    training_noise_scheduler=dict(
                        type=DDPMScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                        clip_sample=False,
                    ),
                    test_noise_scheduler=dict(
                        type=DPMSolverMultistepScheduler,
                        num_train_timesteps=1000,
                        beta_schedule="squaredcos_cap_v2",
                        prediction_type="sample",
                    ),
                    num_inference_timesteps=10,
                    feature_level=[0],
                    act_cfg=dict(type=nn.SiLU, inplace=True),
                ),
                training_cfg=HoloBrainTrainingConfig(
                    temporal_attn_drop=0.05,
                    num_parallel_training_sample=4,
                    teacher_forcing_rate=0.02,
                    loss=dict(
                        type=HoloBrainActionLoss,
                        timestep_loss_weight=1000,
                        parallel_loss_weight=0.1,
                        smooth_l1_beta=0.04,
                        loss_mode="smooth_l1",
                    ),
                ),
                robot_encoder=dict(
                    type=HoloBrainRobotStateEncoder,
                    transformer_cfg=HoloBrainEncoderTransformerConfig(
                        joint_self_attn=dict(
                            type=JointGraphAttention,
                            embed_dims=embed_dims,
                            num_heads=8,
                        ),
                        norm_layer=dict(
                            type=decoder_norm,
                            normalized_shape=embed_dims,
                        ),
                        ffn=dict(
                            type=FFN,
                            embed_dims=embed_dims,
                            feedforward_channels=2048,
                            act_cfg=dict(type=nn.SiLU, inplace=True),
                        ),
                        temp_self_attn=dict(
                            type=RotaryAttention,
                            embed_dims=embed_dims,
                            num_heads=8,
                            max_position_embeddings=32,
                        ),
                        operation_order=[
                            "norm",
                            "joint_self_attn",
                            None,
                            None,
                            "norm",
                            "ffn",
                        ]
                        * 4
                        + ["norm"],
                    ),
                    base_cfg=HoloBrainEncoderBaseConfig(
                        embed_dims=embed_dims,
                        chunk_size=min(8, config["hist_steps"]),
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                        state_dims=state_dims,
                    ),
                ),
            ),
        )
    )
    return model


def build_optimizer(config, model):
    import torch
    from torch import optim

    base_lr = config["lr"]
    max_step = config["max_step"]

    vlm_params = []
    bit16_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "vlm." in name:
            if p.requires_grad:
                vlm_params.append(p)
        elif p.dtype == torch.float16 or p.dtype == torch.bfloat16:
            bit16_params.append(p)
        else:
            other_params.append(p)
    optim_params = [
        {"params": bit16_params},
        {"params": other_params},
    ]
    if len(vlm_params) > 0:
        optim_params.append(
            {"params": vlm_params, "lr": base_lr * 0.1},
        )
    optimizer = optim.AdamW(
        optim_params,
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.0005),
    )
    lr_scheduler = optim.lr_scheduler.ChainedScheduler(
        [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=500
            ),
            optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(max_step * 0.9)],
                gamma=0.1,
            ),
        ]
    )
    return optimizer, lr_scheduler


def build_training_dataset(config, lazy_init=False):
    from dataset_factory import build_training_dataset as build

    return build(config, lazy_init)


def build_validation_dataset(config, lazy_init=False):
    from dataset_factory import build_validation_dataset as build

    return build(config, lazy_init)


def build_processors(config):
    from dataset_factory import build_processors as build

    return build(config)
