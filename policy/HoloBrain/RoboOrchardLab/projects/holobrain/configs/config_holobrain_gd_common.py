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
    batch_size=8,
    max_step=int(1e5),
    step_log_freq=50,
    save_step_freq=5000,
    num_workers=8,
    lr=1e-4,
    training_datasets=[
        "robotwin2_0",
        "robotwin2_0_ur5_wsg",
    ],
    # validation_datasets=["robotwin2_0"],
    deploy_datasets=[
        "robotwin2_0",
        "robotwin2_0_ur5_wsg",
    ],
    dst_wh=(320, 256),
    patch_size=64,
    multi_task=True,
    bert_checkpoint="google-bert/bert-base-uncased",
    checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.0_GD/pretrain/model.safetensors",  # noqa: E501
)


def build_model(config):
    import copy

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from torch import nn

    from robo_orchard_lab.models.bip3d.bert import BertModel
    from robo_orchard_lab.models.bip3d.feature_enhancer import (
        TextImageDeformable2DEnhancer,
    )
    from robo_orchard_lab.models.bip3d.spatial_enhancer import (
        BatchDepthProbGTGenerator,
        DepthFusionSpatialEnhancer,
    )
    from robo_orchard_lab.models.bip3d.structure import BIP3D, BIP3DConfig
    from robo_orchard_lab.models.holobrain import (
        AdaRMSNorm,
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
        UpsampleHead,
    )
    from robo_orchard_lab.models.layers.data_preprocessors import (
        BaseDataPreprocessor,
    )
    from robo_orchard_lab.models.layers.transformer_layers import FFN
    from robo_orchard_lab.models.modules.channel_mapper import ChannelMapper
    from robo_orchard_lab.models.modules.swin_transformer import (
        SwinTransformer,
    )

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

    multi_task = config["multi_task"]
    decoder_operation_order = [
        "t_norm",
        "temp_joint_attn",
        "gate_msa",
        "norm",
        "img_cross_attn",
        "norm",
        "text_cross_attn" if multi_task else None,
        "norm" if multi_task else None,
        "scale_shift",
        "ffn",
        "gate_mlp",
    ] * config.get("decoder_layers", 6)

    if multi_task:
        num_feature_levels = 4
        depth_gt_stride = (8, 16, 32, 64)
    else:
        num_feature_levels = 3
        depth_gt_stride = (8, 16, 32)

    model = BIP3D(
        cfg=BIP3DConfig(
            data_preprocessor=dict(
                type=BaseDataPreprocessor,
                # input image should in BGR convention, it will be converted to RGB here  # noqa: E501
                channel_flip=True,
                unsqueeze_depth_channel=True,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                batch_transforms=[
                    dict(
                        type=BatchDepthProbGTGenerator,
                        min_depth=config["min_depth"],
                        max_depth=config["max_depth"],
                        num_depth=config["num_depth"],
                        origin_stride=2,
                        valid_threshold=0.5,
                        stride=depth_gt_stride,
                    ),
                ],
            ),
            embed_dims=embed_dims,
            backbone=dict(
                type=SwinTransformer,
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                out_indices=(1, 2, 3),
                with_cp=True,
                convert_weights=False,
            ),
            neck=dict(
                type=ChannelMapper,
                in_channels=[192, 384, 768],
                kernel_size=1,
                out_channels=embed_dims,
                act_cfg=None,
                bias=True,
                norm_cfg=dict(type=nn.GroupNorm, num_groups=32),
                num_outs=num_feature_levels,
            ),
            text_encoder=(
                dict(
                    type=BertModel,
                    special_tokens_list=["[CLS]", "[SEP]"],
                    name=config["bert_checkpoint"],
                    pad_to_max=False,
                    use_sub_sentence_represent=True,
                    add_pooling_layer=False,
                    max_tokens=768,
                    use_checkpoint=True,
                    return_tokenized=True,
                )
                if multi_task
                else None
            ),
            feature_enhancer=(
                dict(
                    type=TextImageDeformable2DEnhancer,
                    embed_dims=embed_dims,
                    num_layers=6,
                    text_img_attn_block=dict(
                        v_dim=embed_dims,
                        l_dim=embed_dims,
                        embed_dim=1024,
                        num_heads=4,
                        init_values=1e-4,
                    ),
                    img_attn_block=dict(
                        self_attn_cfg=dict(
                            embed_dims=embed_dims,
                            num_levels=num_feature_levels,
                            im2col_step=1,
                        ),
                        ffn_cfg=dict(
                            embed_dims=embed_dims,
                            feedforward_channels=2048,
                            ffn_drop=0.0,
                        ),
                    ),
                    text_attn_block=dict(
                        self_attn_cfg=dict(
                            num_heads=4,
                            embed_dims=embed_dims,
                        ),
                        ffn_cfg=dict(
                            embed_dims=embed_dims,
                            feedforward_channels=1024,
                            ffn_drop=0.0,
                        ),
                    ),
                    num_feature_levels=4,
                    positional_encoding=dict(
                        num_feats=embed_dims // 2,
                        normalize=True,
                        offset=0.0,
                        temperature=20,
                    ),
                )
                if multi_task
                else None
            ),
            backbone_3d=(
                dict(
                    type=SwinTransformer,
                    in_channels=1,
                    embed_dims=16,
                    depths=[2, 2, 6, 2],
                    num_heads=[4, 8, 8, 16],
                    window_size=7,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    out_indices=(1, 2, 3),
                    with_cp=True,
                    convert_weights=False,
                )
                if config.get("with_depth")
                else None
            ),
            neck_3d=(
                dict(
                    type=ChannelMapper,
                    in_channels=[32, 64, 128],
                    kernel_size=1,
                    out_channels=128,
                    act_cfg=None,
                    bias=True,
                    norm_cfg=dict(type=nn.GroupNorm, num_groups=4),
                    num_outs=num_feature_levels,
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
                    embed_dims=embed_dims,
                    state_dims=state_dims,
                    with_mobile=with_mobile,
                    act_cfg=dict(type=nn.SiLU, inplace=True),
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
                    feature_level=[1, 2],
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
                        state_dims=state_dims,
                        act_cfg=dict(type=nn.SiLU, inplace=True),
                    ),
                ),
            ),
        )
    )
    return model


def build_optimizer(config, model):
    from torch import optim

    model.text_encoder.requires_grad_(False)
    base_lr = config["lr"]
    max_step = config["max_step"]

    backbone_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone." in name or "text_encoder." in name or "neck." in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    optim_params = [
        {"params": backbone_params, "lr": base_lr * 0.1},
        {"params": other_params},
    ]
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
