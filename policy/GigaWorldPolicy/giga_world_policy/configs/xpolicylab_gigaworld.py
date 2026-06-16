import os

# XPolicyLab LeRobot v2.1 training defaults.
# Task selection is controlled by GIGAWORLD_DATA_DIR; the dataset directory
# may contain one task or a joint subset with multiple tasks.

dst_size = (320, 256)
num_frames = int(os.environ.get("GIGAWORLD_NUM_FRAMES", "24"))
action_dim = int(os.environ.get("GIGAWORLD_MODEL_ACTION_DIM", "14"))
state_dim = int(os.environ.get("GIGAWORLD_MODEL_STATE_DIM", str(action_dim)))

exp_name = "xpolicylab_gigaworld"
date_str = os.environ.get("date", "default")
# Default to a repo-relative experiments dir under policy/GigaWorldPolicy; override with GIGAWORLD_OUTPUT_ROOT.
_DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
output_root = os.environ.get("GIGAWORLD_OUTPUT_ROOT", _DEFAULT_OUTPUT_ROOT)
project_dir = f"{output_root}/experiments/{exp_name}_{date_str}"

data_path = os.environ.get("GIGAWORLD_DATA_DIR", "")
norm_path = os.environ.get("GIGAWORLD_NORM_PATH", os.path.join(data_path, "norm_stats_delta.json") if data_path else "")
pretrained_path = os.environ.get(
    "GIGAWORLD_PRETRAINED_PATH",
    os.environ.get("WAN22_DIFFUSERS_PATH", ""),
)
robotype = os.environ.get("GIGAWORLD_ROBOTYPE", "arx5")

view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

if data_path:
    data_or_config = [
        dict(
            _class_name="LeRobotDataset",
            data_path=data_path,
            data_size=None,
            delta_info={"action": num_frames},
            delta_frames={k: image_frame_offsets for k in view_keys},
            video_backend=os.environ.get("GIGAWORLD_VIDEO_BACKEND", "pyav"),
            robotype=robotype,
        )
    ]
else:
    data_or_config = []

config = dict(
    project_dir=project_dir,
    runners=["MoTCasualWATrainerPretrain"],
    wandb=dict(
        project="gwp-xpolicylab",
        name=f"{exp_name}_{date_str}",
        mode=os.environ.get("WANDB_MODE", "offline"),
        init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=int(os.environ.get("GIGAWORLD_BATCH_SIZE_PER_GPU", "2")),
            num_workers=int(os.environ.get("GIGAWORLD_NUM_WORKERS", "4")),
            transform=dict(
                type="WATransformsLerobot",
                dst_size=dst_size,
                num_frames=num_frames,
                is_train=True,
                norm_path=norm_path,
                model_action_dim=action_dim,
                model_state_dim=state_dim,
                num_views=3,
                view_keys=view_keys,
                t5_len=64,
                robotype_to_embed_id={
                    "arx5": 0,
                    "dual_x5": 0,
                    "xpolicylab": 0,
                },
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                skip_action_norm=False,
                tshape=True,
                tshape_head_index=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=pretrained_path,
        checkpoint=os.environ.get("GIGAWORLD_INIT_CHECKPOINT") or None,
        strict=False,
        action_dim=action_dim,
        state_dim=state_dim,
        type="mot",
        mot_checkpoint_mixed_attn=True,
        video_attention_mask_mode="gwp_casual",
        action_expert=dict(
            hidden_dim=1024,
            ffn_dim=4096,
        ),
        flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        action_loss_weight=1.0,
        visual_loss_weight=1.0,
        freeze_action=False,
        use_gt_action_for_video=False,
        view_dir=project_dir,
        state_repeats=1,
        view_interval=200,
    ),
    optimizers=dict(
        type="CAME8Bit",
        lr=float(os.environ.get("GIGAWORLD_LR", "4e-5")),
        weight_decay=float(os.environ.get("GIGAWORLD_WEIGHT_DECAY", "1e-2")),
    ),
    schedulers=dict(
        type="CosineScheduler",
        warmup_steps=int(os.environ.get("GIGAWORLD_WARMUP_STEPS", "1000")),
        decay_epochs=int(os.environ.get("GIGAWORLD_DECAY_EPOCHS", "5")),
        decay_lr=float(os.environ.get("GIGAWORLD_DECAY_LR", "4e-6")),
    ),
    train=dict(
        resume=False,
        max_epochs=int(os.environ.get("GIGAWORLD_MAX_EPOCHS", "5")),
        max_steps=int(os.environ.get("GIGAWORLD_MAX_STEPS", "0")),
        gradient_accumulation_steps=int(os.environ.get("GIGAWORLD_GRAD_ACCUM", "2")),
        mixed_precision=os.environ.get("GIGAWORLD_MIXED_PRECISION", "bf16"),
        checkpoint_interval=int(os.environ.get("GIGAWORLD_CHECKPOINT_INTERVAL", "5000")),
        checkpoint_epoch_interval=int(os.environ.get("GIGAWORLD_CHECKPOINT_EPOCH_INTERVAL", "0")),
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_interval=int(os.environ.get("GIGAWORLD_LOG_INTERVAL", "2")),
        with_ema=True,
        ema=dict(
            enabled=True,
            decay=float(os.environ.get("GIGAWORLD_EMA_DECAY", "0.995")),
            device="model",
        ),
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
