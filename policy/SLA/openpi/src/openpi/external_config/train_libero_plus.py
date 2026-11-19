from openpi.training.config import TrainConfig, AssetsConfig, pi0_config, LeRobotLiberoPlusDataConfig, DataConfig, LeRobotLiberoDataConfig
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer


my_config = TrainConfig(
    name="train_libero_plus",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=LeRobotLiberoPlusDataConfig(
        repo_id="Data/libero_lerobot",
        assets=AssetsConfig(
            assets_dir="assets",
            asset_id="train_libero_plus",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        use_quantile_norm=True,
        unified_gripper=True,
        repack_transforms=LeRobotLiberoDataConfig().repack_transforms,
        action_sequence_keys=("actions",),
    ),
    state_dropout_prob=0.25,
    freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=10).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=30_000,
    save_interval=10_000,
    batch_size=512,
    num_workers=32,
    fsdp_devices=8,
    ema_decay=0.999,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    # Below is Spatial Forcing configs
    align_enabled=True,
    align_target_model="auto",
    vggt_weight_path="checkpoints/omnivggt",
    vla_layers_align=12,
    vla_adapt_layers_align=False,
    vggt_layers_align=-1,
    vggt_adapt_layers_align=False,
    pooling_func="bilinear",
    use_vggt_pe=True,
    use_vlm_norm=True,
    align_loss_coeff=0.5,
    ignore_img_padding_area=True,
    # omnivggt configs
    use_camera_params=False,
    # align hard-device config
    align_device_mode="shared",
    align_vggt_devices=8,
    align_feature_batch_size=32,
)
