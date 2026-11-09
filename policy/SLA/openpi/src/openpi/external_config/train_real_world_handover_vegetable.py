from openpi.training.config import TrainConfig, AssetsConfig, pi0_config, DataConfig, LeRobotPiperDataConfig
from openpi.training import weight_loaders

config_dict = dict(
    name="train_real_world_handover_vegetable",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotPiperDataConfig(
        repo_id="Data/real_world_handover_vegetable/*",
        assets=AssetsConfig(
            assets_dir="assets",
            asset_id="train_real_world_handover_vegetable",
        ),
        use_eef_action=False,
        use_so3_rpy=False,
        unified_gripper=True,
        use_quantile_norm=True,
        base_config=DataConfig(prompt_from_task=True),
    ),
    state_dropout_prob=0.25,
    freeze_filter=pi0_config.Pi0Config(pi05=True).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=20_000,
    save_interval=20_000,
    batch_size=64,
    num_workers=8,
    fsdp_devices=1,
    ema_decay=0.999,
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
my_config = TrainConfig(**config_dict)
