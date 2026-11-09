from openpi.training.config import TrainConfig, AssetsConfig, pi0_config, DataConfig, LeRobotRobotwinDataConfig
from openpi.training import weight_loaders
import openpi.transforms as _transforms

config_dict = dict(
    name="train_robotwin",
    model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m"),
    data=LeRobotRobotwinDataConfig(
        repo_id="Data/robotwin_lerobot",
        assets=AssetsConfig(
            assets_dir="assets",
            asset_id="train_robotwin",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        use_eef_action=True,
        use_so3_rpy=True,
        use_quantile_norm=True,
    ),
    state_dropout_prob=0.25,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, 
        paligemma_variant="gemma_2b_lora", 
        action_expert_variant="gemma_300m",
    ).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=30_000,
    save_interval=30_000,
    batch_size=256,
    num_workers=32,
    fsdp_devices=8,
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
