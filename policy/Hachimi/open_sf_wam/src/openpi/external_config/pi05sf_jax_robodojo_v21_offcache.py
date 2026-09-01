import os

import openpi.transforms as _transforms
from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig, DataConfig, LeRobotAlohaDataConfig, TrainConfig, pi0_config


PI05_BASE_PATH = os.environ.get("PI05_BASE_PATH", "./checkpoints/pi05_base")
VGGT_WEIGHT_PATH = os.environ.get("VGGT_WEIGHT_PATH", "./checkpoints/VGGT-1B")
SF_CACHE_DIR = os.environ.get("SF_CACHE_DIR", "./results/sf_cache")
PI05SF_ASSETS_DIR = os.environ.get("PI05SF_ASSETS_DIR", "./assets/pi05sf_robodojo_v21")

my_config = TrainConfig(
    name="pi05sf_jax_robodojo_v21_offcache",
    project_name="xpolicylab-pi05sf-jax",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotAlohaDataConfig(
        repo_id="RoboDojo_lerobot_v21_video",
        assets=AssetsConfig(
            assets_dir=PI05SF_ASSETS_DIR,
            asset_id="RoboDojo_lerobot_v21_video",
        ),
        adapt_to_pi=False,
        use_delta_joint_actions=False,
        base_config=DataConfig(prompt_from_task=True),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(f"{PI05_BASE_PATH}/params"),
    vggt_weight_path=VGGT_WEIGHT_PATH,
    vla_layers_align=12,
    vggt_layers_align=-1,
    pooling_func="bilinear",
    use_vggt_pe=True,
    use_vlm_norm=True,
    use_camera_params=False,
    align_enabled=True,
    align_target_model="vggt",
    align_loss_coeff=0.2,
    ignore_img_padding_area=True,
    sf_cache_enable=True,
    sf_cache_mode="readonly",
    sf_cache_miss_policy="error",
    sf_cache_dir=SF_CACHE_DIR,
    sf_cache_save_dtype="bf16",
    sf_cache_chunk_size=128,
    sf_cache_strict_shape=True,
    sf_dataset_uid=0,
    seed=0,
    batch_size=256,
    num_workers=8,
    num_train_steps=60_000,
    save_interval=5000,
    keep_period=5000,
    ema_decay=None,
    fsdp_devices=8,
    wandb_enabled=False,
)
