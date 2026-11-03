"""Checkpoint configuration definitions carried with executable skills.

The seed-zero manipulation configuration is adapted from XPolicyLab's OpenPI
training configuration. Its model and data transforms are preserved; the assets
root is resolved from the supplied checkpoint rather than a development checkout.
"""
from pathlib import Path
from openpi.training import config
from openpi.models import pi0_config
from openpi import transforms
from openpi.training import weight_loaders


def get_config(name, checkpoint):
    try:
        return config.get_config(name)
    except ValueError:
        if name != 'pi05_base_aloha_full_sim_arx-x5_seed_0':
            raise
    return config.TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True),
        data=config.LeRobotAlohaDataConfig(
            repo_id='RoboDojo_sim_arx-x5_v30',
            assets=config.AssetsConfig(assets_dir=str(Path(checkpoint) / 'assets'), asset_id='arx_x5_sim'),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'images': {'cam_high': 'observation.images.cam_high',
                           'cam_left_wrist': 'observation.images.cam_left_wrist',
                           'cam_right_wrist': 'observation.images.cam_right_wrist'},
                'state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
            })]),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params'),
        seed=0, batch_size=256, fsdp_devices=2, num_train_steps=60000,
    )
