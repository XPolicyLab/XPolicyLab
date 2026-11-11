import pathlib

from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotRobodojoDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# LeRobot datasets live under Data/ in the policy directory. lerobot treats a relative repo_id as a Hugging Face
# cache path, so the path is made absolute here.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[4] / "Data"

# RoboDojo simulation (arx_x5), end-effector actions. Besides the three current views the model sees
# 9 history frames from cam_high: the episode's first frame and one frame per second over the last
# 8 seconds (the data is recorded at 25 fps).
my_config = TrainConfig(
    name="train_robodojo_sim",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotRobodojoDataConfig(
        repo_id=str(_DATA_ROOT / "RoboDojo_sim_ee"),
        use_eef_action=True,
        use_so3_rpy=True,
        use_quantile_norm=True,
        base_config=DataConfig(prompt_from_task=True, extra_frames=(0, *range(-8 * 25, 0, 25))),
        assets=AssetsConfig(assets_dir="assets", asset_id="train_robodojo_sim"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=60_000,
    save_interval=15_000,
    batch_size=256,
    num_workers=32,
    fsdp_devices=8,
    state_dropout_prob=0.25,
    ema_decay=0.999,
)
