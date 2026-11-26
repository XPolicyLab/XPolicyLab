import pathlib

from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import LeRobotRobodojoRealDataConfig
from openpi.training.config import MixedDataSourceConfig
from openpi.training.config import MixedLeRobotDataConfig
from openpi.training.config import TrainConfig
from openpi.training.config import pi0_config

# One LeRobot dataset per task and robot under Data/RoboDojo_real_ee in the policy directory. lerobot treats a
# relative repo_id as a Hugging Face cache path, so the path is made absolute here.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[4] / "Data" / "RoboDojo_real_ee"
# History frames from cam_high: the episode's first frame and one frame per second over the last 8 seconds.
# piper_x is recorded at 25 fps, arx_x5 and piper at 30 fps, so each source uses its own stride.
_EXTRA_FRAMES_25HZ = (0, *range(-8 * 25, 0, 25))
_EXTRA_FRAMES_30HZ = (0, *range(-8 * 30, 0, 30))

# RoboDojo real robots (piper_x, arx_x5, piper) trained together, end-effector actions.
my_config = TrainConfig(
    name="train_robodojo_real",
    model=pi0_config.Pi0Config(pi05=True),
    data=MixedLeRobotDataConfig(
        repo_id="robodojo_real",
        sources=(
            MixedDataSourceConfig(
                name="real_piper_x",
                weight=1.0,
                data_factory=LeRobotRobodojoRealDataConfig(
                    repo_id=f"{_DATA_ROOT}/*piper_x",
                    use_eef_action=True,
                    use_so3_rpy=True,
                    base_config=DataConfig(prompt_from_task=True, extra_frames=_EXTRA_FRAMES_25HZ),
                ),
            ),
            MixedDataSourceConfig(
                name="real_arx_x5",
                weight=1.0,
                data_factory=LeRobotRobodojoRealDataConfig(
                    repo_id=f"{_DATA_ROOT}/*arx_x5",
                    use_eef_action=True,
                    use_so3_rpy=True,
                    base_config=DataConfig(prompt_from_task=True, extra_frames=_EXTRA_FRAMES_30HZ),
                ),
            ),
            MixedDataSourceConfig(
                name="real_piper",
                weight=1.0,
                data_factory=LeRobotRobodojoRealDataConfig(
                    repo_id=f"{_DATA_ROOT}/*piper",
                    use_eef_action=True,
                    use_so3_rpy=True,
                    base_config=DataConfig(prompt_from_task=True, extra_frames=_EXTRA_FRAMES_30HZ),
                ),
            ),
        ),
        use_quantile_norm=True,
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(assets_dir="assets", asset_id="train_robodojo_real"),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/SLA_base/params"),
    num_train_steps=60_000,
    save_interval=10_000,
    # 512 samples per step. On one node of 80 GB GPUs pass `--batch-size 256 --gradient-accumulation-steps 2`.
    batch_size=512,
    num_workers=32,
    fsdp_devices=8,
    state_dropout_prob=0.25,
    ema_decay=0.999,
)
