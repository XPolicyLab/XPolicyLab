"""GO-1 fine-tuning config for shared RoboDojo multi-task LeRobot datasets."""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments

from go1.configs.go1_base_cfg import BaseDatasetArguments, BaseModelArguments, BaseSpaceArguments
from go1.tools.env_parse import get_bool_env

RUNNAME = os.environ.get("RUNNAME", "go1_robodojo_shared")
DEBUG_MODE = get_bool_env("DEBUG_MODE")

DATA_ROOT_DIR = os.environ.get("DATA_ROOT_DIR", "/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21")
ACTION_DIM = int(os.environ.get("ACTION_DIM", "14"))
STATE_DIM = int(os.environ.get("STATE_DIM", "14"))
CTRL_FREQ = int(os.environ.get("CTRL_FREQ", "25"))
ACTION_CHUNK_SIZE = int(os.environ.get("ACTION_CHUNK_SIZE", "25"))
MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "/mnt/pfs/pg4hw0/qiwei/models/GO-1")
if MODEL_NAME_OR_PATH == "/mnt/pfs/pg4hw0/qiwei/models/GO-1":
    MODEL_NAME_OR_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../../../../models/GO-1")
    )
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "Do your job.")
SEED = int(os.environ.get("TRAIN_SEED", "42"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "12" if not DEBUG_MODE else "1"))
NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", "1"))
REPORT_TO = os.environ.get("REPORT_TO", "tensorboard")
LOGGING_DIR = os.environ.get(
    "LOGGING_DIR", os.path.join(os.path.dirname(__file__), "../../..", "checkpoints", RUNNAME, "tb")
)
WANDB_NAME = os.environ.get("WANDB_NAME", RUNNAME)


@dataclass
class DatasetArguments(BaseDatasetArguments):
    dataset_type: Optional[str] = field(default="lerobot")
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [DATA_ROOT_DIR],
    )
    transforms: Optional[List[str]] = field(default_factory=lambda: [dict(type="Normalize")])


@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default=MODEL_NAME_OR_PATH)
    freeze_llm: bool = field(default=False if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=False if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=False if not DEBUG_MODE else True)
    action_chunk_size: int = field(default=ACTION_CHUNK_SIZE)
    latent_planning: bool = field(default=True)
    freeze_latent_planner: bool = field(default=False)


@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=os.path.join(os.path.dirname(__file__), "../../..", "checkpoints", RUNNAME))
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=20 if not DEBUG_MODE else 0)
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=float(NUM_EPOCHS))
    per_device_train_batch_size: int = field(default=BATCH_SIZE)
    gradient_accumulation_steps: int = field(default=2 if DEBUG_MODE else 1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    do_train: bool = field(default=True)
    deepspeed: str = field(default="go1/zero_stage1_config.json")
    seed: int = field(default=SEED)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=100)
    logging_steps: int = field(default=10)
    logging_dir: str = field(default=LOGGING_DIR)
    run_name: str = field(default=WANDB_NAME)
    report_to: str = field(default=REPORT_TO)


@dataclass
class SpaceArguments(BaseSpaceArguments):
    state_dim: int = field(default=STATE_DIM)
    action_dim: int = field(default=ACTION_DIM)
    space_repack: dict = field(
        default_factory=lambda: {
            "state": "observation.state",
            "action": "action",
            "cam_head_color": "observation.images.cam_high",
            "cam_hand_left_color": "observation.images.cam_left_wrist",
            "cam_hand_right_color": "observation.images.cam_right_wrist",
        }
    )
    ctrl_freq: int = field(default=CTRL_FREQ)
    default_prompt: str = field(default=DEFAULT_PROMPT)
