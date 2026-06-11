import copy
import os

from .xpolicylab_gigaworld import config as _base_config


# Data dir and the video-pretrained init checkpoint are environment-driven.
# Set GIGAWORLD_DATA_DIR and GIGAWORLD_INIT_CHECKPOINT (or VIDEOPT_CKPT in the launcher).
DATA_DIR = os.environ.get("GIGAWORLD_DATA_DIR", "")
VIDEOPT_CKPT = os.environ.get("GIGAWORLD_INIT_CHECKPOINT", "")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


config = copy.deepcopy(_base_config)

# Joint action fine-tuning initialized from the completed videopt stage1
# checkpoint. Both video and action losses are active. This is not a resume:
# optimizer/scheduler state starts fresh while model weights are initialized
# from GIGAWORLD_INIT_CHECKPOINT.
config["project_dir"] = os.environ.get("GIGAWORLD_PROJECT_DIR", config["project_dir"])

train_loader = config["dataloaders"]["train"]
train_loader["batch_size_per_gpu"] = int(os.environ.get("GIGAWORLD_BATCH_SIZE_PER_GPU", "2"))
train_loader["num_workers"] = int(os.environ.get("GIGAWORLD_NUM_WORKERS", "4"))

transform = train_loader["transform"]
transform["num_frames"] = int(os.environ.get("GIGAWORLD_NUM_FRAMES", "28"))
transform["model_action_dim"] = int(os.environ.get("GIGAWORLD_MODEL_ACTION_DIM", "14"))
transform["model_state_dim"] = int(os.environ.get("GIGAWORLD_MODEL_STATE_DIM", "14"))
transform["norm_path"] = os.environ.get(
    "GIGAWORLD_NORM_PATH",
    os.path.join(DATA_DIR, "norm_stats_delta.json") if DATA_DIR else "",
)
transform["is_train"] = True
transform["skip_action_norm"] = False
transform["tshape"] = True
transform["tshape_head_index"] = 0

num_frames = transform["num_frames"]
view_keys = transform["view_keys"]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]
train_loader["data_or_config"] = [
    dict(
        _class_name="LeRobotDataset",
        data_path=os.environ.get("GIGAWORLD_DATA_DIR", DATA_DIR),
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={k: image_frame_offsets for k in view_keys},
        video_backend=os.environ.get("GIGAWORLD_VIDEO_BACKEND", "pyav"),
        robotype=os.environ.get("GIGAWORLD_ROBOTYPE", "arx5"),
    )
]

models = config["models"]
models["checkpoint"] = os.environ.get("GIGAWORLD_INIT_CHECKPOINT", VIDEOPT_CKPT)
models["action_dim"] = transform["model_action_dim"]
models["state_dim"] = transform["model_state_dim"]
models["action_loss_weight"] = float(os.environ.get("GIGAWORLD_ACTION_LOSS_WEIGHT", "1.0"))
models["visual_loss_weight"] = float(os.environ.get("GIGAWORLD_VISUAL_LOSS_WEIGHT", "1.0"))
models["freeze_backbone"] = _env_bool("GIGAWORLD_FREEZE_BACKBONE", False)
models["freeze_action"] = _env_bool("GIGAWORLD_FREEZE_ACTION", False)
models["use_gt_action_for_video"] = _env_bool("GIGAWORLD_USE_GT_ACTION_FOR_VIDEO", False)
models["view_interval"] = int(os.environ.get("GIGAWORLD_VIEW_INTERVAL", "2000"))

config["schedulers"]["warmup_steps"] = int(os.environ.get("GIGAWORLD_WARMUP_STEPS", "1000"))
config["schedulers"]["decay_epochs"] = int(os.environ.get("GIGAWORLD_DECAY_EPOCHS", "5"))

train = config["train"]
train["resume"] = _env_bool("GIGAWORLD_RESUME", False)
train["max_epochs"] = int(os.environ.get("GIGAWORLD_MAX_EPOCHS", "5"))
train["max_steps"] = int(os.environ.get("GIGAWORLD_MAX_STEPS", "0"))
train["gradient_accumulation_steps"] = int(os.environ.get("GIGAWORLD_GRAD_ACCUM", "2"))
train["checkpoint_interval"] = int(os.environ.get("GIGAWORLD_CHECKPOINT_INTERVAL", "25000"))
train["checkpoint_epoch_interval"] = int(os.environ.get("GIGAWORLD_CHECKPOINT_EPOCH_INTERVAL", "1"))
train["log_interval"] = int(os.environ.get("GIGAWORLD_LOG_INTERVAL", "10"))
train["with_ema"] = _env_bool("GIGAWORLD_WITH_EMA", True)
train["ema"]["enabled"] = _env_bool("GIGAWORLD_WITH_EMA", True)

config["wandb"]["project"] = os.environ.get("GIGAWORLD_WANDB_PROJECT", "gwp-xpolicylab")
config["wandb"]["name"] = os.environ.get("GIGAWORLD_WANDB_NAME", "joint_action_stage2")
config["wandb"]["mode"] = os.environ.get("WANDB_MODE", "online")
