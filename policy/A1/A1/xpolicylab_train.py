#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import yaml


ENV_TO_CFG = {
    "DATA_DIR": ("paths", "data_dir"),
    "PRETRAIN_CHECKPOINT": ("paths", "pretrain_checkpoint"),
    "HF_HOME": ("paths", "hf_home"),
    "XDG_CACHE_HOME": ("paths", "xdg_cache_home"),
    "HF_HUB_OFFLINE": ("paths", "hf_hub_offline"),
    "LEROBOT_DATA_PATH": ("paths", "lerobot_data_path"),
    "LEROBOT_OUTPUT_DIR": ("paths", "lerobot_output_dir"),
    "TRAIN_STEPS": ("training", "train_steps"),
    "GLOBAL_BATCH_SIZE": ("training", "global_batch_size"),
    "DEVICE_TRAIN_MICROBATCH_SIZE": ("training", "device_train_microbatch_size"),
    "CROP_MODE": ("training", "crop_mode"),
    "MAX_CROPS": ("training", "max_crops"),
    "NUM_WORKERS": ("training", "num_workers"),
    "SEQ_LEN": ("training", "seq_len"),
    "LOG_INTERVAL": ("training", "log_interval"),
    "SAVE_INTERVAL": ("checkpoint", "save_interval"),
    "SAVE_INTERVAL_UNSHARDED": ("checkpoint", "save_interval_unsharded"),
    "SAVE_NUM_CHECKPOINTS_TO_KEEP": ("checkpoint", "save_num_checkpoints_to_keep"),
    "SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP": ("checkpoint", "save_num_unsharded_checkpoints_to_keep"),
    "EARLY_EXIT": ("model", "early_exit"),
    "TRAIN_EXIT_RANDOM_LAYER": ("model", "train_exit_random_layer"),
    "FT_CONNECTOR": ("model", "ft_connector"),
    "FT_VIT": ("model", "ft_vit"),
    "FT_LLM": ("model", "ft_llm"),
    "FT_EMBEDDING": ("model", "ft_embedding"),
    "CONNECTOR_LR": ("optimizer", "connector_lr"),
    "VIT_LR": ("optimizer", "vit_lr"),
    "LLM_LR": ("optimizer", "llm_lr"),
    "ACTION_HEAD_LR": ("optimizer", "action_head_lr"),
    "CONNECTOR_WEIGHT_DECAY": ("optimizer", "connector_weight_decay"),
    "VIT_WEIGHT_DECAY": ("optimizer", "vit_weight_decay"),
    "LLM_WEIGHT_DECAY": ("optimizer", "llm_weight_decay"),
    "ACTION_HEAD_WEIGHT_DECAY": ("optimizer", "action_head_weight_decay"),
    "ADAM_BETA1": ("optimizer", "beta1"),
    "ADAM_BETA2": ("optimizer", "beta2"),
    "WARMUP_STEPS": ("scheduler", "warmup_steps"),
    "FREEZE_STEPS": ("scheduler", "freeze_steps"),
    "SCHEDULER_ALPHA_F": ("scheduler", "alpha_f"),
    "WARMUP_MIN_LR": ("scheduler", "warmup_min_lr"),
    "TORCH_DISTRIBUTED_TIMEOUT": ("distributed", "torch_distributed_timeout"),
    "TORCH_NCCL_TRACE_BUFFER_SIZE": ("distributed", "torch_nccl_trace_buffer_size"),
    "TORCH_NCCL_DUMP_ON_TIMEOUT": ("distributed", "torch_nccl_dump_on_timeout"),
    "ENABLE_WANDB": ("wandb", "enable"),
    "WANDB_API_KEY": ("wandb", "api_key"),
    "WANDB_PROJECT": ("wandb", "project"),
    "WANDB_ENTITY": ("wandb", "entity"),
    "WANDB_RUN_NAME": ("wandb", "run_name"),
    "WANDB_MODE": ("wandb", "mode"),
    "WANDB_REQUIRED": ("wandb", "required"),
}

DEFAULTS = {
    "HF_HUB_OFFLINE": "0",
    "TRAIN_STEPS": "10000",
    "GLOBAL_BATCH_SIZE": "16",
    "DEVICE_TRAIN_MICROBATCH_SIZE": "1",
    "CROP_MODE": "resize",
    "MAX_CROPS": "8",
    "NUM_WORKERS": "auto",
    "SEQ_LEN": "2048",
    "LOG_INTERVAL": "10",
    "SAVE_INTERVAL": "2000",
    "SAVE_INTERVAL_UNSHARDED": "2000",
    "SAVE_NUM_CHECKPOINTS_TO_KEEP": "0",
    "SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP": "1",
    "EARLY_EXIT": "false",
    "TRAIN_EXIT_RANDOM_LAYER": "false",
    "FT_CONNECTOR": "false",
    "FT_VIT": "false",
    "FT_LLM": "false",
    "FT_EMBEDDING": "lm_head",
    "CONNECTOR_LR": "2e-4",
    "VIT_LR": "6e-6",
    "LLM_LR": "5e-5",
    "ACTION_HEAD_LR": "5e-5",
    "CONNECTOR_WEIGHT_DECAY": "0.0",
    "VIT_WEIGHT_DECAY": "0.0",
    "LLM_WEIGHT_DECAY": "0.0",
    "ACTION_HEAD_WEIGHT_DECAY": "0.0",
    "ADAM_BETA1": "0.9",
    "ADAM_BETA2": "0.95",
    "WARMUP_STEPS": "2000",
    "FREEZE_STEPS": "0",
    "SCHEDULER_ALPHA_F": "0.1",
    "WARMUP_MIN_LR": "",
    "TORCH_DISTRIBUTED_TIMEOUT": "1800",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "ENABLE_WANDB": "false",
    "WANDB_API_KEY": "",
    "WANDB_PROJECT": "a1-xpolicylab",
    "WANDB_ENTITY": "",
    "WANDB_RUN_NAME": "",
    "WANDB_MODE": "online",
}


def nested_get(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def is_true(value):
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def abs_path(path):
    return str(Path(path).expanduser().resolve())


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _pick_stat(raw_stats, *keys):
    for key in keys:
        if key in raw_stats:
            return raw_stats[key]
    return None


def _load_lerobot_stats(dataset_path):
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(
        os.path.basename(str(dataset_path)),
        str(dataset_path),
        CODEBASE_VERSION,
        force_cache_sync=False,
    )
    raw_stats = _jsonable(meta.stats)
    stats = dict(raw_stats)

    state_stats = _pick_stat(raw_stats, "state", "observation.state")
    action_stats = _pick_stat(raw_stats, "action", "actions")
    if state_stats is not None:
        stats["state"] = state_stats
    if action_stats is not None:
        stats["action"] = action_stats
        stats["actions"] = action_stats

    camera_aliases = {
        "cam_head_color": ("cam_head_color", "observation.images.cam_high", "observation.images.cam_head"),
        "cam_hand_left_color": ("cam_hand_left_color", "observation.images.cam_left_wrist"),
        "cam_hand_right_color": ("cam_hand_right_color", "observation.images.cam_right_wrist"),
    }
    for alias, keys in camera_aliases.items():
        camera_stats = _pick_stat(raw_stats, *keys)
        if camera_stats is not None:
            stats[alias] = camera_stats

    return stats


def _write_dataset_stats(dataset_path, ckpt_dir):
    stats = _load_lerobot_stats(dataset_path)
    stats_path = ckpt_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    latest_unsharded = ckpt_dir / "latest-unsharded"
    targets = []
    if latest_unsharded.exists():
        targets.append(latest_unsharded.resolve() if latest_unsharded.is_symlink() else latest_unsharded)
    targets.extend(sorted(ckpt_dir.glob("step*-unsharded"), key=lambda path: path.stat().st_mtime, reverse=True)[:1])

    for target in targets:
        if target.is_dir():
            shutil.copy2(stats_path, target / "dataset_stats.json")

    return stats_path


def _clean_checkpoint_artifacts(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return

    latest_unsharded = ckpt_dir / "latest-unsharded"
    keep_unsharded = None
    if latest_unsharded.exists():
        try:
            keep_unsharded = latest_unsharded.resolve()
        except OSError:
            keep_unsharded = None

    unsharded = sorted(ckpt_dir.glob("step*-unsharded"), key=lambda path: path.stat().st_mtime, reverse=True)
    if keep_unsharded is None and unsharded:
        keep_unsharded = unsharded[0].resolve()
        latest_unsharded.unlink(missing_ok=True)
        try:
            latest_unsharded.symlink_to(unsharded[0].name, target_is_directory=True)
        except FileExistsError:
            pass

    for marker in ("latest", "latest-action-head"):
        marker_path = ckpt_dir / marker
        if marker_path.exists() or marker_path.is_symlink():
            if marker_path.is_dir() and not marker_path.is_symlink():
                shutil.rmtree(marker_path, ignore_errors=True)
            else:
                marker_path.unlink(missing_ok=True)

    for path in ckpt_dir.glob("step*"):
        if not path.is_dir():
            continue
        if path.name.endswith("-unsharded"):
            if keep_unsharded is not None and path.resolve() != keep_unsharded:
                shutil.rmtree(path, ignore_errors=True)
        else:
            shutil.rmtree(path, ignore_errors=True)


def load_config(policy_dir, a1_dir):
    os.environ.setdefault("SCRIPT_DIR", str(policy_dir))
    os.environ.setdefault("A1_DIR", str(a1_dir))

    config_file = os.environ.get("A1_TRAIN_CONFIG")
    if not config_file:
        local_config = a1_dir / "train_config.local.yaml"
        config_file = local_config if local_config.is_file() else a1_dir / "train_config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, Path(config_file)


def resolve_env(cfg, policy_dir, a1_dir, workspace_dir):
    data_dir_default = workspace_dir / "models"
    path_defaults = {
        "DATA_DIR": str(data_dir_default),
        "PRETRAIN_CHECKPOINT": str(data_dir_default / "a1-pretrain"),
        "HF_HOME": str(policy_dir / ".cache" / "huggingface"),
        "XDG_CACHE_HOME": str(policy_dir / ".cache"),
        "LEROBOT_OUTPUT_DIR": str(policy_dir / "data"),
    }

    values = {}
    for env_name, cfg_path in ENV_TO_CFG.items():
        if env_name in os.environ:
            value = os.environ[env_name]
        else:
            value = stringify(nested_get(cfg, cfg_path))
            if value:
                value = os.path.expandvars(value)
            else:
                value = path_defaults.get(env_name, DEFAULTS.get(env_name, ""))
        values[env_name] = value
        os.environ[env_name] = value

    for path_name in ("DATA_DIR", "PRETRAIN_CHECKPOINT", "HF_HOME", "XDG_CACHE_HOME", "LEROBOT_OUTPUT_DIR"):
        values[path_name] = abs_path(values[path_name])
        os.environ[path_name] = values[path_name]

    return values


def resolve_dataset_path(args, values, policy_dir):
    override_path = os.environ.get("LEROBOT_DATA_PATH_OVERRIDE")
    if override_path:
        dataset_path = Path(override_path).expanduser().resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"LeRobot dataset path does not exist: {dataset_path}")
        return dataset_path

    shared_name = f"{args.bench_name}_sim_{args.env_cfg_type.replace('_', '-')}_v21"
    candidates = []
    if values.get("LEROBOT_DATA_PATH"):
        candidates.append(Path(values["LEROBOT_DATA_PATH"]))
    candidates.extend(
        [
            Path("/mnt/xspark-data/xspark_shared/lerobot") / shared_name,
            Path(values["LEROBOT_OUTPUT_DIR"]) / f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}",
        ]
    )

    data_dir = Path(values["LEROBOT_OUTPUT_DIR"])
    if data_dir.is_dir():
        pattern = f"{args.bench_name}-*-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
        candidates.extend(sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    task_name = os.environ.get("TASK_NAME")
    if task_name:
        output_dir = Path(values["LEROBOT_OUTPUT_DIR"])
        subprocess.run(
            [
                "bash",
                str(policy_dir / "process_data.sh"),
                args.bench_name,
                task_name,
                args.env_cfg_type,
                args.expert_data_num,
                args.action_type,
                os.environ.get("DATASET_FPS", "30"),
                str(output_dir),
            ],
            check=True,
        )
        generated = output_dir / f"{args.bench_name}-{task_name}-{args.env_cfg_type}-{args.expert_data_num}-{args.action_type}"
        if generated.is_dir():
            return generated.resolve()

    raise FileNotFoundError(
        "No LeRobot dataset found. Set LEROBOT_DATA_PATH to an existing dataset, "
        "or set TASK_NAME to let A1 convert a local single-task dataset."
    )


def write_runtime_configs(a1_dir, dataset_path):
    normalization_type = os.environ.get("NORMALIZATION_TYPE", os.environ.get("A1_NORMALIZATION_TYPE", "bounds"))
    use_num_images = os.environ.get("USE_NUM_IMAGES", os.environ.get("A1_USE_NUM_IMAGES", "3"))
    dataset_cfg = a1_dir / "configs" / "datasets" / "xpolicylab_runtime.yaml"
    experiment_cfg = a1_dir / "configs" / "experiments" / "xpolicylab_runtime.yaml"
    dataset_cfg.parent.mkdir(parents=True, exist_ok=True)
    experiment_cfg.parent.mkdir(parents=True, exist_ok=True)
    dataset_cfg.write_text(
        "\n".join(
            [
                "image_augmentation:",
                "  enable: true",
                "  enable_random_erasing: true",
                "  enable_sharpening: true",
                "  augmentation_prob: 0.5",
                "",
                "lerobot:",
                f"  - path: {dataset_path}",
                "    weight: 1.0",
                f"    normalization_type: {normalization_type}",
                f"    use_num_images: {use_num_images}",
                "    image_augmentation: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    experiment_cfg.write_text(
        "model_config: models/pretrain.yaml\n"
        "dataset_config: datasets/xpolicylab_runtime.yaml\n",
        encoding="utf-8",
    )
    return dataset_cfg, experiment_cfg


def run(args):
    policy_dir = Path(__file__).resolve().parents[1]
    a1_dir = Path(__file__).resolve().parent
    workspace_dir = policy_dir.parents[2]
    xpl_dir = workspace_dir / "XPolicyLab"
    utils_dir = xpl_dir / "utils"

    cfg, config_file = load_config(policy_dir, a1_dir)
    values = resolve_env(cfg, policy_dir, a1_dir, workspace_dir)

    for mkdir_key in ("HF_HOME", "XDG_CACHE_HOME"):
        Path(values[mkdir_key]).mkdir(parents=True, exist_ok=True)
    (policy_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    pretrain_checkpoint = Path(values["PRETRAIN_CHECKPOINT"])
    default_pretrain = Path(values["DATA_DIR"]) / "a1-pretrain"
    if not pretrain_checkpoint.is_dir():
        if default_pretrain.is_dir():
            print(f"[WARN] PRETRAIN_CHECKPOINT does not exist: {pretrain_checkpoint}", file=sys.stderr)
            print(f"[WARN] Falling back to default pretrain checkpoint: {default_pretrain}", file=sys.stderr)
            pretrain_checkpoint = default_pretrain.resolve()
        else:
            raise FileNotFoundError(f"PRETRAIN_CHECKPOINT does not exist: {pretrain_checkpoint}")

    dataset_path = resolve_dataset_path(args, values, policy_dir)
    dataset_cfg, experiment_cfg = write_runtime_configs(a1_dir, dataset_path)

    action_dim = subprocess.check_output(
        ["bash", str(utils_dir / "get_action_dim.sh"), str(workspace_dir), args.env_cfg_type],
        text=True,
    ).strip()

    gpu_list = [gpu for gpu in args.gpu_id.split(",") if gpu != ""]
    nproc = str(len(gpu_list))
    if values["NUM_WORKERS"] == "auto":
        values["NUM_WORKERS"] = "0" if len(gpu_list) > 1 else "2"

    run_basename = (
        f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-"
        f"{args.expert_data_num}-{args.action_type}-{args.seed}"
    )
    runname = os.environ.get("RUNNAME") or run_basename
    wandb_run_name = values["WANDB_RUN_NAME"] or runname
    ckpt_dir = policy_dir / "checkpoints" / runname
    (ckpt_dir / "log").mkdir(parents=True, exist_ok=True)
    (policy_dir / "checkpoints" / f"{run_basename}.latest").write_text(str(ckpt_dir), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu_id,
            "DATA_DIR": values["DATA_DIR"],
            "PRETRAIN_CHECKPOINT": str(pretrain_checkpoint),
            "HF_HOME": values["HF_HOME"],
            "HF_HUB_OFFLINE": values["HF_HUB_OFFLINE"],
            "XDG_CACHE_HOME": values["XDG_CACHE_HOME"],
            "TORCH_DISTRIBUTED_TIMEOUT": values["TORCH_DISTRIBUTED_TIMEOUT"],
            "TORCH_NCCL_TRACE_BUFFER_SIZE": values["TORCH_NCCL_TRACE_BUFFER_SIZE"],
            "TORCH_NCCL_DUMP_ON_TIMEOUT": values["TORCH_NCCL_DUMP_ON_TIMEOUT"],
            "PYTHONPATH": f"{a1_dir}:{env.get('PYTHONPATH', '')}",
            "WANDB_API_KEY": values["WANDB_API_KEY"],
            "WANDB_PROJECT": values["WANDB_PROJECT"],
            "WANDB_MODE": values["WANDB_MODE"],
            "WANDB_REQUIRED": values["WANDB_REQUIRED"] or values["ENABLE_WANDB"],
            "WANDB_RUN_NAME": wandb_run_name,
        }
    )

    wandb_args = ["--wandb_debug"]
    if is_true(values["ENABLE_WANDB"]):
        if not values["WANDB_API_KEY"]:
            raise ValueError("ENABLE_WANDB=true but WANDB_API_KEY is empty.")
        wandb_args = ["--wandb_project", values["WANDB_PROJECT"], "--wandb_run_name", wandb_run_name]
        if values["WANDB_ENTITY"]:
            wandb_args.extend(["--wandb_entity", values["WANDB_ENTITY"]])

    extra_args = []
    if values["WARMUP_MIN_LR"]:
        extra_args.extend(["--warmup_min_lr", values["WARMUP_MIN_LR"]])
    for env_key, cli_flag in (
        ("EARLY_EXIT", "--early_exit"),
        ("TRAIN_EXIT_RANDOM_LAYER", "--train_exit_random_layer"),
        ("FT_CONNECTOR", "--ft_connector"),
        ("FT_VIT", "--ft_vit"),
        ("FT_LLM", "--ft_llm"),
    ):
        if is_true(values[env_key]):
            extra_args.append(cli_flag)
    if values["FT_EMBEDDING"]:
        extra_args.extend(["--ft_embedding", values["FT_EMBEDDING"]])

    cmd = [
        "torchrun",
        "--nnodes=1",
        "--node-rank=0",
        "--master-addr=127.0.0.1",
        f"--nproc-per-node={nproc}",
        f"--master-port={get_free_port()}",
        "launch_scripts/train_vla.py",
        "qwen2_7b",
        "--checkpoint",
        str(pretrain_checkpoint),
        "--vision_backbone",
        "openai",
        "--vla_config_path",
        "xpolicylab_runtime.yaml",
        *wandb_args,
        "--train_steps",
        values["TRAIN_STEPS"],
        "--save_interval",
        values["SAVE_INTERVAL"],
        "--save_interval_unsharded",
        values["SAVE_INTERVAL_UNSHARDED"],
        "--save_num_checkpoints_to_keep",
        values["SAVE_NUM_CHECKPOINTS_TO_KEEP"],
        "--save_num_unsharded_checkpoints_to_keep",
        values["SAVE_NUM_UNSHARDED_CHECKPOINTS_TO_KEEP"],
        "--global_batch_size",
        values["GLOBAL_BATCH_SIZE"],
        "--device_train_microbatch_size",
        values["DEVICE_TRAIN_MICROBATCH_SIZE"],
        "--num_workers",
        values["NUM_WORKERS"],
        "--seq_len",
        values["SEQ_LEN"],
        "--crop_mode",
        values["CROP_MODE"],
        "--max_crops",
        values["MAX_CROPS"],
        "--connector_learning_rate",
        values["CONNECTOR_LR"],
        "--vit_learning_rate",
        values["VIT_LR"],
        "--llm_learning_rate",
        values["LLM_LR"],
        "--action_head_learning_rate",
        values["ACTION_HEAD_LR"],
        "--connector_weight_decay",
        values["CONNECTOR_WEIGHT_DECAY"],
        "--vit_weight_decay",
        values["VIT_WEIGHT_DECAY"],
        "--llm_weight_decay",
        values["LLM_WEIGHT_DECAY"],
        "--action_head_weight_decay",
        values["ACTION_HEAD_WEIGHT_DECAY"],
        "--adam_beta1",
        values["ADAM_BETA1"],
        "--adam_beta2",
        values["ADAM_BETA2"],
        "--warmup_steps",
        values["WARMUP_STEPS"],
        "--freeze_steps",
        values["FREEZE_STEPS"],
        "--scheduler_alpha_f",
        values["SCHEDULER_ALPHA_F"],
        *extra_args,
        "--log_interval",
        values["LOG_INTERVAL"],
        f"--seed={args.seed}",
        f"--save_folder={ckpt_dir}",
        "--save_overwrite",
    ]

    print(f"[INFO] GPU ID (to use): {args.gpu_id}")
    print(f"[INFO] XPolicyLab action dim: {action_dim}")
    print(f"[INFO] Training config: {config_file}")
    print(f"[INFO] DATASET_PATH: {dataset_path}")
    print(f"[INFO] PRETRAIN_CHECKPOINT: {pretrain_checkpoint}")
    print(f"[INFO] Runtime dataset config: {dataset_cfg}")
    print(f"[INFO] Runtime experiment config: {experiment_cfg}")
    print(f"[INFO] RUNNAME: {runname}")
    print(f"[INFO] SEQ_LEN: {values['SEQ_LEN']}, MAX_CROPS: {values['MAX_CROPS']}, NUM_WORKERS: {values['NUM_WORKERS']}")
    print(f"[INFO] Command: {' '.join(shlex.quote(part) for part in cmd)}")
    stats_path = _write_dataset_stats(dataset_path, ckpt_dir)
    _clean_checkpoint_artifacts(ckpt_dir)
    print(f"[INFO] Dataset stats saved to: {stats_path}")
    env["DATASET_STATS_PATH"] = str(stats_path)

    log_file = ckpt_dir / "log" / f"training_{subprocess.check_output(['date', '+%Y%m%d_%H%M'], text=True).strip()}.txt"
    with open(log_file, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=a1_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    stats_path = _write_dataset_stats(dataset_path, ckpt_dir)
    _clean_checkpoint_artifacts(ckpt_dir)
    print(f"[INFO] Dataset stats refreshed at: {stats_path}")
    print(f"[INFO] Training complete. Checkpoints saved to: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num")
    parser.add_argument("action_type")
    parser.add_argument("seed")
    parser.add_argument("gpu_id")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
