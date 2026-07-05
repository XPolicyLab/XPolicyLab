"""
XPolicyLab Planning Module training orchestrator (not a LeRobot data converter).

Invoked by ../../train.sh (train_module=planning|both). Runs prepare -> copy -> train -> merge using
upstream llamafactory_data_preparation.py and LLaMA-Factory, with run-scoped
paths, seed/data_seed in train YAML, and per-run config files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ADAPTER_DIR = Path(__file__).resolve().parent
MEM0_ROOT = ADAPTER_DIR.parent
DATA_PREP_SCRIPT = MEM0_ROOT / "scripts" / "llama_data_preparation" / "llamafactory_data_preparation.py"

BASE_MODEL_DIR_NAME = "Qwen3-VL-8B-Instruct"
TEMPLATE = "qwen3_vl_nothink"


def get_bench_name(lerobot_dataset_path: str) -> str:
    return Path(lerobot_dataset_path).name or "dataset"


def get_json_filename(bench_name: str) -> str:
    return f"{bench_name}_high_level_finetune_data.json"


def get_images_folder_name(bench_name: str) -> str:
    return f"{bench_name}_images"


def prepare_json_path(bench_name: str) -> Path:
    return MEM0_ROOT / "llamafactory_data" / bench_name / get_json_filename(bench_name)


def prepare_meta_path(json_path: Path) -> Path:
    return json_path.with_suffix(json_path.suffix + ".meta")


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in config.items():
            if v is None:
                f.write(f"{k}: null\n")
            elif isinstance(v, bool):
                f.write(f"{k}: {str(v).lower()}\n")
            else:
                f.write(f"{k}: {v}\n")


def write_merge_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None, env_name: str | None = None) -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env_name:
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}", file=sys.stderr)
        sys.exit(e.returncode)


def prepare_meta_matches(cfg: dict, meta_path: Path) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    repo = str(Path(cfg["lerobot_dataset_path"]).resolve())
    return (
        meta.get("episode_start_id") == cfg["episode_start_id"]
        and meta.get("episode_end_id") == cfg["episode_end_id"]
        and meta.get("lerobot_dataset_path") == repo
    )


def write_prepare_meta(cfg: dict, json_path: Path) -> None:
    meta = {
        "episode_start_id": cfg["episode_start_id"],
        "episode_end_id": cfg["episode_end_id"],
        "lerobot_dataset_path": str(Path(cfg["lerobot_dataset_path"]).resolve()),
    }
    prepare_meta_path(json_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def step_prepare(cfg: dict) -> str:
    lerobot_path = cfg["lerobot_dataset_path"]
    if not lerobot_path or not Path(lerobot_path).is_dir():
        raise SystemExit(
            "[planning] lerobot_dataset_path is missing or not a directory. "
            "Run process_data.sh ... Mn first."
        )
    bench_name = get_bench_name(lerobot_path)
    json_path = prepare_json_path(bench_name)
    force = cfg.get("force_prepare", False)

    if json_path.is_file() and not force:
        if prepare_meta_matches(cfg, prepare_meta_path(json_path)):
            print(f"[prepare] Output exists and meta matches; skipping: {json_path}")
            return bench_name
        raise SystemExit(
            f"[planning] Prepare output exists but episode/repo metadata differs: {json_path}\n"
            "Set FORCE_PREPARE=true to regenerate, or remove llamafactory_data/ for this dataset."
        )

    if cfg.get("dry_run"):
        print(f"[prepare] DRY_RUN: would run data prep -> {json_path}")
        return bench_name

    cmd = [
        sys.executable,
        str(DATA_PREP_SCRIPT),
        "--lerobot_dataset_path",
        str(Path(lerobot_path).resolve()),
        "--episode_start_id",
        str(cfg["episode_start_id"]),
        "--episode_end_id",
        str(cfg["episode_end_id"]),
    ]
    print("[prepare] Running llamafactory_data_preparation.py (mem0 env)...")
    run_cmd(cmd, cwd=MEM0_ROOT, env_name=cfg.get("conda_env_mem0"))
    if not json_path.is_file():
        raise SystemExit(f"[planning] Data prep did not produce {json_path}")
    write_prepare_meta(cfg, json_path)
    print(f"[prepare] Done: {json_path}")
    return bench_name


def step_copy(cfg: dict, bench_name: str) -> None:
    if cfg.get("dry_run"):
        print(f"[copy] DRY_RUN: would copy dataset '{bench_name}' to LLaMA-Factory/data")
        return

    lf_root = Path(cfg["llamafactory_root"])
    if not lf_root.is_dir():
        raise SystemExit(
            f"[planning] llamafactory_root not found: {lf_root}\n"
            "Run: bash install_planning.sh"
        )
    data_dir = lf_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    src = MEM0_ROOT / "llamafactory_data" / bench_name
    json_name = get_json_filename(bench_name)
    images_name = get_images_folder_name(bench_name)
    src_json = src / json_name
    src_images = src / images_name
    if not src_json.is_file():
        raise SystemExit(f"[planning] Missing {src_json}. Run step 'prepare' first.")

    shutil.copy2(src_json, data_dir / json_name)
    dst_images = data_dir / images_name
    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)
    print(f"[copy] Copied {json_name} and {images_name} -> {data_dir}")

    info_path = data_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info[bench_name] = {
        "file_name": json_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[copy] Updated {info_path} with dataset '{bench_name}'.")


def _adapter_dir(cfg: dict) -> Path:
    return Path(cfg["adapter_output_dir"]).resolve()


def _export_dir(cfg: dict) -> Path:
    if cfg.get("export_dir"):
        return Path(cfg["export_dir"]).resolve()
    return Path(cfg["merged_output_dir"]).resolve()


def _run_config_dir(cfg: dict) -> Path:
    return Path(cfg["run_config_dir"]).resolve()


def step_train(cfg: dict, bench_name: str) -> str:
    base_out = Path(cfg["base_output_dir"]).resolve()
    lf_root = Path(cfg["llamafactory_root"]).resolve()
    base_model_path = base_out / BASE_MODEL_DIR_NAME
    if not base_model_path.is_dir():
        raise SystemExit(
            f"[planning] Base model dir not found: {base_model_path}\n"
            "Download Qwen3-VL-8B-Instruct (cd Mem_0/checkpoints && python _download.py)."
        )

    output_dir = _adapter_dir(cfg)
    run_dir = _run_config_dir(cfg)
    train_yaml = run_dir / "planning_train.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    train_config = {
        "model_name_or_path": str(base_model_path),
        "image_max_pixels": cfg.get("image_max_pixels", 131072),
        "video_max_pixels": cfg.get("video_max_pixels", 16384),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_target": "all",
        "dataset": bench_name,
        "template": TEMPLATE,
        "cutoff_len": cfg.get("cutoff_len", 4096),
        "max_samples": cfg.get("max_samples", 1000),
        "overwrite_cache": True,
        "preprocessing_num_workers": cfg.get("preprocessing_num_workers", 16),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 8),
        "output_dir": str(output_dir),
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": cfg.get("report_to", "wandb"),
        "run_name": cfg.get("run_name", ""),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 16),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": cfg.get("learning_rate", 1.0e-4),
        "num_train_epochs": cfg.get("num_train_epochs", 25),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
        "seed": seed,
        "data_seed": seed,
    }
    if cfg.get("full_determinism"):
        train_config["full_determinism"] = True

    write_yaml(train_yaml, train_config)
    print(f"[train] Wrote {train_yaml}")
    print(
        f"[train]   seed={seed} data_seed={seed} output_dir={output_dir} "
        f"run_name={cfg.get('run_name', '')}"
    )

    if cfg.get("dry_run"):
        print("[train] DRY_RUN: skipping llamafactory-cli train")
        return str(output_dir)

    cmd = ["llamafactory-cli", "train", str(train_yaml)]
    print("[train] Running LLaMA-Factory train (llama_factory env)...")
    env_extra: dict[str, str] = {}
    run_name = cfg.get("run_name")
    if run_name and cfg.get("report_to") == "wandb":
        env_extra["WANDB_RUN_NAME"] = str(run_name)

    env_name = cfg.get("conda_env_llamafactory")
    if env_name:
        wrapped = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **env_extra}
        subprocess.run(wrapped, cwd=lf_root, check=True, env=env)
    else:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **env_extra}
        subprocess.run(cmd, cwd=lf_root, check=True, env=env)

    print(f"[train] Done. Adapter: {output_dir}")
    return str(output_dir)


def step_merge(cfg: dict, adapter_path: str) -> None:
    export_dir = _export_dir(cfg)
    lf_root = Path(cfg["llamafactory_root"]).resolve()
    base_model_path = Path(cfg["base_output_dir"]).resolve() / BASE_MODEL_DIR_NAME
    run_dir = _run_config_dir(cfg)
    merge_yaml = run_dir / "planning_merge.yaml"

    adapter_resolved = Path(adapter_path).resolve()
    if not adapter_resolved.is_dir() and not cfg.get("dry_run"):
        raise SystemExit(f"[planning] LoRA adapter not found: {adapter_resolved}")

    merge_lines = [
        "# DO NOT use quantized model or quantization_bit when merging lora adapters",
        "model_name_or_path: " + str(base_model_path),
        "adapter_name_or_path: " + str(adapter_resolved),
        "template: " + TEMPLATE,
        "trust_remote_code: true",
        "export_dir: " + str(export_dir),
        "export_size: " + str(cfg.get("export_size", 5)),
        "export_device: " + cfg.get("export_device", "cpu"),
        "export_legacy_format: false",
    ]
    write_merge_yaml(merge_yaml, merge_lines)
    print(f"[merge] Wrote {merge_yaml}")
    print(f"[merge]   adapter={adapter_resolved} export_dir={export_dir}")

    if cfg.get("dry_run"):
        print("[merge] DRY_RUN: skipping llamafactory-cli export")
        return

    cmd = ["llamafactory-cli", "export", str(merge_yaml)]
    print("[merge] Running LLaMA-Factory export (llama_factory env)...")
    run_cmd(cmd, cwd=lf_root, env_name=cfg.get("conda_env_llamafactory"))
    print(f"[merge] Done. Merged model: {export_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mem_0 Planning Module train orchestrator (XPolicyLab)")
    p.add_argument(
        "--steps",
        nargs="+",
        default=["prepare", "copy", "train", "merge"],
        choices=["prepare", "copy", "train", "merge"],
    )
    p.add_argument("--no-conda-run", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs only; skip LLaMA-Factory train/export",
    )
    p.add_argument("--force-prepare", action="store_true")

    p.add_argument("--lerobot_dataset_path", required=True)
    p.add_argument("--llamafactory_root", required=True)
    p.add_argument("--base_output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--run_config_dir", required=True)
    p.add_argument("--adapter_output_dir", required=True)
    p.add_argument("--merged_output_dir", required=True)
    p.add_argument("--export_dir", default="")
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--episode_start_id", type=int, default=0)
    p.add_argument("--episode_end_id", type=int, required=True)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--num_train_epochs", type=int, default=25)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--report_to", default="wandb")
    p.add_argument("--cutoff_len", type=int, default=4096)
    p.add_argument("--image_max_pixels", type=int, default=131072)
    p.add_argument("--video_max_pixels", type=int, default=16384)
    p.add_argument("--preprocessing_num_workers", type=int, default=16)
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--export_size", type=int, default=5)
    p.add_argument("--export_device", default="cpu")
    p.add_argument("--full-determinism", action="store_true")
    p.add_argument("--conda_env_mem0", default="mem0")
    p.add_argument("--conda_env_llamafactory", default="llama_factory")
    return p


def main() -> None:
    args = build_parser().parse_args()
    enable_wandb = os.environ.get("ENABLE_WANDB", "true").strip().lower() not in {"0", "false", "no"}
    cfg: dict[str, Any] = {
        "lerobot_dataset_path": args.lerobot_dataset_path,
        "llamafactory_root": args.llamafactory_root,
        "base_output_dir": args.base_output_dir,
        "run_name": args.run_name,
        "run_config_dir": args.run_config_dir,
        "adapter_output_dir": args.adapter_output_dir,
        "merged_output_dir": args.merged_output_dir,
        "export_dir": args.export_dir or "",
        "seed": args.seed,
        "episode_start_id": args.episode_start_id,
        "episode_end_id": args.episode_end_id,
        "max_samples": args.max_samples,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "report_to": args.report_to if enable_wandb else "none",
        "cutoff_len": args.cutoff_len,
        "image_max_pixels": args.image_max_pixels,
        "video_max_pixels": args.video_max_pixels,
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "dataloader_num_workers": args.dataloader_num_workers,
        "export_size": args.export_size,
        "export_device": args.export_device,
        "full_determinism": args.full_determinism,
        "dry_run": args.dry_run,
        "force_prepare": args.force_prepare,
    }
    if args.no_conda_run:
        cfg["conda_env_mem0"] = None
        cfg["conda_env_llamafactory"] = None
    else:
        cfg["conda_env_mem0"] = args.conda_env_mem0
        cfg["conda_env_llamafactory"] = args.conda_env_llamafactory

    bench_name: str | None = None
    adapter_path: str | None = None

    if "prepare" in args.steps:
        bench_name = step_prepare(cfg)
    if "copy" in args.steps:
        if bench_name is None:
            bench_name = get_bench_name(cfg["lerobot_dataset_path"])
        step_copy(cfg, bench_name)
    if "train" in args.steps:
        if bench_name is None:
            bench_name = get_bench_name(cfg["lerobot_dataset_path"])
        adapter_path = step_train(cfg, bench_name)
    if "merge" in args.steps:
        if adapter_path is None:
            adapter_path = str(_adapter_dir(cfg))
        step_merge(cfg, adapter_path)

    print("All requested steps completed.")


if __name__ == "__main__":
    main()
