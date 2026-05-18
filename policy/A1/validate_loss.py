#!/usr/bin/env python3
"""Validate an A1 checkpoint on XPolicyLab data by computing action loss."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
A1_DIR = SCRIPT_DIR / "A1"
ROOT_DIR = SCRIPT_DIR.parent.parent.parent
os.environ.setdefault("DATA_DIR", "/mnt/pfs/pg4hw0/qiwei/models")
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
if str(A1_DIR) not in sys.path:
    sys.path.insert(0, str(A1_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from a1.data import build_mm_preprocessor  # noqa: E402
from a1.data.collator import MMCollatorForAction  # noqa: E402
from a1.data.dataset import DeterministicDataset  # noqa: E402
from a1.data.vla.lerobot_datasets import LeRobotDatasetWrapper  # noqa: E402
from a1.data.vla.utils import NormalizationType  # noqa: E402
from a1.torch_util import seed_all  # noqa: E402

from model import _load_a1_model  # noqa: E402
from process_data import create_empty_dataset, load_data  # noqa: E402
from XPolicyLab.utils.load_file import load_json, load_yaml  # noqa: E402


def _find_hdf5_files(data_path: Path, max_episodes: int | None) -> list[Path]:
    candidates = []
    for base in (data_path / "data", data_path):
        if base.is_dir():
            candidates = sorted(base.glob("episode_*.hdf5"))
            if candidates:
                break
    if not candidates:
        raise FileNotFoundError(f"No episode_*.hdf5 found under {data_path} or {data_path / 'data'}")
    return candidates[:max_episodes] if max_episodes else candidates


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta").is_dir() and ((path / "data").is_dir() or (path / "videos").is_dir())


def _convert_to_lerobot(args, robot_action_dim_info: dict, robot_type: str) -> Path:
    data_path = Path(args.data_path).resolve()
    if _is_lerobot_dataset(data_path):
        return data_path

    ep_files = _find_hdf5_files(data_path, args.max_episodes)
    repo_id = args.repo_id or f"val-loss-{args.task_name}-{args.env_cfg_type}-{args.action_type}-{len(ep_files)}eps"
    output_root = Path(args.output_dir).resolve()
    output_path = output_root / repo_id
    if output_path.exists():
        if args.rebuild:
            shutil.rmtree(output_path)
        else:
            return output_path

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=args.fps,
        mode=args.mode,
        robot_action_dim_info=robot_action_dim_info,
        root=str(output_root),
    )

    for ep_file in tqdm(ep_files, desc="Converting episodes"):
        ep = load_data(ep_file, args.action_type, robot_action_dim_info)
        frames = ep["state"].shape[0]
        for i in range(frames):
            instruction = args.instruction
            if ep["instructions"]:
                instruction = ep["instructions"][min(i, len(ep["instructions"]) - 1)]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")

            frame = {
                "observation.state": ep["state"][i],
                "action": ep["action"][i],
            }
            for camera_name, imgs in ep["images"].items():
                if i < len(imgs):
                    frame[camera_name] = imgs[i]
            dataset.add_frame(frame, task=str(instruction))
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()

    return output_path


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _action_loss(outputs, batch: dict, action_head: str) -> torch.Tensor:
    def one_loss(output):
        if action_head == "l1_regression":
            pred = output["predicted_actions"]
            target = batch["action"]
            valid_mask = (~batch["action_pad_mask"].bool()).to(device=pred.device)
            loss_elems = F.l1_loss(pred, target, reduction="none")
        else:
            pred = output["diffusion_pred"]
            target = output["diffusion_target"]
            valid_mask = (~batch["action_pad_mask"].bool()).to(device=pred.device)
            loss_elems = F.mse_loss(pred, target, reduction="none")

        if valid_mask.shape != target.shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} does not match target shape {target.shape}")
        return (loss_elems * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    if isinstance(outputs, list):
        return torch.stack([one_loss(output) for output in outputs]).mean()
    return one_loss(outputs)


def _forward_loss(model, model_cfg, batch: dict, device: torch.device) -> torch.Tensor:
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast("cuda", enabled=autocast_enabled, dtype=torch.bfloat16):
        outputs = model.forward(
            input_ids=batch["input_ids"],
            target_actions=batch.get("action"),
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
            images=batch.get("images"),
            image_masks=batch.get("image_masks"),
            image_input_idx=batch.get("image_input_idx"),
            subsegment_ids=batch.get("subsegment_ids"),
            position_ids=batch.get("position_ids"),
            action_proprio=batch.get("proprio"),
            proprio_token_idx=batch["proprio_token_idx"],
            output_hidden_states=False,
            train_exit_random_layer=None,
            use_cache=True if model_cfg.action_head == "flow_matching" else False,
        )
    return _action_loss(outputs, batch, model_cfg.action_head)


def main():
    parser = argparse.ArgumentParser(description="Compute A1 checkpoint validation action loss.")
    parser.add_argument("--data_path", required=True, help="Raw XPolicyLab data dir or an existing LeRobot dataset dir.")
    parser.add_argument("--ckpt", required=True, help="A1 checkpoint directory containing model.pt and config.yaml.")
    parser.add_argument("--env_cfg_type", required=True, help="For example: arx_x5.")
    parser.add_argument("--task_name", default="stack_bowls")
    parser.add_argument("--action_type", choices=["joint", "ee"], default="ee")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default="Do your job.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / ".cache" / "val_loss_lerobot"))
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--normalization_type", default=None, choices=["bounds", "normal"])
    parser.add_argument("--no_wrist_image", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)

    env_cfg = load_yaml(ROOT_DIR / "env_cfg" / f"{args.env_cfg_type}.yml")
    robot_type = env_cfg["config"]["robot"]
    robot_action_dim_info = load_json(ROOT_DIR / "env_cfg" / "robot" / "_robot_info.json")[robot_type]
    lerobot_path = _convert_to_lerobot(args, robot_action_dim_info, robot_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    model, model_cfg = _load_a1_model(str(Path(args.ckpt).resolve()), args.seed)
    model.to(device)
    model.eval()

    if args.normalization_type:
        norm_type = NormalizationType(args.normalization_type)
    else:
        norm_type = None

    dataset = LeRobotDatasetWrapper(
        str(lerobot_path),
        chunk_size=model_cfg.num_actions_chunk,
        fixed_action_dim=model_cfg.fixed_action_dim,
        normalization_type=norm_type,
        use_proprio=True,
        use_wrist_image=not args.no_wrist_image,
        image_aug=False,
    )
    preprocessor = build_mm_preprocessor(
        model_cfg,
        for_inference=False,
        shuffle_messages=False,
        is_training=False,
        require_image_features=True,
    )
    processed = DeterministicDataset(dataset, preprocessor, args.seed)
    loader = DataLoader(
        processed,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=MMCollatorForAction(
            model_config=model_cfg,
            use_proprio=True,
            max_sequence_length=args.sequence_length,
            include_metadata=True,
            pad="to_max",
            max_crops=model_cfg.get_max_crops(),
        ),
    )

    losses = []
    for step, batch in enumerate(tqdm(loader, desc="Validating", total=min(len(loader), args.max_batches))):
        if step >= args.max_batches:
            break
        batch = _move_to_device(batch, device)
        loss = _forward_loss(model, model_cfg, batch, device)
        losses.append(float(loss.detach().cpu()))
        print(f"[VAL] batch={step} action_loss={losses[-1]:.6f}")

    if not losses:
        raise RuntimeError("No validation batches were produced.")

    summary = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "lerobot_path": str(lerobot_path),
        "num_batches": len(losses),
        "mean_action_loss": float(np.mean(losses)),
        "std_action_loss": float(np.std(losses)),
        "min_action_loss": float(np.min(losses)),
        "max_action_loss": float(np.max(losses)),
    }
    print("[VAL_SUMMARY] " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
