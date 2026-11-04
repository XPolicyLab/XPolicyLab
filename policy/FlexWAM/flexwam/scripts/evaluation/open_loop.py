"""Offline open-loop evaluation.

python scripts/evaluation/open_loop.py \
    --task <task> \
    --checkpoint-path /path/to/step.pt \
    --dataset-stats-path /path/to/dataset_stats.json \
    --split val \
    --episode-index 0 \
    --num-episodes 1 \
    --num-frames 200 \
    --episode-mode full \
    --num-inference-steps 20 \
    --skip-training-loss \
    --output-dir /path/to/output
"""


from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flexwam.utils.tmpdir import configure_tmpdir  # noqa: E402

configure_tmpdir()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from flexwam.utils.camera_metrics import configure_camera_metrics
from flexwam.utils.config_resolvers import register_default_resolvers
from flexwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from flexwam.models.backbone.wan22.loader import _load_registered_model, _resolve_configs
from flexwam.models.backbone.wan22.text_encoder import HuggingfaceTokenizer


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
JOINT_NAMES = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
]
BIMANUAL_EE_POSE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_gripper",
]


def _action_dim_names(action_dim: int) -> list[str]:
    if action_dim == len(JOINT_NAMES):
        return list(JOINT_NAMES)
    if action_dim == len(BIMANUAL_EE_POSE_NAMES):
        return list(BIMANUAL_EE_POSE_NAMES)
    return [f"dim_{index}" for index in range(action_dim)]


def _normalize_mixed_precision(mixed_precision: str) -> str:
    value = str(mixed_precision).strip().lower()
    if value not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed precision `{mixed_precision}`. "
            "Expected one of: no, fp16, bf16."
        )
    return value


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _autocast_context(device: str, dtype: torch.dtype):
    if str(device).startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _resolve_existing_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_dataset_stats_path(
    dataset_stats_path: str | None,
    checkpoint_path: Path,
) -> Path:
    if dataset_stats_path:
        return _resolve_existing_path(dataset_stats_path, "Dataset stats")

    candidate_paths = [
        checkpoint_path.parents[2] / "dataset_stats.json",
        checkpoint_path.parent / "dataset_stats.json",
        checkpoint_path.parent.parent / "dataset_stats.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            logger.info("Using inferred dataset stats path: %s", candidate)
            return candidate.resolve()

    raise FileNotFoundError(
        "Dataset stats path was not provided and could not be inferred from checkpoint path. "
        "Please pass --dataset-stats-path explicitly."
    )


def _compose_task_cfg(task_name: str) -> DictConfig:
    from flexwam.config import compose_task
    return compose_task(task_name)


def _resolve_project_relative_path(path_like: str | None) -> str | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((PROJECT_ROOT / path).resolve())


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    prompts: list[str] = []
    seen = set()
    for dataset_dir in dataset_dirs:
        tasks_path = Path(dataset_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = str(json.loads(line)["task"])
                prompt = DEFAULT_PROMPT.format(task=task)
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    return prompts


def _ensure_text_embedding_cache(cfg: DictConfig, device: str, model_dtype: torch.dtype, overwrite: bool = False) -> None:
    data_nodes = [cfg.data.train, cfg.data.val]
    cache_dirs: list[Path] = []
    dataset_dirs: list[str] = []
    context_lens = set()
    for node in data_nodes:
        for dataset_dir in node.dataset_dirs:
            dataset_dir = _resolve_project_relative_path(str(dataset_dir))
            if dataset_dir not in dataset_dirs:
                dataset_dirs.append(dataset_dir)
        cache_dir = Path(_resolve_project_relative_path(str(node.text_embedding_cache_dir)))
        if cache_dir not in cache_dirs:
            cache_dirs.append(cache_dir)
        context_lens.add(int(node.context_len))

    if len(context_lens) != 1:
        raise ValueError(f"Expected one context_len, got {sorted(context_lens)}")
    context_len = next(iter(context_lens))
    prompts = _read_unique_prompts(dataset_dirs)
    if not prompts:
        logger.warning("No prompts found in dataset tasks.jsonl; skipping text embedding cache check.")
        return

    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    enc_id = _model_id_to_enc_id(model_id)
    missing: list[tuple[Path, str, Path]] = []
    for prompt in prompts:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
        for cache_dir in cache_dirs:
            cache_path = cache_dir / filename
            if overwrite or not cache_path.exists():
                missing.append((cache_dir, prompt, cache_path))

    if not missing:
        logger.info("Text embedding cache is complete for %d prompt(s).", len(prompts))
        return

    logger.info(
        "Text embedding cache missing %d file(s); generating with Wan T5 encoder.",
        len(missing),
    )
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    text_dtype = torch.bfloat16 if model_dtype in {torch.float16, torch.bfloat16} else torch.float32
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=text_dtype,
        device=text_device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    prompt_to_cache_dirs: dict[str, set[Path]] = {}
    for cache_dir, prompt, _ in missing:
        prompt_to_cache_dirs.setdefault(prompt, set()).add(cache_dir)

    with torch.no_grad():
        for prompt, prompt_cache_dirs in prompt_to_cache_dirs.items():
            ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
            ids = ids.to(text_device)
            mask = mask.to(device=text_device, dtype=torch.bool)
            context = text_encoder(ids, mask)[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            mask_cpu = mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous()
            payload = {"context": context, "mask": mask_cpu}
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            for cache_dir in prompt_cache_dirs:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / filename
                torch.save(payload, cache_path)
                logger.info("Saved text embedding cache: %s", cache_path)


def _build_model(cfg: DictConfig, checkpoint_path: Path, device: str, model_dtype: torch.dtype):
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.load_text_encoder = False
    if "action_dit_pretrained_path" in model_cfg and model_cfg.action_dit_pretrained_path is not None:
        model_cfg.action_dit_pretrained_path = _resolve_project_relative_path(
            model_cfg.action_dit_pretrained_path
        )

    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    configure_camera_metrics(model, cfg.data.train)
    model.load_checkpoint(str(checkpoint_path))
    model = model.to(device).eval()
    return model


def _build_dataset(cfg: DictConfig, split: str, dataset_stats_path: Path):
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split `{split}`. Expected `train` or `val`.")
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[split], resolve=True))
    dataset_cfg.pretrained_norm_stats = str(dataset_stats_path)
    if "text_embedding_cache_dir" in dataset_cfg and dataset_cfg.text_embedding_cache_dir is not None:
        dataset_cfg.text_embedding_cache_dir = _resolve_project_relative_path(
            dataset_cfg.text_embedding_cache_dir
        )
    dataset = instantiate(dataset_cfg)
    return dataset


def _prepare_eval_video(video: torch.Tensor) -> tuple[torch.Tensor, int, str]:
    """Add a batch dim and report the video frame count.

    Single-view samples are `[C,T,H,W]` or `[B,C,T,H,W]`.
    Independent multi-view samples are `[T,V,C,H,W]` or `[B,T,V,C,H,W]`.
    """
    if video.ndim == 4 and int(video.shape[0]) == 3:
        return video.unsqueeze(0), int(video.shape[1]), "cthw"
    # Dataset samples are unbatched. Independent multi-view is [T,V,3,H,W],
    # which also has shape[1]==3 when there are three cameras, so check channels first.
    if video.ndim == 5 and int(video.shape[2]) == 3:
        return video.unsqueeze(0), int(video.shape[0]), "tvchw"
    if video.ndim == 5 and int(video.shape[1]) == 3:
        return video, int(video.shape[2]), "cthw"
    if video.ndim == 6 and int(video.shape[3]) == 3:
        return video, int(video.shape[1]), "tvchw"
    raise ValueError(
        "Expected video `[C,T,H,W]`, `[B,C,T,H,W]`, `[T,V,C,H,W]`, or `[B,T,V,C,H,W]`, "
        f"got {tuple(video.shape)}"
    )


def _to_batched_eval_sample(sample: dict[str, Any]) -> dict[str, Any]:
    video = sample["video"]
    prompt = sample["prompt"]
    action = sample.get("action", None)
    proprio = sample.get("proprio", None)
    context = sample.get("context", None)
    context_mask = sample.get("context_mask", None)

    if not isinstance(video, torch.Tensor):
        raise TypeError(
            f"Expected tensor video for evaluation, got {type(video)}. "
            "Evaluation expects `video` with shape [3,T,H,W], [B,3,T,H,W], or [T,V,3,H,W]."
        )
    video, num_video_frames, video_layout = _prepare_eval_video(video)
    if num_video_frames <= 1:
        raise ValueError(f"`sample['video']` must have at least 2 frames, got {num_video_frames}")

    if isinstance(prompt, str):
        prompt = [prompt]
    elif isinstance(prompt, tuple):
        prompt = list(prompt)
    elif not isinstance(prompt, list):
        raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
    if len(prompt) != video.shape[0]:
        raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")

    action_horizon = None
    if action is not None:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"`sample['action']` must be a torch.Tensor, got {type(action)}")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got {tuple(action.shape)}")
        if action.shape[1] % (num_video_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video frames-1="
                f"{num_video_frames - 1}, got {action.shape[1]}"
            )
        action_horizon = int(action.shape[1])

    if proprio is not None:
        if not isinstance(proprio, torch.Tensor):
            raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 3:
            raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got {tuple(proprio.shape)}")

    if context is not None or context_mask is not None:
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist in eval sample.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )

    return {
        "video": video,
        "video_layout": video_layout,
        "prompt": prompt,
        "action": action,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "action_horizon": action_horizon,
    }


def _denormalize_actions(
    processor,
    proprio: torch.Tensor,
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
    denorm_actions: dict[str, torch.Tensor] = {}
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]

    for action_name, raw_action in (("pred", pred_action), ("gt", gt_action)):
        if not isinstance(raw_action, torch.Tensor):
            raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
        if raw_action.ndim == 2:
            action_btd = raw_action.unsqueeze(0)
        elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
            action_btd = raw_action
        else:
            raise ValueError(
                f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
            )
        action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

        batch = {
            "action": action_btd,
            "state": proprio,
        }
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        merged_batch = {
            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
        }
        merged_batch = processor.action_state_merger.forward(merged_batch)
        denorm_action = merged_batch["action"].unsqueeze(0)
        if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
            raise ValueError(
                f"Denormalized {action_name} action must have shape [1, T, D], "
                f"got {tuple(denorm_action.shape)}"
            )
        denorm_actions[action_name] = denorm_action

    pred_np = denorm_actions["pred"][0].numpy().astype(np.float32, copy=False)
    gt_np = denorm_actions["gt"][0].numpy().astype(np.float32, copy=False)
    return pred_np, gt_np


def _evaluate_dataset_sample(
    *,
    model,
    dataset,
    sample_index: int,
    num_inference_steps: int,
    infer_seed: int | None,
    compute_training_loss: bool,
    tiled: bool,
):
    sample = dataset[sample_index]
    batched_sample = _to_batched_eval_sample(sample)
    prompt = batched_sample["prompt"][0]
    gt_action = batched_sample["action"]
    proprio = batched_sample["proprio"]
    if batched_sample["video_layout"] == "tvchw":
        video = batched_sample["video"][0]
        input_image = video[0]
        num_frames = int(video.shape[0])
    else:
        video = batched_sample["video"][0]
        input_image = video[:, 0].unsqueeze(0)
        num_frames = int(video.shape[1])

    infer_kwargs = {
        "prompt": prompt,
        "input_image": input_image,
        "action_horizon": batched_sample["action_horizon"],
        "proprio": proprio[0, 0] if proprio is not None else None,
        "num_inference_steps": num_inference_steps,
        "seed": infer_seed,
        "tiled": tiled,
    }
    if batched_sample["context"] is not None:
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = batched_sample["context"][0]
        infer_kwargs["context_mask"] = batched_sample["context_mask"][0]

    training_loss = None
    with torch.no_grad():
        with _autocast_context(str(model.device), model.torch_dtype):
            if compute_training_loss:
                training_loss, _ = model.training_loss(batched_sample)
                training_loss = float(training_loss.detach().float().item())
            if hasattr(model, "infer_action"):
                if "num_video_frames" in inspect.signature(model.infer_action).parameters:
                    infer_kwargs["num_video_frames"] = num_frames
                pred = model.infer_action(**infer_kwargs)
            else:
                infer_kwargs["num_frames"] = num_frames
                infer_kwargs["action"] = gt_action[0] if gt_action is not None else None
                infer_kwargs["text_cfg_scale"] = 1.0
                infer_kwargs["action_cfg_scale"] = 1.0
                pred = model.infer(**infer_kwargs)

    pred_action = pred.get("action", None)
    if pred_action is None or gt_action is None or proprio is None:
        raise ValueError("Sample evaluation requires action and proprio outputs.")

    processor = dataset.lerobot_dataset.processor
    pred_denorm, gt_denorm = _denormalize_actions(
        processor=processor,
        proprio=proprio,
        pred_action=pred_action,
        gt_action=gt_action,
    )

    action_is_pad = sample.get("action_is_pad", None)
    if action_is_pad is None:
        valid_mask = np.ones(pred_denorm.shape[0], dtype=bool)
    else:
        valid_mask = ~torch.as_tensor(action_is_pad, dtype=torch.bool).cpu().numpy()

    return {
        "sample_index": int(sample_index),
        "prompt": prompt,
        "training_loss": training_loss,
        "pred_action": pred_denorm,
        "gt_action": gt_denorm,
        "valid_mask": valid_mask.astype(bool, copy=False),
    }


def _plot_joint_curves(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    output_path: Path,
    title: str,
    x_label: str,
):
    dim_names = _action_dim_names(int(pred.shape[1]))
    action_dim = len(dim_names)
    ncols = 2 if action_dim > 1 else 1
    nrows = int(np.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(8, 2.2 * nrows)), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    x = np.arange(pred.shape[0], dtype=np.int32)
    for dim, name in enumerate(dim_names):
        ax = axes[dim]
        ax.plot(x, gt[:, dim], label="GT", linewidth=1.8, color="#1f77b4")
        ax.plot(x, pred[:, dim], label="Pred", linewidth=1.4, color="#d62728")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        if dim >= action_dim - ncols:
            ax.set_xlabel(x_label)
    for ax in axes[action_dim:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_npz(path: Path, **arrays: Any) -> None:
    np.savez_compressed(path, **arrays)


def _compute_aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No sample results were collected.")

    action_dim = int(results[0]["pred_action"].shape[1])
    dim_names = _action_dim_names(action_dim)
    step_abs_sum = np.zeros(action_dim, dtype=np.float64)
    step_sq_sum = np.zeros(action_dim, dtype=np.float64)
    first_abs_sum = np.zeros(action_dim, dtype=np.float64)
    first_sq_sum = np.zeros(action_dim, dtype=np.float64)
    total_valid_steps = 0
    total_samples = 0
    training_losses: list[float] = []

    for result in results:
        pred_action = result["pred_action"]
        gt_action = result["gt_action"]
        valid_mask = result["valid_mask"]
        if valid_mask.ndim != 1:
            raise ValueError(f"Expected 1D valid mask, got shape {tuple(valid_mask.shape)}")

        if valid_mask.any():
            diff = pred_action[valid_mask] - gt_action[valid_mask]
            step_abs_sum += np.abs(diff).sum(axis=0)
            step_sq_sum += np.square(diff).sum(axis=0)
            total_valid_steps += int(diff.shape[0])

        first_diff = pred_action[0] - gt_action[0]
        first_abs_sum += np.abs(first_diff)
        first_sq_sum += np.square(first_diff)
        total_samples += 1

        if result["training_loss"] is not None:
            training_losses.append(float(result["training_loss"]))

    if total_valid_steps == 0:
        raise ValueError("All evaluated samples are padded; cannot compute aggregate metrics.")

    per_dim_mae = step_abs_sum / total_valid_steps
    per_dim_mse = step_sq_sum / total_valid_steps
    per_dim_first_mae = first_abs_sum / total_samples
    per_dim_first_mse = first_sq_sum / total_samples

    return {
        "num_samples": total_samples,
        "num_valid_action_steps": total_valid_steps,
        "mean_training_loss": float(np.mean(training_losses)) if training_losses else None,
        "full_horizon_mae": float(per_dim_mae.mean()),
        "full_horizon_mse": float(per_dim_mse.mean()),
        "first_step_mae": float(per_dim_first_mae.mean()),
        "first_step_mse": float(per_dim_first_mse.mean()),
        "per_dim_full_horizon_mae": {name: float(value) for name, value in zip(dim_names, per_dim_mae)},
        "per_dim_full_horizon_mse": {name: float(value) for name, value in zip(dim_names, per_dim_mse)},
        "per_dim_first_step_mae": {name: float(value) for name, value in zip(dim_names, per_dim_first_mae)},
        "per_dim_first_step_mse": {name: float(value) for name, value in zip(dim_names, per_dim_first_mse)},
    }


def _collect_eval_indices(dataset_len: int, num_samples: int, sample_stride: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError(f"`num_samples` must be positive, got {num_samples}")
    if sample_stride <= 0:
        raise ValueError(f"`sample_stride` must be positive, got {sample_stride}")

    indices = list(range(0, dataset_len, sample_stride))
    return indices[: min(num_samples, len(indices))]


def _get_episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    episode_data_index = dataset.lerobot_dataset.episode_data_index
    num_episodes = int(len(episode_data_index["from"]))
    if episode_index < 0 or episode_index >= num_episodes:
        raise IndexError(
            f"`episode_index` out of range: got {episode_index}, valid range [0, {num_episodes - 1}]"
        )
    start = int(episode_data_index["from"][episode_index].item())
    end = int(episode_data_index["to"][episode_index].item())
    return start, end


def _get_num_episodes(dataset) -> int:
    episode_data_index = dataset.lerobot_dataset.episode_data_index
    return int(len(episode_data_index["from"]))


def _collect_episode_indices(
    dataset,
    *,
    start_episode_index: int,
    num_episodes: int,
    episode_stride: int,
) -> list[int]:
    total_episodes = _get_num_episodes(dataset)
    if num_episodes <= 0:
        raise ValueError(f"`num_episodes` must be positive, got {num_episodes}")
    if episode_stride <= 0:
        raise ValueError(f"`episode_stride` must be positive, got {episode_stride}")
    if start_episode_index < 0 or start_episode_index >= total_episodes:
        raise IndexError(
            f"`episode_index` out of range: got {start_episode_index}, "
            f"valid range [0, {total_episodes - 1}]"
        )

    episode_indices = list(range(start_episode_index, total_episodes, episode_stride))
    return episode_indices[: min(num_episodes, len(episode_indices))]


def _evaluate_episode_results(
    *,
    model,
    dataset,
    episode_index: int,
    num_inference_steps: int,
    infer_seed: int | None,
    tiled: bool,
    max_frames: int | None,
    episode_mode: str,
    chunk_stride: int,
) -> dict[str, Any]:
    episode_start, episode_end = _get_episode_bounds(dataset, episode_index)
    logger.info(
        "Episode %d bounds on split `%s`: [%d, %d)",
        episode_index,
        getattr(dataset, "is_training_set", "unknown"),
        episode_start,
        episode_end,
    )

    sample_results: list[dict[str, Any]] = []
    first_step_pred_rows: list[np.ndarray] = []
    first_step_gt_rows: list[np.ndarray] = []
    first_step_indices: list[int] = []

    episode_total_frames = episode_end - episode_start
    if max_frames is None:
        effective_frames = episode_total_frames
    else:
        effective_frames = min(max_frames, episode_total_frames)

    if episode_mode == "chunk_only":
        sample_indices = range(episode_start, episode_start + effective_frames, chunk_stride)
        progress_total = len(range(0, effective_frames, chunk_stride))
    elif episode_mode == "full":
        sample_indices = range(episode_start, episode_start + effective_frames)
        progress_total = effective_frames
    else:
        raise ValueError(f"Unsupported episode_mode `{episode_mode}`.")

    for sample_index in tqdm(
        sample_indices,
        desc=f"episode-{episode_index:04d}-{episode_mode}",
        total=progress_total,
        dynamic_ncols=True,
    ):
        result = _evaluate_dataset_sample(
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            num_inference_steps=num_inference_steps,
            infer_seed=infer_seed,
            compute_training_loss=False,
            tiled=tiled,
        )
        if not bool(result["valid_mask"][0]):
            break
        sample_results.append(result)
        if episode_mode == "full":
            first_step_pred_rows.append(result["pred_action"][0].copy())
            first_step_gt_rows.append(result["gt_action"][0].copy())
            first_step_indices.append(sample_index)

    if not sample_results:
        raise RuntimeError(
            f"Episode {episode_index} has no valid first-step samples on the requested split."
        )

    first_step_pred = None
    first_step_gt = None
    first_step_indices_arr = np.asarray(first_step_indices, dtype=np.int32)
    if first_step_pred_rows:
        first_step_pred = np.stack(first_step_pred_rows, axis=0).astype(np.float32, copy=False)
        first_step_gt = np.stack(first_step_gt_rows, axis=0).astype(np.float32, copy=False)

    return {
        "episode_index": int(episode_index),
        "episode_start": int(episode_start),
        "episode_end": int(episode_end),
        "timeline_length": int(effective_frames),
        "episode_mode": episode_mode,
        "sample_results": sample_results,
        "first_step_pred": first_step_pred,
        "first_step_gt": first_step_gt,
        "first_step_indices": first_step_indices_arr,
    }


def _build_chunk_timeline(
    *,
    sample_results: list[dict[str, Any]],
    episode_start: int,
    timeline_length: int,
    chunk_stride: int | None,
) -> dict[str, Any]:
    if not sample_results:
        raise ValueError("`sample_results` must not be empty.")

    horizon = int(sample_results[0]["pred_action"].shape[0])
    stride = horizon if chunk_stride is None else int(chunk_stride)
    if stride <= 0:
        raise ValueError(f"`chunk_stride` must be positive, got {stride}")

    action_dim = int(sample_results[0]["pred_action"].shape[1])
    pred_timeline = np.full((timeline_length, action_dim), np.nan, dtype=np.float32)
    gt_timeline = np.full((timeline_length, action_dim), np.nan, dtype=np.float32)
    chunk_start_offsets: list[int] = []
    chunk_start_dataset_indices: list[int] = []

    for result in sample_results:
        offset = int(result["sample_index"]) - int(episode_start)
        if offset < 0 or offset >= timeline_length:
            continue
        valid_mask = result["valid_mask"]
        valid_steps = np.nonzero(valid_mask)[0]
        if valid_steps.size == 0:
            continue
        chunk_start_offsets.append(offset)
        chunk_start_dataset_indices.append(int(result["sample_index"]))
        for step in valid_steps.tolist():
            dst = offset + step
            if dst >= timeline_length:
                break
            pred_timeline[dst] = result["pred_action"][step]
            gt_timeline[dst] = result["gt_action"][step]

    return {
        "pred_timeline": pred_timeline,
        "gt_timeline": gt_timeline,
        "chunk_stride": stride,
        "chunk_horizon": horizon,
        "chunk_start_offsets": np.asarray(chunk_start_offsets, dtype=np.int32),
        "chunk_start_dataset_indices": np.asarray(chunk_start_dataset_indices, dtype=np.int32),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline open-loop evaluation for FlexWAM on Piper datasets.",
    )
    parser.add_argument("--task", required=True, help="Hydra task name under configs/task.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the FlexWAM checkpoint.")
    parser.add_argument(
        "--dataset-stats-path",
        default=None,
        help="Optional path to dataset_stats.json. If omitted, infer from the checkpoint run directory.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="How many frames to render per episode. Use 0 or a negative value for the full episode.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=32,
        help="How often to start a full-chunk prediction when building the chunk timeline. "
        "Defaults to the action horizon, e.g. 32.",
    )
    parser.add_argument(
        "--episode-mode",
        default="full",
        choices=["full", "chunk_only"],
        help="`full`: evaluate every frame and build both first-step and chunk plots. "
        "`chunk_only`: only evaluate one sample every chunk_stride frames.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-training-loss", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--output-dir", required=True, help="Directory to save metrics and plots.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    register_default_resolvers()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = _resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats_path, checkpoint_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_task_cfg(args.task)
    model_dtype = _mixed_precision_to_dtype(args.mixed_precision)
    _ensure_text_embedding_cache(cfg, args.device, model_dtype)
    dataset = _build_dataset(cfg, args.split, dataset_stats_path)
    model = _build_model(cfg, checkpoint_path, args.device, model_dtype)

    logger.info("Dataset split=%s | size=%d", args.split, len(dataset))
    eval_indices = _collect_eval_indices(
        dataset_len=len(dataset),
        num_samples=args.num_samples,
        sample_stride=args.sample_stride,
    )
    logger.info("Evaluating %d samples with stride=%d", len(eval_indices), args.sample_stride)

    results: list[dict[str, Any]] = []
    for sample_index in tqdm(eval_indices, desc="offline-eval", dynamic_ncols=True):
        result = _evaluate_dataset_sample(
            model=model,
            dataset=dataset,
            sample_index=sample_index,
            num_inference_steps=args.num_inference_steps,
            infer_seed=args.seed,
            compute_training_loss=not args.skip_training_loss,
            tiled=args.tiled,
        )
        results.append(result)

    metrics = _compute_aggregate_metrics(results)
    metrics["split"] = args.split
    metrics["task"] = args.task
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["dataset_stats_path"] = str(dataset_stats_path)
    metrics["evaluated_indices"] = eval_indices

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    logger.info("Saved metrics to %s", metrics_path)

    max_frames = None if args.num_frames <= 0 else int(args.num_frames)
    episode_indices = _collect_episode_indices(
        dataset,
        start_episode_index=args.episode_index,
        num_episodes=args.num_episodes,
        episode_stride=args.episode_stride,
    )
    logger.info(
        "Rendering %d episode(s) on split `%s`: %s",
        len(episode_indices),
        args.split,
        episode_indices,
    )

    episode_artifacts: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        episode_eval = _evaluate_episode_results(
            model=model,
            dataset=dataset,
            episode_index=episode_index,
            num_inference_steps=args.num_inference_steps,
            infer_seed=args.seed,
            tiled=args.tiled,
            max_frames=max_frames,
            episode_mode=args.episode_mode,
            chunk_stride=int(args.chunk_stride),
        )

        first_step_png = None
        first_step_npz = None
        if episode_eval["first_step_pred"] is not None and episode_eval["first_step_gt"] is not None:
            first_step_png = output_dir / f"episode_{episode_index:04d}_first_step.png"
            _plot_joint_curves(
                pred=episode_eval["first_step_pred"],
                gt=episode_eval["first_step_gt"],
                output_path=first_step_png,
                title=(
                    f"Episode first-step prediction vs GT | split={args.split} | episode={episode_index} | "
                    f"frames={episode_eval['first_step_pred'].shape[0]}"
                ),
                x_label="frame index",
            )
            first_step_npz = output_dir / f"episode_{episode_index:04d}_first_step.npz"
            _save_npz(
                first_step_npz,
                pred_action=episode_eval["first_step_pred"],
                gt_action=episode_eval["first_step_gt"],
                dataset_indices=episode_eval["first_step_indices"],
            )

        chunk_timeline = _build_chunk_timeline(
            sample_results=episode_eval["sample_results"],
            episode_start=episode_eval["episode_start"],
            timeline_length=episode_eval["timeline_length"],
            chunk_stride=args.chunk_stride,
        )
        chunk_png = output_dir / f"episode_{episode_index:04d}_chunk_timeline.png"
        _plot_joint_curves(
            pred=chunk_timeline["pred_timeline"],
            gt=chunk_timeline["gt_timeline"],
            output_path=chunk_png,
            title=(
                f"Episode chunk prediction vs GT | split={args.split} | episode={episode_index} | "
                f"frames={chunk_timeline['pred_timeline'].shape[0]} | stride={chunk_timeline['chunk_stride']} | "
                f"horizon={chunk_timeline['chunk_horizon']}"
            ),
            x_label="frame index",
        )
        chunk_npz = output_dir / f"episode_{episode_index:04d}_chunk_timeline.npz"
        _save_npz(
            chunk_npz,
            pred_action=chunk_timeline["pred_timeline"],
            gt_action=chunk_timeline["gt_timeline"],
            chunk_start_offsets=chunk_timeline["chunk_start_offsets"],
            chunk_start_dataset_indices=chunk_timeline["chunk_start_dataset_indices"],
            chunk_stride=np.asarray(chunk_timeline["chunk_stride"], dtype=np.int32),
            chunk_horizon=np.asarray(chunk_timeline["chunk_horizon"], dtype=np.int32),
            first_step_dataset_indices=episode_eval["first_step_indices"],
        )

        if first_step_png is not None:
            logger.info("Saved first-step plot to %s", first_step_png)
        logger.info("Saved chunk timeline plot to %s", chunk_png)
        episode_artifacts.append(
            {
                "episode_index": int(episode_index),
                "episode_mode": episode_eval["episode_mode"],
                "num_frames": int(episode_eval["timeline_length"]),
                "first_step_plot_path": None if first_step_png is None else str(first_step_png),
                "first_step_npz_path": None if first_step_npz is None else str(first_step_npz),
                "chunk_plot_path": str(chunk_png),
                "chunk_npz_path": str(chunk_npz),
                "dataset_indices_used": episode_eval["first_step_indices"].tolist(),
                "chunk_start_dataset_indices": chunk_timeline["chunk_start_dataset_indices"].tolist(),
            }
        )

    summary = {
        "metrics_path": str(metrics_path),
        "episodes": episode_artifacts,
    }
    summary_path = output_dir / "artifacts.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    logger.info("Saved artifact index to %s", summary_path)


if __name__ == "__main__":
    main()
