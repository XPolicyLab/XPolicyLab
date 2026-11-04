#!/usr/bin/env python3
"""Run one open-loop sample with a fixed seed. Works on flexwam (framework) and fastwam (clean)."""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

LOGGER = logging.getLogger("open_loop_one")


def _setup_path(root: Path, lingbot: str | None, model_base: str | None) -> None:
    src = root / "src"
    for path in (root, src):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    if lingbot:
        os.environ["FLEXWAM_LINGBOT_PATH"] = lingbot
        os.environ.setdefault(
            "FLEXWAM_ACTION_DIT_PATH",
            f"{lingbot}/LingBotActionDiT_backbone_2048hdim.pt",
        )
    if model_base:
        os.environ["FASTWAM_MODEL_BASE_PATH"] = model_base
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", model_base)
    elif os.environ.get("FASTWAM_MODEL_BASE_PATH"):
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.environ["FASTWAM_MODEL_BASE_PATH"])
    elif os.environ.get("FLEXWAM_MODEL_BASE_PATH"):
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.environ["FLEXWAM_MODEL_BASE_PATH"])


def _detect_pkg(root: Path) -> str:
    if (root / "src" / "flexwam").is_dir():
        return "flexwam"
    if (root / "src" / "fastwam").is_dir():
        return "fastwam"
    raise RuntimeError(f"Neither flexwam nor fastwam found under {root / 'src'}")


def _lingbot_path() -> str:
    path = os.environ.get("FLEXWAM_LINGBOT_PATH")
    if not path:
        raise RuntimeError("Pass --lingbot-path or set FLEXWAM_LINGBOT_PATH.")
    return path


def _compose(pkg: str, root: Path, dataset_dir: str, text_cache: str, task: str):
    lingbot = _lingbot_path()
    if pkg == "flexwam":
        from flexwam.config import compose_task
        from flexwam.utils.config_resolvers import register_default_resolvers
        from flexwam.utils.tmpdir import configure_tmpdir

        configure_tmpdir()
        register_default_resolvers()
        return compose_task(
            task,
            config_dir=root / "configs",
            overrides=(
                f"data.roots=[{dataset_dir}]",
                f"data.val_roots=[{dataset_dir}]",
                f"data.text.cache_dir={text_cache}",
                f"model.model_path={lingbot}",
                f"model.action_dit_pretrained_path={lingbot}/LingBotActionDiT_backbone_2048hdim.pt",
            ),
        )

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.utils.tmpdir import configure_tmpdir

    configure_tmpdir()
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str((root / "configs").resolve()), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                f"task={task}",
                f"data.train.dataset_dirs=[{dataset_dir}]",
                f"data.val.dataset_dirs=[{dataset_dir}]",
                f"data.train.text_embedding_cache_dir={text_cache}",
                f"data.val.text_embedding_cache_dir={text_cache}",
                f"model.model_path={lingbot}",
                f"model.action_dit_pretrained_path={lingbot}/LingBotActionDiT_backbone_2048hdim.pt",
            ],
        )


def _build_model(pkg: str, cfg, checkpoint: Path, device: str, dtype: torch.dtype):
    if pkg == "flexwam":
        from flexwam.utils.camera_metrics import configure_camera_metrics
    else:
        from fastwam.utils.camera_metrics import configure_camera_metrics

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.load_text_encoder = False
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    configure_camera_metrics(model, cfg.data.train)
    model.load_checkpoint(str(checkpoint))
    return model.to(device).eval()


def _build_dataset(cfg, stats: Path, split: str, root: Path):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[split], resolve=True))
    dataset_cfg.pretrained_norm_stats = str(stats)
    cache = dataset_cfg.get("text_embedding_cache_dir")
    if cache is not None:
        cache_path = Path(str(cache)).expanduser()
        if not cache_path.is_absolute():
            cache_path = (root / cache_path).resolve()
        dataset_cfg.text_embedding_cache_dir = str(cache_path)
    return instantiate(dataset_cfg)


def _first_frame_and_horizon(video: torch.Tensor, action: torch.Tensor):
    if video.ndim == 5:
        # independent multi-view: [T, V, C, H, W]
        num_video_frames = int(video.shape[0])
        input_image = video[0]
        action_horizon = int(action.shape[0])
        return input_image, num_video_frames, action_horizon
    if video.ndim == 4:
        # concatenated: [C, T, H, W]
        num_video_frames = int(video.shape[1])
        input_image = video[:, 0]
        action_horizon = int(action.shape[0])
        return input_image, num_video_frames, action_horizon
    raise ValueError(f"Unexpected video shape {tuple(video.shape)}")


def _denormalize(processor, proprio: torch.Tensor, pred_action: torch.Tensor, gt_action: torch.Tensor):
    proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    out = {}
    for name, raw in (("pred", pred_action), ("gt", gt_action)):
        action_btd = raw.detach().to(device="cpu", dtype=torch.float32)
        if action_btd.ndim == 2:
            action_btd = action_btd.unsqueeze(0)
        batch = {"action": action_btd, "state": proprio.unsqueeze(0) if proprio.ndim == 2 else proprio}
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        merged = {
            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
        }
        merged = processor.action_state_merger.forward(merged)
        out[name] = merged["action"].numpy().astype(np.float32, copy=False)
    return out["pred"], out["gt"]


FLEXIV_DIM_NAMES = ("x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "gripper")


def _plot_open_loop(pred: np.ndarray, gt: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_dim = int(pred.shape[1])
    names = list(FLEXIV_DIM_NAMES[:n_dim]) if n_dim <= len(FLEXIV_DIM_NAMES) else [f"dim{i}" for i in range(n_dim)]
    n_rows = (n_dim + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    x = np.arange(pred.shape[0], dtype=np.int32)
    for dim, name in enumerate(names):
        ax = axes[dim]
        ax.plot(x, gt[:, dim], label="GT", linewidth=1.8, color="#1f77b4")
        ax.plot(x, pred[:, dim], label="Pred", linewidth=1.4, color="#d62728")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        if dim >= n_dim - 2:
            ax.set_xlabel("action step")
    for ax in axes[n_dim:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--lingbot-path", default=os.environ.get("FLEXWAM_LINGBOT_PATH"))
    parser.add_argument(
        "--model-base-path",
        default=os.environ.get("FASTWAM_MODEL_BASE_PATH") or os.environ.get("FLEXWAM_MODEL_BASE_PATH"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = args.root.expanduser().resolve()
    _setup_path(root, args.lingbot_path, args.model_base_path)
    pkg = _detect_pkg(root)
    LOGGER.info("root=%s pkg=%s", root, pkg)
    if pkg == "flexwam":
        from flexwam.utils.misc import register_work_dir
    else:
        from fastwam.utils.misc import register_work_dir
    register_work_dir(Path(args.output).expanduser().resolve().parent)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    text_cache = str(Path(args.text_cache).expanduser().resolve())
    cfg = _compose(pkg, root, args.dataset_dir, text_cache, args.task)
    dtype = torch.bfloat16
    model = _build_model(pkg, cfg, Path(args.checkpoint), args.device, dtype)
    dataset = _build_dataset(cfg, Path(args.dataset_stats), args.split, root)
    LOGGER.info("dataset size=%d sample=%d", len(dataset), args.sample_index)

    sample = dataset[args.sample_index]
    video = sample["video"]
    action = sample["action"]
    proprio = sample["proprio"]
    input_image, num_video_frames, action_horizon = _first_frame_and_horizon(video, action)
    LOGGER.info(
        "video=%s action=%s proprio=%s input_image=%s horizon=%d video_frames=%d",
        tuple(video.shape),
        tuple(action.shape),
        tuple(proprio.shape),
        tuple(input_image.shape),
        action_horizon,
        num_video_frames,
    )

    infer_kwargs = {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": action_horizon,
        "proprio": proprio[0],
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "tiled": False,
        "context": sample["context"],
        "context_mask": sample["context_mask"],
    }
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = num_video_frames

    autocast = torch.autocast(device_type="cuda", dtype=dtype) if str(args.device).startswith("cuda") else nullcontext()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad(), autocast:
        pred = model.infer_action(**infer_kwargs)

    pred_action = pred["action"]
    pred_denorm, gt_denorm = _denormalize(dataset.lerobot_dataset.processor, proprio, pred_action, action)
    pad = sample.get("action_is_pad")
    if pad is None:
        valid = np.ones(pred_denorm.shape[0], dtype=bool)
    else:
        valid = ~torch.as_tensor(pad, dtype=torch.bool).cpu().numpy()
    diff = pred_denorm[valid] - gt_denorm[valid]
    payload = {
        "label": args.label,
        "pkg": pkg,
        "root": str(root),
        "checkpoint": args.checkpoint,
        "task": args.task,
        "sample_index": args.sample_index,
        "split": args.split,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "prompt": str(sample["prompt"]),
        "mae": float(np.abs(diff).mean()),
        "mse": float(np.square(diff).mean()),
        "first_step_l2": float(np.linalg.norm(pred_denorm[0] - gt_denorm[0])),
        "video_shape": list(video.shape),
        "video_mean": float(video.float().mean()),
        "gt_action_mean": float(gt_denorm.mean()),
        "pred_action": pred_denorm,
        "gt_action": gt_denorm,
        "valid_mask": valid.astype(np.bool_),
        "pred_normed": pred_action.detach().float().cpu().numpy(),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pred_action=payload["pred_action"],
        gt_action=payload["gt_action"],
        valid_mask=payload["valid_mask"],
        pred_normed=payload["pred_normed"],
    )
    summary = {k: v for k, v in payload.items() if k not in {"pred_action", "gt_action", "valid_mask", "pred_normed"}}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    plot_path = out.with_suffix(".png")
    _plot_open_loop(
        pred_denorm,
        gt_denorm,
        plot_path,
        title=(
            f"{args.label} | split={args.split} sample={args.sample_index} | "
            f"mae={payload['mae']:.4f} first_step_l2={payload['first_step_l2']:.4f}"
        ),
    )
    LOGGER.info("saved %s mae=%.6f first_step_l2=%.6f plot=%s", out, payload["mae"], payload["first_step_l2"], plot_path)


if __name__ == "__main__":
    main()
