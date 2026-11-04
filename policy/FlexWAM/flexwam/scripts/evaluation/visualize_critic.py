#!/usr/bin/env python3
"""Router/critic/GT table video. drop1 uses a single p=1 full-camera blackout."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flexwam.config import resolve_config  # noqa: E402
from deploy.policy import _derive_router_display_order  # noqa: E402
from flexwam.models.architectures.flexwam import FlexWAM  # noqa: E402
from flexwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from flexwam.utils.video_io import save_mp4  # noqa: E402

LABEL_W = 280
CAM_W = 640
ROW_H = 36
CAPTION_H = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render router/critic/GT table videos.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--lingbot-video",
        type=Path,
        default=PROJECT_ROOT / "lingbot-video",
    )
    parser.add_argument(
        "--backbone",
        choices=("lingbot", "wan"),
        default="lingbot",
        help="LingBot uses --lingbot-video; WAN uses --action-dit-pretrained-path.",
    )
    parser.add_argument(
        "--action-dit-pretrained-path",
        type=Path,
        default=None,
        help="Wan ActionDiT init (checkpoint still loads trained weights).",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=None,
        help="Override data.text.cache_dir for dataset text embeddings.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cams", nargs="+", default=None, help="Logit/view order, matching the model.")
    parser.add_argument("--display-order", nargs="+", default=None)
    parser.add_argument(
        "--uniform-probe",
        action="store_true",
        help="Force CF delta-action GT to freeze the camera mixer at 1:1:1.",
    )
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--n-gt", type=int, default=1)
    parser.add_argument("--drop-ratio", type=float, default=1.0)
    parser.add_argument(
        "--drop-min",
        type=float,
        default=None,
        help="If set with --drop-max, sample a new p~U[min,max] for each GT row.",
    )
    parser.add_argument(
        "--drop-max",
        type=float,
        default=None,
        help="If set with --drop-min, sample a new p~U[min,max] for each GT row.",
    )
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-fps", type=int, default=15)
    return parser.parse_args()


def _cams_from_cfg(cfg) -> tuple[str, ...]:
    images = cfg.data.train.shape_meta.images
    return tuple(str(item.key) for item in images)


def _default_display_order(cams: tuple[str, ...]) -> tuple[str, ...]:
    preferred = ("left_wrist", "global", "right_wrist")
    if all(name in cams for name in preferred):
        return preferred
    return _derive_router_display_order(cams)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _zvec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.2f}" for v in values) + "]"


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _sample_drop_ratio(drop_ratio: float, drop_min: float | None, drop_max: float | None) -> float:
    if drop_min is None and drop_max is None:
        return float(drop_ratio)
    lo = float(drop_ratio if drop_min is None else drop_min)
    hi = float(drop_ratio if drop_max is None else drop_max)
    if hi <= lo:
        return lo
    return float(np.random.uniform(lo, hi))


def _video_first_frame_uint8(video: torch.Tensor, view_idx: int) -> np.ndarray:
    frame = video[0, view_idx].detach().float().cpu().numpy()
    frame = np.transpose(np.clip(frame * 0.5 + 0.5, 0.0, 1.0), (1, 2, 0))
    return (frame * 255.0).round().astype(np.uint8)


def _verify_tensor_views_match_files(
    dataset,
    cams: tuple[str, ...],
    dataset_dir: Path,
    episode: int = 0,
) -> None:
    sample = dataset[0]
    video = sample["video"]
    if video.ndim != 5:
        raise RuntimeError(f"Expected dataset video [T,V,C,H,W], got {tuple(video.shape)}")
    if int(video.shape[1]) != len(cams):
        raise RuntimeError(f"Video has {video.shape[1]} views but cams={cams}")
    episode_name = f"episode_{episode:06d}"
    file_frames = {
        name: np.asarray(
            Image.fromarray(
                _read_all_frames(
                    dataset_dir / "videos" / "chunk-000" / f"images.{name}" / f"{episode_name}.mp4"
                )[0],
                mode="RGB",
            ).resize((int(video.shape[-1]), int(video.shape[-2]))),
            dtype=np.uint8,
        )
        for name in cams
    }
    print(f"view check video={tuple(video.shape)} cams={cams}", flush=True)
    for view_idx, camera_name in enumerate(cams):
        tensor_frame = _video_first_frame_uint8(video, view_idx)
        scores = {
            name: float(np.mean((tensor_frame.astype(np.float32) - frame.astype(np.float32)) ** 2))
            for name, frame in file_frames.items()
        }
        best_name = min(scores, key=scores.get)
        score_txt = " ".join(f"{name}={mse:.1f}" for name, mse in scores.items())
        print(f"  view{view_idx} tensor->{camera_name}  mse[{score_txt}] best={best_name}", flush=True)
        if best_name != camera_name:
            raise RuntimeError(
                f"View {view_idx} labeled {camera_name} matches file {best_name} instead: {scores}"
            )


def _read_all_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        reader.close()
    return frames


def _episode_frame_indices(dataset, episode: int, stride: int) -> list[tuple[int, int]]:
    inner = dataset.lerobot_dataset.multi_dataset._datasets[0]
    hf = inner.hf_dataset.with_format("numpy")
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    fr = np.asarray(hf["frame_index"]).reshape(-1)
    n = min(len(dataset), int(ep.shape[0]))
    out = []
    for idx in range(n):
        if int(ep[idx]) != int(episode):
            continue
        frame = int(fr[idx])
        if frame % stride == 0:
            out.append((idx, frame))
    return out


def _head_logits(model: FlexWAM, module, ctx: dict) -> np.ndarray:
    if module is getattr(model, "action_camera_critic", None):
        features = ctx.get("critic_features", ctx.get("router_features"))
    else:
        features = ctx.get("router_features", ctx.get("critic_features"))
    kwargs = {
        "first_frame_features": features,
        "video_meta": ctx["video_meta"],
        "proprio": ctx.get("proprio"),
        "state_keep_mask": ctx.get("state_keep_mask"),
        "return_logits": True,
    }
    if "video_tokens" in ctx:
        kwargs["video_pre"] = {"tokens": ctx["video_tokens"], "meta": ctx["video_meta"]}
    accepted = inspect.signature(model._call_camera_head).parameters
    logits = model._call_camera_head(
        module,
        **{key: value for key, value in kwargs.items() if key in accepted},
    )
    return logits.float()[0].detach().cpu().numpy()


def _critic_cf_enabled(cfg) -> bool:
    critic = cfg.model.get("action_critic", None)
    if critic is None:
        return True
    return bool(critic.get("enabled", True))


def _deploy_infer_router(
    model: FlexWAM,
    batch: dict,
    *,
    action_horizon: int,
    num_inference_steps: int,
) -> np.ndarray:
    """Same router weights as deploy (`infer_action` -> `router_weights`)."""
    video = batch["video"]
    if video.ndim == 6:
        input_image = video[0, 0]
        proprio = batch["proprio"][0, 0]
    elif video.ndim == 5:
        input_image = video[0]
        proprio = batch["proprio"][0]
    else:
        raise ValueError(f"Expected video [B,T,V,C,H,W] or [T,V,C,H,W], got {tuple(video.shape)}")
    context = batch["context"]
    context_mask = batch["context_mask"]
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    model.eval()
    if model.camera_router_enabled:
        model.action_camera_router_train_enabled = True
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=int(action_horizon),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            text_cfg_scale=1.0,
            num_inference_steps=int(num_inference_steps),
        )
    router = torch.as_tensor(pred["router_weights"]).detach().float().cpu().numpy().reshape(-1)
    return router


def _prepare_cf_training(model: FlexWAM, *, uniform_probe: bool | None) -> None:
    model.train()
    model.router_cf_interval = 1
    model.camera_critic_enabled = True
    model.set_train_progress(0, stage=4)
    model.action_camera_critic_train_enabled = True
    if model.camera_router_enabled:
        model.action_camera_router_train_enabled = True
    model.action_camera_critic.eval()
    model.action_camera_router.eval()
    if uniform_probe is not None:
        model.router_cf_uniform_probe = bool(uniform_probe)


def _run_cf_trial(model: FlexWAM, ctx: dict, drop_ratio: float, cams: tuple[str, ...]) -> dict:
    delta = model._pixel_critic_cf_deltas(ctx, drop_ratio)
    z = FlexWAM._zscore_per_sample(delta, model.router_cf_zscore_eps).float()[0].cpu().numpy()
    intervention = str(getattr(model, "router_cf_intervention", "pixel"))
    return {
        "p": float(drop_ratio),
        "intervention": intervention,
        "z": z.tolist(),
        "delta": delta.float()[0].detach().cpu().numpy().tolist(),
        "argmax": cams[int(np.argmax(z))],
    }


def compute_rows(
    model: FlexWAM,
    dataset,
    episode: int,
    *,
    stride: int,
    cams: tuple[str, ...],
    n_gt: int,
    drop_ratio: float,
    drop_min: float | None = None,
    drop_max: float | None = None,
    action_horizon: int,
    num_inference_steps: int,
    run_cf_critic: bool,
    uniform_probe: bool | None = None,
) -> list[dict]:
    pairs = _episode_frame_indices(dataset, episode, stride)
    if not pairs:
        raise RuntimeError(f"No stride-{stride} frames found for episode {episode}.")
    print(f"episode {episode} samples: {len(pairs)}  first={pairs[0]} last={pairs[-1]}", flush=True)

    rows = []
    for sample_idx, frame_index in pairs:
        sample = dataset[sample_idx]
        batch = default_collate([sample])
        router = _deploy_infer_router(
            model,
            batch,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
        )
        if run_cf_critic:
            _prepare_cf_training(model, uniform_probe=uniform_probe)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = model.training_loss(batch)
            del loss
            ctx = model._deferred_pixel_cf_ctx
            if ctx is None:
                raise RuntimeError(f"CF ctx missing at episode {episode} frame {frame_index}")
            with torch.no_grad():
                critic = _head_logits(model, model.action_camera_critic, ctx)
            softmax = torch.softmax(torch.from_numpy(critic), dim=-1).numpy()
            gts = [
                _run_cf_trial(model, ctx, _sample_drop_ratio(drop_ratio, drop_min, drop_max), cams)
                for _ in range(n_gt)
            ]
            model._deferred_pixel_cf_ctx = None
            model.eval()
        else:
            critic = np.zeros(len(cams), dtype=np.float64)
            softmax = np.full(len(cams), 1.0 / len(cams), dtype=np.float64)
            gts = [
                {
                    "p": 1.0,
                    "intervention": "n/a",
                    "z": [0.0] * len(cams),
                    "delta": [0.0] * len(cams),
                    "argmax": cams[0],
                }
            ]
        row = {
            "frame": int(frame_index),
            "dataset_idx": int(sample_idx),
            "critic": critic.tolist(),
            "critic_softmax": softmax.tolist(),
            "router": router.tolist(),
            "argmax": cams[int(np.argmax(critic))],
            "gts": gts,
            "mse_vs_gts": [_mse(critic, np.asarray(gt["z"], dtype=np.float64)) for gt in gts],
            "router_source": "deploy_infer_action",
        }
        rows.append(row)
        gt_bits = "  ".join(
            (
                f"GT{i + 1} router-zero z={_zvec(gt['z'])} {gt['argmax']} mse={mse:.2f}"
                if str(gt.get("intervention", "pixel")) == "router_zero"
                else f"GT{i + 1} p={gt['p']:.2f} z={_zvec(gt['z'])} {gt['argmax']} mse={mse:.2f}"
            )
            for i, (gt, mse) in enumerate(zip(gts, row["mse_vs_gts"]))
        )
        print(
            f"ep{episode} frame {frame_index:4d}  "
            f"critic={_zvec(critic)} sm={_zvec(softmax)}  {gt_bits}",
            flush=True,
        )
        del batch, sample
        torch.cuda.empty_cache()
    return rows


def _cam_value(values, camera_name: str, cams: tuple[str, ...]) -> float:
    return float(values[cams.index(camera_name)])


def _softmax(values: list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.clip(e.sum(), 1e-8, None)


def _draw_cell(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill, *, bold: bool = False) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=(50, 50, 50), width=1)
    font = _font(20 if bold else 18)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, fill=fill, font=font)


def _table_block(last_row: dict, *, cams: tuple[str, ...], display_order: tuple[str, ...]) -> np.ndarray:
    n_metric = 3 + len(last_row["gts"])
    width = LABEL_W + CAM_W * len(display_order)
    height = CAPTION_H + ROW_H * n_metric
    pil = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(pil)
    intervention = str(last_row["gts"][0].get("intervention", "pixel")) if last_row["gts"] else "pixel"
    if intervention == "router_zero":
        caption = (
            "Cell = value / softmax. GT: mixer 1:1:1, then set that column's router to 0. "
            "A cell is z after zeroing that camera."
        )
    elif len(last_row["gts"]) > 1:
        caption = (
            "Cell = value / softmax. Each GT row = one p, then drop all cameras in turn. "
            "A cell is z after dropping that column."
        )
    else:
        caption = "Cell = value / softmax. GT: p=1, drop each camera fully; value is z after dropping that column."
    draw.text((12, 10), caption, fill=(255, 210, 80), font=_font(18))

    def col_x(i: int) -> int:
        return LABEL_W + i * CAM_W

    y = CAPTION_H
    _draw_cell(draw, (0, y, LABEL_W, y + ROW_H), "raw / softmax", (180, 180, 180))
    for i, camera_name in enumerate(display_order):
        _draw_cell(draw, (col_x(i), y, col_x(i) + CAM_W, y + ROW_H), camera_name, (255, 255, 255), bold=True)

    critic_raw = [_cam_value(last_row["critic"], name, cams) for name in display_order]
    critic_sm = [_cam_value(last_row["critic_softmax"], name, cams) for name in display_order]
    metric_rows: list[tuple[str, list[float], list[float] | None]] = [
        ("router", [_cam_value(last_row["router"], name, cams) for name in display_order], None),
        ("critic", critic_raw, critic_sm),
    ]
    for i, gt in enumerate(last_row["gts"]):
        raw = [_cam_value(gt["z"], name, cams) for name in display_order]
        label = f"GT  p={float(gt['p']):.2f}" if len(last_row["gts"]) == 1 else f"GT{i + 1}  p={float(gt['p']):.2f}"
        if str(gt.get("intervention", "pixel")) == "router_zero":
            label = "GT  1:1:0" if len(last_row["gts"]) == 1 else f"GT{i + 1}  1:1:0"
        metric_rows.append((label, raw, _softmax(raw).tolist()))

    for r, (label, raw_values, sm_values) in enumerate(metric_rows):
        y = CAPTION_H + (r + 1) * ROW_H
        _draw_cell(draw, (0, y, LABEL_W, y + ROW_H), label, (255, 220, 80) if label.startswith("GT") else (200, 200, 200))
        win = int(np.argmax(raw_values))
        for i, value in enumerate(raw_values):
            if sm_values is None:
                text = f"{value:.3f}"
            else:
                text = f"{value:+.2f}/{sm_values[i]:.2f}"
            fill = (80, 255, 120) if i == win else (220, 220, 220)
            _draw_cell(draw, (col_x(i), y, col_x(i) + CAM_W, y + ROW_H), text, fill)
    return np.asarray(pil, dtype=np.uint8)


def _image_block(
    videos: dict[str, list[np.ndarray]],
    frame_idx: int,
    *,
    display_order: tuple[str, ...],
) -> np.ndarray:
    label = np.zeros((videos[display_order[0]][frame_idx].shape[0], LABEL_W, 3), dtype=np.uint8)
    pil = Image.fromarray(label, mode="RGB")
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(["dataset", "images"]):
        draw.text((12, 16 + i * 28), line, fill=(180, 180, 180), font=_font(18))
    views = [np.asarray(pil, dtype=np.uint8)]
    for camera_name in display_order:
        image = videos[camera_name][frame_idx]
        if image.shape[1] != CAM_W:
            image = np.asarray(Image.fromarray(image, mode="RGB").resize((CAM_W, image.shape[0])), dtype=np.uint8)
        views.append(image)
    return np.concatenate(views, axis=1)


def render_video(
    rows: list[dict],
    episode: int,
    *,
    dataset_dir: Path,
    cams: tuple[str, ...],
    display_order: tuple[str, ...],
    out_dir: Path,
    output_fps: int,
) -> Path:
    by_frame = {int(row["frame"]): row for row in rows}
    infer_frames = sorted(by_frame)
    episode_name = f"episode_{episode:06d}"
    videos = {
        name: _read_all_frames(dataset_dir / "videos" / "chunk-000" / f"images.{name}" / f"{episode_name}.mp4")
        for name in cams
    }
    n_frames = min(len(frames) for frames in videos.values())
    last_row = by_frame[infer_frames[0]]
    infer_idx = 0
    composed: list[Image.Image] = []
    for frame_idx in range(n_frames):
        if infer_idx + 1 < len(infer_frames) and frame_idx >= infer_frames[infer_idx + 1]:
            infer_idx += 1
            last_row = by_frame[infer_frames[infer_idx]]
        table = _table_block(last_row, cams=cams, display_order=display_order)
        images = _image_block(videos, frame_idx, display_order=display_order)
        if table.shape[1] != images.shape[1]:
            raise RuntimeError(f"Width mismatch: table {table.shape} vs images {images.shape}")
        frame = np.concatenate([table, images], axis=0)
        pad_h = (16 - frame.shape[0] % 16) % 16
        pad_w = (16 - frame.shape[1] % 16) % 16
        if pad_h or pad_w:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        composed.append(Image.fromarray(frame, mode="RGB"))
        if (frame_idx + 1) % 100 == 0:
            print(f"{episode_name} composed {frame_idx + 1}/{n_frames}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode_name}.mp4"
    save_mp4(composed, str(out_path), fps=output_fps)
    print(f"Wrote {out_path}  frames={n_frames}", flush=True)
    return out_path


def compose_cfg(
    task: str,
    dataset_dir: Path,
    lingbot_video: Path,
    *,
    backbone: str = "lingbot",
    action_dit_pretrained_path: Path | None = None,
    text_cache_dir: Path | None = None,
):
    overrides = [
        f"task={task}",
        f"data.roots=[{dataset_dir}]",
        "data.split.val_ratio=0",
    ]
    if text_cache_dir is not None:
        overrides.append(f"data.text.cache_dir={text_cache_dir}")
    if backbone == "lingbot":
        overrides.extend(
            [
                f"model.model_path={lingbot_video}",
                f"model.action_dit_pretrained_path={lingbot_video}/LingBotActionDiT_backbone_2048hdim.pt",
            ]
        )
    elif backbone == "wan":
        if action_dit_pretrained_path is None:
            raise ValueError("--action-dit-pretrained-path is required when --backbone=wan.")
        overrides.append(f"model.action_dit_pretrained_path={action_dit_pretrained_path}")
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return resolve_config(compose(config_name="train", overrides=overrides))


def load_model_and_dataset(
    cfg,
    ckpt: Path,
    stats: Path,
    *,
    uniform_probe: bool | None = None,
    run_cf_critic: bool = True,
) -> tuple[FlexWAM, object]:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.load_text_encoder = False
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(ckpt))
    model.cuda()
    model.eval()
    if model.camera_router_enabled:
        model.action_camera_router_train_enabled = True
    if run_cf_critic:
        if uniform_probe is not None:
            model.router_cf_uniform_probe = bool(uniform_probe)
        print(
            f"router_cf_intervention={getattr(model, 'router_cf_intervention', 'pixel')} "
            f"router_cf_uniform_probe={bool(model.router_cf_uniform_probe)} "
            f"pixel_drop=[{model.router_cf_pixel_drop_min},{model.router_cf_pixel_drop_max}]",
            flush=True,
        )
    else:
        print("critic CF disabled for this checkpoint; router uses deploy infer_action only.", flush=True)

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    data_cfg.pretrained_norm_stats = str(stats)
    data_cfg.val_set_proportion = 0.0
    Path("./runs").mkdir(parents=True, exist_ok=True)
    dataset = instantiate(data_cfg)
    return model, dataset


def main() -> None:
    args = parse_args()
    ckpt = args.checkpoint_path.expanduser().resolve()
    stats = (args.dataset_stats_path or (ckpt.parents[2] / "dataset_stats.json")).expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    lingbot_video = args.lingbot_video.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    register_default_resolvers()
    cfg = compose_cfg(
        args.task,
        dataset_dir,
        lingbot_video,
        backbone=str(args.backbone),
        action_dit_pretrained_path=(
            args.action_dit_pretrained_path.expanduser().resolve()
            if args.action_dit_pretrained_path is not None
            else None
        ),
        text_cache_dir=(
            args.text_cache_dir.expanduser().resolve() if args.text_cache_dir is not None else None
        ),
    )
    cams = tuple(args.cams) if args.cams else _cams_from_cfg(cfg)
    display_order = tuple(args.display_order) if args.display_order else _default_display_order(cams)
    episodes = tuple(int(ep) for ep in args.episodes)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cf_critic = _critic_cf_enabled(cfg)
    action_horizon = int(cfg.data.train.num_frames) - 1
    num_inference_steps = int(getattr(cfg, "eval_num_inference_steps", 10))
    print(
        f"cams={cams} display_order={display_order} out_dir={out_dir} "
        f"run_cf_critic={run_cf_critic} action_horizon={action_horizon} "
        f"num_inference_steps={num_inference_steps}",
        flush=True,
    )

    model = None
    dataset = None
    view_checked = False
    for episode in episodes:
        json_path = out_dir / f"episode_{episode:06d}.mp4.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            rows = payload["rows"]
            print(f"Loaded {len(rows)} rows from {json_path}", flush=True)
        else:
            if model is None:
                model, dataset = load_model_and_dataset(
                    cfg,
                    ckpt,
                    stats,
                    uniform_probe=True if args.uniform_probe else None,
                    run_cf_critic=run_cf_critic,
                )
            if not view_checked:
                _verify_tensor_views_match_files(dataset, cams, dataset_dir, episode=int(episodes[0]))
                view_checked = True
            rows = compute_rows(
                model,
                dataset,
                episode,
                stride=int(args.stride),
                cams=cams,
                n_gt=int(args.n_gt),
                drop_ratio=float(args.drop_ratio),
                drop_min=args.drop_min,
                drop_max=args.drop_max,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                run_cf_critic=run_cf_critic,
                uniform_probe=True if args.uniform_probe else None,
            )
            n_gt = int(args.n_gt)
            intervention = str(getattr(model, "router_cf_intervention", "pixel"))
            if intervention == "router_zero":
                layout = "top=router/critic/GT router-zero 1:1:0 table, bottom=dataset images"
            elif n_gt > 1:
                layout = "top=router/critic/GT1-3 table, bottom=dataset images"
            else:
                layout = "top=router/critic/GT p=1 table, bottom=dataset images"
            payload = {
                "checkpoint": str(ckpt),
                "episode": int(episode),
                "stride": int(args.stride),
                "n_gt_per_frame": n_gt,
                "intervention": intervention,
                "drop_ratio": float(args.drop_ratio),
                "drop_min": args.drop_min,
                "drop_max": args.drop_max,
                "uniform_probe": bool(getattr(model, "router_cf_uniform_probe", False)),
                "router_source": "deploy_infer_action",
                "run_cf_critic": bool(run_cf_critic),
                "num_inference_steps": int(num_inference_steps),
                "action_horizon": int(action_horizon),
                "camera_order": list(cams),
                "display_order": list(display_order),
                "layout": layout,
                "output_fps": int(args.output_fps),
                "rows": rows,
            }
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {json_path}", flush=True)
        render_video(
            rows,
            episode,
            dataset_dir=dataset_dir,
            cams=cams,
            display_order=display_order,
            out_dir=out_dir,
            output_fps=int(args.output_fps),
        )


if __name__ == "__main__":
    main()
