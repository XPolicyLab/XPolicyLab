"""FlexWAM policy and evaluation support; use the scripts/ entrypoints."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = os.environ.get("FLEXWAM_SKIP_DOWNLOAD", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flexwam.utils.tmpdir import configure_tmpdir  # noqa: E402

configure_tmpdir()

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from deploy.policy import (  # noqa: E402
    FlexWAMPolicy,
    _ROUTER_BAR_HEIGHT,
    _resolve_dataset_stats_path,
    _resolve_existing_path,
    _resolve_task_name,
    build_arg_parser,
    logger as base_logger,
    register_default_resolvers,
    _compose_task_cfg,
)
from deploy.robot import require_client_keys, resolve_robot_type


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"flexwam_piper_server_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    base_logger.setLevel(logging.INFO)


def _to_heatmap_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2 or array.size == 0:
        return None
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value > min_value:
        norm = (heatmap - min_value) / (max_value - min_value)
    else:
        norm = np.zeros_like(heatmap, dtype=np.float32)
    red = np.clip(1.5 * norm - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(norm - 0.5) * 3.0, 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * norm, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _format_attention_cell_value(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 100.0:
        return f"{value:.1f}"
    if abs_value >= 1.0:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    if abs_value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    draw.text(
        (
            left + max((right - left - text_width) // 2, 0),
            top + max((bottom - top - text_height) // 2 - text_box[1], 0),
        ),
        text,
        font=font,
        fill=fill,
    )


def _load_heatmap_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _overlay_attention_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray | None,
    *,
    label: str,
    draw_grid: bool = True,
    draw_cell_values: bool = True,
) -> np.ndarray:
    base = Image.fromarray(image, mode="RGB")
    if heatmap is None:
        overlay = np.asarray(base, dtype=np.uint8).copy()
    else:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        heatmap_rgb = Image.fromarray(_heatmap_to_rgb(heatmap), mode="RGB").resize(
            base.size,
            resample=Image.Resampling.BILINEAR,
        )
        overlay = (
            np.asarray(base, dtype=np.float32) * 0.55
            + np.asarray(heatmap_rgb, dtype=np.float32) * 0.45
        ).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, _ROUTER_BAR_HEIGHT], fill=(0, 0, 0))
    draw.text((6, 7), label, fill=(255, 255, 255), font=_load_heatmap_font(16))

    if heatmap is not None and heatmap.ndim == 2 and heatmap.size > 0 and (draw_grid or draw_cell_values):
        grid_h, grid_w = heatmap.shape
        content_top = _ROUTER_BAR_HEIGHT
        content_height = max(pil.height - content_top, 1)
        cell_width = pil.width / float(grid_w)
        cell_height = content_height / float(grid_h)
        cell_font = _load_heatmap_font(max(7, min(12, int(min(cell_width, cell_height) * 0.32))))
        for row in range(grid_h):
            for col in range(grid_w):
                x0 = int(round(col * cell_width))
                x1 = int(round((col + 1) * cell_width))
                y0 = content_top + int(round(row * cell_height))
                y1 = content_top + int(round((row + 1) * cell_height))
                if draw_grid:
                    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 255), width=1)
                if draw_cell_values:
                    _centered_text(
                        draw,
                        (x0, y0, x1, y1),
                        _format_attention_cell_value(float(heatmap[row, col])),
                        font=cell_font,
                        fill=(255, 255, 255),
                    )

    return np.asarray(pil, dtype=np.uint8)


def _render_attention_value_grid(
    heatmap: np.ndarray | None,
    *,
    camera_name: str,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = (int(size[0]), int(size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Attention grid size must be positive, got {size}.")
    pil = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, width, _ROUTER_BAR_HEIGHT], fill=(0, 0, 0))
    title_font = _load_heatmap_font(16)
    if heatmap is None:
        _centered_text(
            draw,
            (0, 0, width, _ROUTER_BAR_HEIGHT),
            f"{camera_name} attn n/a",
            font=title_font,
            fill=(255, 255, 0),
        )
        return np.asarray(pil, dtype=np.uint8)

    heatmap = np.asarray(heatmap, dtype=np.float32)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    _centered_text(
        draw,
        (0, 0, width, _ROUTER_BAR_HEIGHT),
        (
            f"{camera_name} attn "
            f"[{_format_attention_cell_value(min_value)}, {_format_attention_cell_value(max_value)}]"
        ),
        font=title_font,
        fill=(255, 255, 0),
    )
    colors = _heatmap_to_rgb(heatmap)
    grid_h, grid_w = heatmap.shape
    pad = 4
    content_top = _ROUTER_BAR_HEIGHT
    inner_width = max(width - 2 * pad, 1)
    inner_height = max(height - content_top - pad, 1)
    cell_width = inner_width / float(grid_w)
    cell_height = inner_height / float(grid_h)
    cell_font = _load_heatmap_font(max(6, min(14, int(min(cell_width, cell_height) * 0.38))))
    for row in range(grid_h):
        for col in range(grid_w):
            x0 = pad + int(round(col * cell_width))
            x1 = pad + int(round((col + 1) * cell_width))
            y0 = content_top + int(round(row * cell_height))
            y1 = content_top + int(round((row + 1) * cell_height))
            color = tuple(int(channel) for channel in colors[row, col])
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color, outline=(40, 40, 40))
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            _centered_text(
                draw,
                (x0, y0, x1, y1),
                _format_attention_cell_value(float(heatmap[row, col])),
                font=cell_font,
                fill=(20, 20, 20) if luminance > 140 else (255, 255, 255),
            )
    return np.asarray(pil, dtype=np.uint8)


class MultiViewFlexWAMPolicy(FlexWAMPolicy):
    """Piper deployment policy for FlexWAM independent multi-view inputs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.concat_multi_camera != "independent":
            raise ValueError(
                "FlexWAM deployment expects `data.train.concat_multi_camera='independent'`, "
                f"got {self.concat_multi_camera!r}."
            )
        num_views = int(getattr(self.model.video_expert, "num_views", 1))
        if num_views != len(self.camera_order):
            raise ValueError(
                f"FlexWAM checkpoint expects {num_views} views, but deploy uses {len(self.camera_order)} cameras "
                f"for robot_type={self.robot_type}."
            )
        if self.lingbot_text_encoder is None and not bool(getattr(self.model.video_expert, "use_camera_embedding", False)):
            logger.warning("FlexWAM deployment is running without camera embeddings enabled in the video expert.")


def main() -> None:
    _configure_logging()
    parser = build_arg_parser()
    parser.set_defaults(
        task="conveyor",
        task_config_path=None,
        default_prompt="Pick up the blue cubes from the conveyor belt and place them into the blue bowl. Pick up the red cubes and place them into the red bowl.",
    )
    args = parser.parse_args()
    register_default_resolvers()

    task_name = _resolve_task_name(args.task, args.task_config_path)
    cfg = _compose_task_cfg(task_name)
    checkpoint_path = _resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats_path, checkpoint_path)
    action_horizon = int(args.action_horizon) if args.action_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"Invalid action horizon: {action_horizon}")
    num_inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(getattr(cfg, "eval_num_inference_steps", 10))
    )
    if num_inference_steps <= 0:
        raise ValueError(f"Invalid num_inference_steps: {num_inference_steps}")
    prompt_template = None if args.prompt_template == "" else str(args.prompt_template)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        args.device = "cpu"

    action_representation = str(args.action_representation)

    policy = MultiViewFlexWAMPolicy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=str(args.device),
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        default_prompt=args.default_prompt,
        prompt_template=prompt_template,
        log_full_io=bool(args.log_full_io),
        action_representation=action_representation,
        delta_joint_actions=bool(args.delta),
        robot_type=resolve_robot_type(args),
        cuda_graph=bool(args.cuda_graph),
    )

    policy_name = policy.robot.policy_name(backbone=policy.backbone)
    state_layout, output_layout = policy.robot.layouts()

    metadata = {
        "policy_name": policy_name,
        "robot_type": policy.robot_type,
        "backbone": policy.backbone,
        "task": task_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "action_horizon": action_horizon,
        "num_video_frames": int(cfg.data.train.num_frames),
        "state_dim": int(policy.processor.proprio_output_dim),
        "action_dim": int(policy.processor.action_output_dim),
        "action_representation": policy.action_representation,
        "delta_joint_actions": bool(policy.delta_joint_actions),
        "input_image_shape": [len(policy.camera_order), 3, int(policy.video_size_hw[0]), int(policy.video_size_hw[1])],
        "camera_order": list(policy.camera_order),
        "prompt_template": prompt_template,
        "state_layout": state_layout,
        "output_layout": output_layout,
        "cuda_graph": bool(getattr(policy.model, "use_cuda_graph", False)),
        "client_keys": require_client_keys(policy.robot),
    }
    if policy.lingbot_video_path is not None:
        metadata["lingbot_video_path"] = str(policy.lingbot_video_path)

    logger.info(
        "Starting %s websocket server on %s:%d | task=%s | ckpt=%s | model_base=%s",
        policy.backbone,
        args.host,
        args.port,
        task_name,
        checkpoint_path,
        os.environ.get("FLEXWAM_MODEL_BASE_PATH")
        or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        or "auto(/path/to/fastwam, ./checkpoints)",
    )

    from deploy.server import TimedWebsocketPolicyServer
    server = TimedWebsocketPolicyServer(
        policy=policy,
        host=str(args.host),
        port=int(args.port),
        metadata=metadata,
    )
    print(f"Listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
