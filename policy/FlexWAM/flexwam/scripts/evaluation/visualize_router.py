#!/usr/bin/env python3
"""Generate a combined FlexWAM router MP4 from offline camera videos.

python scripts/evaluation/visualize_router.py \
    --task <task> \
    --checkpoint-path /path/to/step.pt \
    --dataset-stats-path /path/to/dataset_stats.json \
    --default-prompt "<task prompt>" \
    --action-horizon 32 \
    --num-inference-steps 10 \
    --delta \
    --global-video /path/to/images.global/episode.mp4 \
    --output-video /path/to/router.mp4 \
    --router-infer-stride 30
"""


from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import imageio
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deploy.policy import (  # noqa: E402
    CAMERA_ORDER,
    _compose_multiview_router_image,
    _resolve_dataset_stats_path,
    _resolve_existing_path,
    _resolve_task_name,
    build_arg_parser,
    register_default_resolvers,
    _compose_task_cfg,
)
from deploy.robot import camera_client_key, client_field_key, get_robot, resolve_robot_type  # noqa: E402
from deploy.multiview_policy import (  # noqa: E402
    MultiViewFlexWAMPolicy,
    _overlay_attention_heatmap,
    _render_attention_value_grid,
    _to_heatmap_array,
)
from flexwam.utils.video_io import save_mp4  # noqa: E402


logger = logging.getLogger(__name__)
PIPER_STATE_DIM = 14


def _to_rgb_uint8(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected image frame with 3 dims, got shape {array.shape}.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}.")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _read_selected_video_frames(
    path: Path,
    *,
    start_frame: int,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[list[np.ndarray], float | None]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps")
    frames: list[np.ndarray] = []
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx < start_frame:
                continue
            if (frame_idx - start_frame) % frame_stride != 0:
                continue
            frames.append(_to_rgb_uint8(frame))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return frames, None if fps is None else float(fps)


def _load_state_array(path: Path | None, *, state_dim: int = PIPER_STATE_DIM) -> np.ndarray | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if "state" in payload:
            array = payload["state"]
        elif "states" in payload:
            array = payload["states"]
        else:
            first_key = sorted(payload.files)[0]
            array = payload[first_key]
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        state_column = None
        for name in ("state", "observation.state"):
            if name in table.column_names:
                state_column = name
                break
        if state_column is None:
            state_columns = [name for name in table.column_names if name.startswith("observation.state.")]
            if state_columns:
                arrays = [np.asarray(table[name].to_pylist(), dtype=np.float32) for name in sorted(state_columns)]
                array = np.stack(arrays, axis=1)
            else:
                raise ValueError(
                    f"Could not find state column in parquet {path}. "
                    f"Available columns: {table.column_names}"
                )
        else:
            array = np.asarray(table[state_column].to_pylist(), dtype=np.float32)
    elif suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            array = np.asarray([row.get("state", row) for row in rows], dtype=np.float32)
        else:
            payload = json.loads(text)
            array = np.asarray(payload.get("state", payload.get("states", payload)), dtype=np.float32)
    else:
        array = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        if array.shape[0] != state_dim:
            raise ValueError(f"Expected state vector length {state_dim}, got {array.shape}.")
        return array
    if array.ndim == 2:
        if array.shape[1] != state_dim:
            raise ValueError(f"Expected state array shape [T,{state_dim}], got {array.shape}.")
        return array
    raise ValueError(f"Expected state shape [{state_dim}] or [T,{state_dim}], got {array.shape}.")


def _infer_episode_parquet_path(global_video: Path) -> Path | None:
    parts = list(global_video.parts)
    try:
        videos_idx = parts.index("videos")
    except ValueError:
        return None
    if videos_idx + 1 >= len(parts):
        return None
    chunk_name = parts[videos_idx + 1]
    root = Path(*parts[:videos_idx])
    return root / "data" / chunk_name / global_video.with_suffix(".parquet").name


def _infer_camera_video_path(global_video: Path, camera_name: str) -> Path:
    target_part = f"images.{camera_name}"
    parts = list(global_video.parts)
    for idx, part in enumerate(parts):
        if part == "images.global":
            parts[idx] = target_part
            return Path(*parts)
    parent_name = global_video.parent.name
    if parent_name == "global":
        return global_video.parent.parent / camera_name / global_video.name
    if parent_name == "images.global":
        return global_video.parent.parent / target_part / global_video.name
    return global_video.parent.parent / target_part / global_video.name


def _state_for_frame(states: np.ndarray | None, frame_idx: int, *, state_dim: int) -> np.ndarray:
    if states is None:
        return np.zeros((state_dim,), dtype=np.float32)
    if states.ndim == 1:
        return states.astype(np.float32, copy=True)
    return states[min(frame_idx, states.shape[0] - 1)].astype(np.float32, copy=True)


def _sync_torch_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable_timing_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_timing_value(item) for key, item in value.items()}
    return float(value)


def _format_timing_message(policy_timing: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in policy_timing.items():
        if isinstance(value, (list, tuple)):
            continue
        if isinstance(value, dict):
            continue
        try:
            parts.append(f"{key}={float(value):.1f}ms")
        except (TypeError, ValueError):
            continue
    return " ".join(parts)


class OfflineRouterVideoPolicy(MultiViewFlexWAMPolicy):
    def __init__(self, *args: Any, save_router_frame_dir: Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.router_records: list[dict[str, Any]] = []
        self.latest_router_weights: dict[str, float] | None = None
        self.latest_heatmaps: dict[str, np.ndarray | None] | None = None
        self.current_source_frame_index: int | None = None
        self.save_router_frame_dir = None if save_router_frame_dir is None else Path(save_router_frame_dir).expanduser().resolve()
        if self.save_router_frame_dir is not None:
            self.save_router_frame_dir.mkdir(parents=True, exist_ok=True)

    def _save_multiview_router_images(
        self,
        *,
        request_index: int,
        received_images: dict[str, np.ndarray],
        router_weights: dict[str, float] | None,
        **kwargs: Any,
    ) -> None:
        if router_weights is None:
            logger.warning("Frame %d produced no router_weights; skipping router frame.", request_index)
            return
        self.latest_router_weights = {name: float(value) for name, value in router_weights.items()}
        self.router_records.append(
            {
                "request_index": int(request_index),
                "source_frame_index": self.current_source_frame_index,
                "router_weights": dict(self.latest_router_weights),
            }
        )


def _primary_heatmaps_from_pred(pred: Any, camera_names: list[str]) -> dict[str, np.ndarray | None] | None:
    if not isinstance(pred, dict):
        return None
    payload = pred.get("action_attention_heatmaps")
    if not isinstance(payload, dict):
        return None
    primary = payload.get("primary")
    if not isinstance(primary, dict):
        return None
    return {name: _to_heatmap_array(primary.get(name)) for name in camera_names}


def _compose_heatmap_and_value_frame(
    received_images: dict[str, np.ndarray],
    router_weights: dict[str, float] | None,
    heatmaps: dict[str, np.ndarray | None] | None,
    *,
    display_order: tuple[str, ...],
) -> np.ndarray | None:
    overlay_row: list[np.ndarray] = []
    value_row: list[np.ndarray] = []
    for camera_name in display_order:
        if camera_name not in received_images:
            continue
        heatmap = None if heatmaps is None else heatmaps.get(camera_name)
        if isinstance(router_weights, dict) and camera_name in router_weights:
            router_bar = f"{camera_name}: {float(router_weights[camera_name]):.4f}"
        else:
            router_bar = f"{camera_name}: n/a"
        overlay = _overlay_attention_heatmap(
            received_images[camera_name],
            heatmap,
            label=f"{router_bar} | L12-18 mean",
            draw_grid=False,
            draw_cell_values=False,
        )
        overlay_row.append(overlay)
        height, width = overlay.shape[:2]
        value_row.append(
            _render_attention_value_grid(
                heatmap,
                camera_name=camera_name,
                size=(int(width), int(height)),
            )
        )
    if not overlay_row:
        return None
    return np.concatenate(
        [np.concatenate(overlay_row, axis=1), np.concatenate(value_row, axis=1)],
        axis=0,
    )


def _iter_observations(
    camera_frames: dict[str, list[np.ndarray] | None],
    states: np.ndarray | None,
    prompt: str,
    *,
    robot,
    start_frame: int,
    frame_stride: int,
    state_dim: int,
) -> Iterator[dict[str, Any]]:
    if not camera_frames:
        raise ValueError("No camera frames were provided.")
    missing = [name for name, frames in camera_frames.items() if frames is None]
    if missing:
        logger.warning("Missing camera videos %s; duplicating global frames for those views.", missing)
    num_frames = min(len(frames) for frames in camera_frames.values() if frames)
    global_frames = camera_frames["global"]

    for frame_idx in range(num_frames):
        source_frame_idx = int(start_frame) + frame_idx * int(frame_stride)
        obs: dict[str, Any] = {
            client_field_key(robot, "prompt"): prompt,
            client_field_key(robot, "state"): _state_for_frame(states, source_frame_idx, state_dim=state_dim),
        }
        for camera_name, frames in camera_frames.items():
            frame = global_frames[frame_idx] if frames is None else frames[frame_idx]
            obs[camera_client_key(robot, camera_name)] = frame
        yield obs


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Generate a combined FlexWAM router MP4 from offline camera video(s)."
    parser.set_defaults(task_config_path=None)
    parser.add_argument("--global-video", required=True, help="Path to images.global episode mp4.")
    parser.add_argument(
        "--left-wrist-video",
        default=None,
        help="Optional path to images.left_wrist episode mp4. Defaults to sibling of --global-video.",
    )
    parser.add_argument(
        "--right-wrist-video",
        default=None,
        help="Optional path to images.right_wrist episode mp4. Defaults to sibling of --global-video.",
    )
    parser.add_argument(
        "--global2-video",
        default=None,
        help="Optional path to images.global2 episode mp4. Used with Flexiv 3-cam. Defaults to sibling of --global-video.",
    )
    parser.add_argument("--output-video", required=True, help="Output combined router mp4 path.")
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional [D] or [T,D] state file: npy/npz/json/jsonl/txt/csv/parquet. Dim matches --robot-type.",
    )
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--frame-stride", default=1, type=int)
    parser.add_argument(
        "--router-infer-stride",
        default=30,
        type=int,
        help="Run the model once every N selected video frames; intermediate frames reuse the latest router weights.",
    )
    parser.add_argument("--max-frames", default=None, type=int)
    parser.add_argument("--output-fps", default=None, type=int, help="Output fps. Defaults to source fps divided by frame stride.")
    parser.add_argument(
        "--save-router-frame-dir",
        default=None,
        help="Optional directory to save router PNG frames.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help=(
            "Write a veg2-style attention heatmap video instead of the camera-router overlay. "
            "Top row is the image overlay; bottom row prints per-cell attention values. "
            "Intermediate frames reuse the latest heatmap."
        ),
    )
    args = parser.parse_args()
    if not args.task:
        parser.error("--task is required.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    register_default_resolvers()

    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0.")
    if args.router_infer_stride <= 0:
        raise ValueError("--router-infer-stride must be > 0.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0 when provided.")

    task_name = _resolve_task_name(args.task, args.task_config_path)
    cfg = _compose_task_cfg(task_name)
    checkpoint_path = _resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats_path, checkpoint_path)
    action_horizon = int(args.action_horizon) if args.action_horizon is not None else int(cfg.data.train.num_frames) - 1
    num_inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(getattr(cfg, "eval_num_inference_steps", 10))
    )
    prompt_template = None if args.prompt_template == "" else str(args.prompt_template)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        args.device = "cpu"

    robot_type = resolve_robot_type(args)
    robot = get_robot(robot_type)
    state_dim = int(robot.state_dim)
    extra_camera_names = tuple(name for name in robot.camera_order if name != "global")
    extra_video_args = {
        "left_wrist": args.left_wrist_video,
        "right_wrist": args.right_wrist_video,
        "global2": args.global2_video,
    }

    global_video = _resolve_existing_path(args.global_video, "Global video")
    camera_videos: dict[str, Path | None] = {"global": global_video}
    for camera_name in extra_camera_names:
        explicit_path = extra_video_args.get(camera_name)
        if explicit_path is None:
            inferred_video = _infer_camera_video_path(global_video, camera_name)
            camera_videos[camera_name] = inferred_video if inferred_video.exists() else None
            if camera_videos[camera_name] is None:
                logger.warning(
                    "Could not find inferred %s video %s; global frames will be duplicated for %s.",
                    camera_name,
                    inferred_video,
                    camera_name,
                )
        else:
            camera_videos[camera_name] = _resolve_existing_path(explicit_path, f"{camera_name} video")

    camera_frames: dict[str, list[np.ndarray] | None] = {}
    source_fps = None
    for camera_name, video_path in camera_videos.items():
        if video_path is None:
            camera_frames[camera_name] = None
            continue
        logger.info("Reading %s video: %s", camera_name, video_path)
        frames, fps = _read_selected_video_frames(
            video_path,
            start_frame=int(args.start_frame),
            frame_stride=int(args.frame_stride),
            max_frames=args.max_frames,
        )
        if camera_name == "global":
            source_fps = fps
            if not frames:
                raise ValueError(f"No frames selected from {video_path}.")
        camera_frames[camera_name] = frames

    state_path = None if args.state_path is None else _resolve_existing_path(args.state_path, "State file")
    if state_path is None:
        inferred_state_path = _infer_episode_parquet_path(global_video)
        if inferred_state_path is not None and inferred_state_path.exists():
            state_path = inferred_state_path
            logger.info("Using inferred LeRobot state parquet: %s", state_path)
        else:
            logger.warning(
                "Could not infer a LeRobot parquet state file from %s; using zero state for all frames.",
                global_video,
            )
    states = _load_state_array(state_path, state_dim=state_dim)
    if states is not None and states.ndim == 2:
        logger.info("Loaded %d state frame(s) from %s.", states.shape[0], state_path)
    output_fps = int(args.output_fps) if args.output_fps is not None else max(1, round((source_fps or 8.0) / int(args.frame_stride)))

    router_frame_dir = args.save_router_frame_dir

    policy = OfflineRouterVideoPolicy(
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
        action_representation=str(args.action_representation),
        delta_joint_actions=bool(args.delta),
        robot_type=robot_type,
        cuda_graph=bool(args.cuda_graph),
        save_router_frame_dir=None if router_frame_dir is None else Path(router_frame_dir),
    )
    if bool(args.heatmap):
        policy.collect_action_attention = True

    total = 0
    router_frames: list[Image.Image] = []
    for frame_idx, obs in enumerate(_iter_observations(
        camera_frames=camera_frames,
        states=states,
        prompt=str(args.default_prompt),
        robot=policy.robot,
        start_frame=int(args.start_frame),
        frame_stride=int(args.frame_stride),
        state_dim=state_dim,
    )):
        if frame_idx % int(args.router_infer_stride) == 0 or policy.latest_router_weights is None:
            source_frame_index = int(args.start_frame) + frame_idx * int(args.frame_stride)
            policy.current_source_frame_index = source_frame_index
            _sync_torch_device(str(args.device))
            infer_t0 = time.perf_counter()
            pred = policy.infer(obs)
            _sync_torch_device(str(args.device))
            infer_wall_sec = time.perf_counter() - infer_t0
            infer_wall_ms = infer_wall_sec * 1000.0
            policy_timing = pred.get("policy_timing", {}) if isinstance(pred, dict) else {}
            if policy.router_records:
                policy.router_records[-1]["inference_wall_sec"] = float(infer_wall_sec)
                policy.router_records[-1]["inference_wall_ms"] = float(infer_wall_ms)
                policy.router_records[-1]["policy_timing_ms"] = {
                    str(key): _jsonable_timing_value(value) for key, value in policy_timing.items()
                }
            message = (
                "[router inference] "
                f"selected_frame={frame_idx} source_frame={source_frame_index} "
                f"wall={infer_wall_sec:.3f}s ({infer_wall_ms:.1f}ms)"
            )
            timing_message = _format_timing_message(policy_timing) if policy_timing else ""
            if timing_message:
                message += " " + timing_message
            print(message, flush=True)
            if policy.latest_router_weights is None:
                raise RuntimeError(
                    "Model inference did not return router weights. "
                    "Check that the checkpoint has action_camera_router enabled."
                )
            if bool(args.heatmap):
                policy.latest_heatmaps = _primary_heatmaps_from_pred(pred, list(camera_frames))
                if policy.latest_heatmaps is None or all(
                    heatmap is None for heatmap in policy.latest_heatmaps.values()
                ):
                    raise RuntimeError(
                        "Model inference did not return action attention heatmaps. "
                        "Check that return_action_attention_stats is enabled."
                    )

        received_images = {
            camera_name: obs[camera_client_key(policy.robot, camera_name)]
            for camera_name in camera_frames
        }
        display_order = tuple(getattr(policy, "router_display_order", CAMERA_ORDER))
        if bool(args.heatmap):
            combined = _compose_heatmap_and_value_frame(
                received_images,
                policy.latest_router_weights,
                policy.latest_heatmaps,
                display_order=display_order,
            )
        else:
            combined = _compose_multiview_router_image(
                received_images,
                policy.latest_router_weights,
                display_order=display_order,
            )
        if combined is None:
            raise RuntimeError(f"Failed to compose visualization for frame {frame_idx}.")
        image = Image.fromarray(combined, mode="RGB")
        router_frames.append(image)
        if policy.save_router_frame_dir is not None:
            image.save(policy.save_router_frame_dir / f"{frame_idx + 1}.png")

        total += 1
        if total % 30 == 0:
            logger.info(
                "Rendered %d frame(s), model inferences=%d.",
                total,
                len(policy.router_records),
            )

    if not router_frames:
        raise RuntimeError("No router frames were generated.")
    output_path = Path(args.output_video).expanduser().resolve()
    save_mp4(router_frames, str(output_path), fps=output_fps)
    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    inference_wall_times = [
        float(record["inference_wall_sec"])
        for record in policy.router_records
        if "inference_wall_sec" in record
    ]
    meta_path.write_text(
        json.dumps(
            {
                "created_at_unix_ns": time.time_ns(),
                "task": task_name,
                "checkpoint_path": str(checkpoint_path),
                "dataset_stats_path": str(dataset_stats_path),
                "global_video": str(global_video),
                "camera_videos": {
                    name: None if path is None else str(path) for name, path in camera_videos.items()
                },
                "state_path": None if state_path is None else str(state_path),
                "duplicated_global_for_missing_views": any(path is None for path in camera_videos.values()),
                "robot_type": robot_type,
                "state_dim": state_dim,
                "start_frame": int(args.start_frame),
                "frame_stride": int(args.frame_stride),
                "router_infer_stride": int(args.router_infer_stride),
                "num_router_frames": len(router_frames),
                "num_model_inferences": len(policy.router_records),
                "inference_timing": {
                    "wall_sec": inference_wall_times,
                    "mean_wall_sec": (
                        None if not inference_wall_times else float(np.mean(inference_wall_times))
                    ),
                    "min_wall_sec": (
                        None if not inference_wall_times else float(np.min(inference_wall_times))
                    ),
                    "max_wall_sec": (
                        None if not inference_wall_times else float(np.max(inference_wall_times))
                    ),
                },
                "output_fps": output_fps,
                "heatmap": bool(args.heatmap),
                "heatmap_row_key": "primary" if bool(args.heatmap) else None,
                "attention_value_row": bool(args.heatmap),
                "camera_order": list(getattr(policy, "camera_order", CAMERA_ORDER)),
                "router_display_order": list(getattr(policy, "router_display_order", CAMERA_ORDER)),
                "records": policy.router_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved combined %s video to %s",
        "heatmap" if bool(args.heatmap) else "router",
        output_path,
    )
    logger.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    main()
