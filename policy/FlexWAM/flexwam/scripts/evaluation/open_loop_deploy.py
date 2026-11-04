"""Deploy-path open-loop evaluation: run MultiViewFlexWAMPolicy.infer() against dataset GT."""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from deploy.policy import (
    _compose_task_cfg,
    _resolve_dataset_stats_path,
    _resolve_existing_path,
)
from deploy.multiview_policy import MultiViewFlexWAMPolicy
from deploy.robot import (
    add_robot_type_argument,
    camera_client_key,
    client_field_key,
    get_robot,
    resolve_robot_type,
)
from flexwam.utils.config_resolvers import register_default_resolvers


PIPER_JOINT_NAMES = [
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

FLEXIV_DIM_NAMES = [
    "xyz_x",
    "xyz_y",
    "xyz_z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
]


def _action_dim_labels(robot_type: str, action_dim: int) -> list[str]:
    if robot_type == "piper" and action_dim == len(PIPER_JOINT_NAMES):
        return list(PIPER_JOINT_NAMES)
    if robot_type == "flexiv" and action_dim == len(FLEXIV_DIM_NAMES):
        return list(FLEXIV_DIM_NAMES)
    return [f"dim_{index}" for index in range(action_dim)]


def _tensor_image_to_uint8_hwc(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().to(device="cpu", dtype=torch.float32)
    image = image.permute(1, 2, 0).clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().clamp(0, 255).numpy()
    return image.astype(np.uint8)


def _is_independent_multiview(cfg) -> bool:
    concat_mode = getattr(cfg.data.train, "concat_multi_camera", None)
    return str(concat_mode) == "independent"


def _build_deploy_obs(
    sample: dict,
    raw_state: np.ndarray,
    robot,
    camera_order: tuple[str, ...],
) -> dict[str, Any]:
    obs = {
        client_field_key(robot, "state"): raw_state[0],
        client_field_key(robot, "prompt"): sample["prompt"],
    }
    video = sample["video"]
    if video.ndim != 5:
        raise ValueError(
            "Independent multiview deploy expects video [T,V,C,H,W], "
            f"got {tuple(video.shape)}"
        )
    frame = video[0]
    if int(frame.shape[0]) != len(camera_order):
        raise ValueError(
            f"Sample has {int(frame.shape[0])} views but camera_order has {len(camera_order)}: "
            f"{list(camera_order)}"
        )
    for view_idx, camera_name in enumerate(camera_order):
        obs[camera_client_key(robot, camera_name)] = _tensor_image_to_uint8_hwc(frame[view_idx])
    return obs


def _first_step_state_delta(pred: np.ndarray, state: np.ndarray, *, robot_type: str) -> dict[str, float]:
    metrics = {
        "d0_from_state": float(np.linalg.norm(pred[0] - state)),
        "chunk_span": float(np.linalg.norm(np.ptp(pred, axis=0))),
    }
    if robot_type != "piper":
        return metrics
    metrics.update(
        {
            "left_d0_from_state": float(np.linalg.norm(pred[0, :6] - state[:6])),
            "right_d0_from_state": float(np.linalg.norm(pred[0, 7:13] - state[7:13])),
            "left_chunk_span": float(np.linalg.norm(np.ptp(pred[:, :6], axis=0))),
            "right_chunk_span": float(np.linalg.norm(np.ptp(pred[:, 7:13], axis=0))),
        }
    )
    return metrics


def _denormalize_action_state(processor, action: torch.Tensor, proprio: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    action_btd = action.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
    proprio_btd = proprio.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
    batch = {
        "action": action_btd,
        "state": proprio_btd,
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)

    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    merged = {
        "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
        "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
    }
    merged = processor.action_state_merger.forward(merged)
    return (
        merged["action"].detach().cpu().numpy().astype(np.float32, copy=False),
        merged["state"].detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _plot_actions(
    pred: np.ndarray,
    gt: np.ndarray,
    output_path: Path,
    title: str,
    *,
    dim_names: list[str],
) -> None:
    action_dim = pred.shape[1]
    if gt.shape[1] != action_dim or len(dim_names) != action_dim:
        raise ValueError(
            f"Action plot shape mismatch: pred={pred.shape}, gt={gt.shape}, labels={len(dim_names)}"
        )
    ncols = 2 if action_dim > 1 else 1
    nrows = int(np.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4 * nrows, 6)), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    x = np.arange(pred.shape[0], dtype=np.int32)
    for dim, name in enumerate(dim_names):
        ax = axes[dim]
        ax.plot(x, gt[:, dim], label="GT", linewidth=1.8)
        ax.plot(x, pred[:, dim], label="DeployPolicy pred", linewidth=1.4)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    for ax in axes[action_dim:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_dataset(cfg, split: str, dataset_stats_path: Path, dataset_dir: str | None):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[split], resolve=True))
    dataset_cfg.pretrained_norm_stats = str(dataset_stats_path)
    if dataset_dir is not None:
        dataset_cfg.dataset_dirs = [str(Path(dataset_dir).expanduser().resolve())]
    return instantiate(dataset_cfg)


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
    return int(len(dataset.lerobot_dataset.episode_data_index["from"]))


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


def _evaluate_window(
    *,
    dataset,
    processor,
    policy,
    robot,
    robot_type: str,
    sample_index: int,
) -> dict[str, Any] | None:
    sample = dataset[sample_index]
    gt_action, raw_state_seq = _denormalize_action_state(
        processor,
        action=sample["action"],
        proprio=sample["proprio"],
    )
    if policy.delta_joint_actions:
        gt_action = robot.integrate_delta(raw_state_seq[0], gt_action)
    obs = _build_deploy_obs(sample, raw_state_seq, policy.robot, policy.camera_order)
    pred = policy.infer(obs)[client_field_key(policy.robot, "action")].astype(np.float32, copy=False)
    horizon = min(pred.shape[0], gt_action.shape[0])
    pred = pred[:horizon]
    gt = gt_action[:horizon]
    valid = (
        ~torch.as_tensor(sample.get("action_is_pad", torch.zeros(horizon, dtype=torch.bool)))
        .cpu()
        .numpy()[:horizon]
    )
    if not bool(valid[0]):
        return None
    diff = pred[valid] - gt[valid]
    state_delta = _first_step_state_delta(pred, raw_state_seq[0], robot_type=robot_type)
    return {
        "sample_index": int(sample_index),
        "horizon": int(horizon),
        "valid_steps": int(valid.sum()),
        "mae": float(np.abs(diff).mean()) if diff.size else None,
        "mse": float(np.square(diff).mean()) if diff.size else None,
        "first_step_l2": float(np.linalg.norm(pred[0] - gt[0])),
        "pred": pred,
        "gt": gt,
        **state_delta,
    }


def _aggregate_window_metrics(per_sample: list[dict[str, Any]], robot_type: str) -> dict[str, Any]:
    mae_values = [m["mae"] for m in per_sample if m["mae"] is not None]
    aggregate = {
        "num_samples": len(per_sample),
        "mean_mae": float(np.mean(mae_values)) if mae_values else None,
        "mean_first_step_l2": float(np.mean([m["first_step_l2"] for m in per_sample])),
        "mean_d0_from_state": float(np.mean([m["d0_from_state"] for m in per_sample])),
        "mean_chunk_span": float(np.mean([m["chunk_span"] for m in per_sample])),
    }
    if robot_type == "piper":
        aggregate.update(
            {
                "mean_left_d0_from_state": float(np.mean([m["left_d0_from_state"] for m in per_sample])),
                "mean_right_d0_from_state": float(np.mean([m["right_d0_from_state"] for m in per_sample])),
                "mean_left_chunk_span": float(np.mean([m["left_chunk_span"] for m in per_sample])),
                "mean_right_chunk_span": float(np.mean([m["right_chunk_span"] for m in per_sample])),
            }
        )
    return aggregate


def _serializable_window_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"pred", "gt"}}


def _make_policy(
    *,
    cfg,
    checkpoint_path: Path,
    dataset_stats_path: Path,
    device: str,
    action_horizon: int | None,
    num_inference_steps: int,
    seed: int,
    robot_type: str,
    delta_joint_actions: bool,
    cuda_graph: bool = False,
) -> MultiViewFlexWAMPolicy:
    return MultiViewFlexWAMPolicy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=device,
        action_horizon=action_horizon if action_horizon is not None else int(cfg.data.train.num_frames) - 1,
        num_inference_steps=num_inference_steps,
        sigma_shift=None,
        seed=seed,
        default_prompt=None,
        # Dataset samples already store the fully wrapped DEFAULT_PROMPT string.
        prompt_template=None,
        log_full_io=False,
        action_representation="auto",
        delta_joint_actions=delta_joint_actions,
        robot_type=robot_type,
        cuda_graph=cuda_graph,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployment-path open-loop evaluation.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-stats-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode",
        default="samples",
        choices=["samples", "episode"],
        help="`samples`: stride-sampled windows. `episode`: infer every frame of one or more episodes.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Frames to evaluate per episode in `--mode episode`. Use 0 or negative for the full episode.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=32,
        help="In episode mode, also save a full-chunk plot every this many frames.",
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the action denoise step as a CUDA graph after warmup.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Match deploy.server: denormalized deltas are integrated to absolute targets.",
    )
    add_robot_type_argument(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    register_default_resolvers()
    args = _parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = _resolve_existing_path(args.checkpoint_path, "Checkpoint")
    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats_path, checkpoint_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_task_cfg(args.task)
    if not _is_independent_multiview(cfg):
        raise ValueError(
            "Open-loop deploy eval requires `data.train.concat_multi_camera='independent'`."
        )
    dataset = _build_dataset(cfg, args.split, dataset_stats_path, args.dataset_dir)
    robot_type = resolve_robot_type(args)
    robot = get_robot(robot_type)
    policy = _make_policy(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=args.device,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        robot_type=robot_type,
        delta_joint_actions=bool(args.delta),
        cuda_graph=bool(args.cuda_graph),
    )
    dim_names = _action_dim_labels(robot_type, robot.action_dim)
    processor = dataset.lerobot_dataset.processor
    common = {
        "task": args.task,
        "mode": args.mode,
        "robot_type": robot_type,
        "delta_joint_actions": bool(policy.delta_joint_actions),
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "independent_multiview": True,
        "lingbot_video_path": None if policy.lingbot_video_path is None else str(policy.lingbot_video_path),
    }

    if args.mode == "episode":
        max_frames = None if args.num_frames <= 0 else int(args.num_frames)
        episode_indices = _collect_episode_indices(
            dataset,
            start_episode_index=args.episode_index,
            num_episodes=args.num_episodes,
            episode_stride=args.episode_stride,
        )
        logging.info(
            "Episode mode on split `%s`: %s (dataset size=%d, episodes=%d)",
            args.split,
            episode_indices,
            len(dataset),
            _get_num_episodes(dataset),
        )
        episode_artifacts = []
        for episode_index in episode_indices:
            episode_start, episode_end = _get_episode_bounds(dataset, episode_index)
            total_frames = episode_end - episode_start
            effective_frames = total_frames if max_frames is None else min(max_frames, total_frames)
            logging.info(
                "Episode %d bounds on split `%s`: [%d, %d) frames=%d eval=%d",
                episode_index,
                args.split,
                episode_start,
                episode_end,
                total_frames,
                effective_frames,
            )
            per_sample = []
            first_pred_rows = []
            first_gt_rows = []
            first_indices = []
            for offset in tqdm(
                range(effective_frames),
                desc=f"episode-{episode_index:04d}",
                dynamic_ncols=True,
            ):
                sample_index = episode_start + offset
                result = _evaluate_window(
                    dataset=dataset,
                    processor=processor,
                    policy=policy,
                    robot=robot,
                    robot_type=robot_type,
                    sample_index=sample_index,
                )
                if result is None:
                    break
                per_sample.append(result)
                first_pred_rows.append(result["pred"][0])
                first_gt_rows.append(result["gt"][0])
                first_indices.append(sample_index)
                if args.chunk_stride > 0 and offset % args.chunk_stride == 0:
                    _plot_actions(
                        result["pred"],
                        result["gt"],
                        output_dir / f"episode_{episode_index:04d}_sample_{sample_index:06d}_chunk.png",
                        title=(
                            f"Deploy policy open-loop chunk | episode={episode_index} | "
                            f"frame={offset} | sample={sample_index}"
                        ),
                        dim_names=dim_names,
                    )
            if not per_sample:
                raise RuntimeError(
                    f"Episode {episode_index} has no valid first-step samples on split `{args.split}`."
                )
            first_pred = np.stack(first_pred_rows, axis=0)
            first_gt = np.stack(first_gt_rows, axis=0)
            first_step_png = output_dir / f"episode_{episode_index:04d}_first_step.png"
            _plot_actions(
                first_pred,
                first_gt,
                first_step_png,
                title=(
                    f"Deploy policy episode first-step | task={args.task} | "
                    f"episode={episode_index} | frames={first_pred.shape[0]}"
                ),
                dim_names=dim_names,
            )
            first_step_npz = output_dir / f"episode_{episode_index:04d}_first_step.npz"
            np.savez_compressed(
                first_step_npz,
                first_pred=first_pred,
                first_gt=first_gt,
                sample_indices=np.asarray(first_indices, dtype=np.int32),
            )
            episode_metrics = {
                "episode_index": int(episode_index),
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
                "requested_frames": int(effective_frames),
                "evaluated_frames": len(per_sample),
                "first_step_plot": str(first_step_png),
                "first_step_npz": str(first_step_npz),
                "sample_metrics": [_serializable_window_metrics(item) for item in per_sample],
                **_aggregate_window_metrics(per_sample, robot_type),
            }
            episode_artifacts.append(episode_metrics)
            logging.info(
                "Episode %d first-step frames=%d mean_l2=%.6f mean_mae=%.6f",
                episode_index,
                episode_metrics["evaluated_frames"],
                episode_metrics["mean_first_step_l2"],
                episode_metrics["mean_mae"],
            )
        aggregate = {
            **common,
            "episode_index": args.episode_index,
            "num_episodes": len(episode_artifacts),
            "num_frames": args.num_frames,
            "chunk_stride": args.chunk_stride,
            "episodes": episode_artifacts,
            "mean_mae": float(np.mean([item["mean_mae"] for item in episode_artifacts])),
            "mean_first_step_l2": float(
                np.mean([item["mean_first_step_l2"] for item in episode_artifacts])
            ),
        }
    else:
        indices = list(range(args.start_index, len(dataset), args.sample_stride))[: args.num_samples]
        if not indices:
            raise ValueError("No sample indices selected.")
        per_sample = []
        first_pred_rows = []
        first_gt_rows = []
        for sample_index in tqdm(indices, desc="deploy-open-loop", dynamic_ncols=True):
            result = _evaluate_window(
                dataset=dataset,
                processor=processor,
                policy=policy,
                robot=robot,
                robot_type=robot_type,
                sample_index=sample_index,
            )
            if result is None:
                continue
            per_sample.append(result)
            first_pred_rows.append(result["pred"][0])
            first_gt_rows.append(result["gt"][0])
            _plot_actions(
                result["pred"],
                result["gt"],
                output_dir / f"sample_{sample_index:06d}_chunk.png",
                title=f"Deploy policy open-loop chunk | sample={sample_index}",
                dim_names=dim_names,
            )
        if not per_sample:
            raise RuntimeError("No valid samples were evaluated.")
        first_pred = np.stack(first_pred_rows, axis=0)
        first_gt = np.stack(first_gt_rows, axis=0)
        _plot_actions(
            first_pred,
            first_gt,
            output_dir / "first_step_over_samples.png",
            title=f"Deploy policy first-step open-loop | task={args.task}",
            dim_names=dim_names,
        )
        np.savez_compressed(
            output_dir / "deploy_open_loop_actions.npz",
            first_pred=first_pred,
            first_gt=first_gt,
            sample_indices=np.asarray([item["sample_index"] for item in per_sample], dtype=np.int32),
        )
        aggregate = {
            **common,
            **_aggregate_window_metrics(per_sample, robot_type),
            "sample_metrics": [_serializable_window_metrics(item) for item in per_sample],
        }

    (output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    logging.info("Saved deploy open-loop diagnostics to %s", output_dir)


if __name__ == "__main__":
    main()
