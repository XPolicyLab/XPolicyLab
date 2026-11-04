"""Offline pose diagnostics using the saved run config, cached text and deploy integrator.

No robot connection. Uses fixed seeds, excludes padded actions, and saves every
prediction. The accumulated-error plot is a sensitivity calculation with logged
observations, not a closed-loop robot simulation.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluation.run_one_open_loop import _build_dataset, _build_model
from scripts.evaluation.open_loop_deploy import _denormalize_action_state
from deploy.robot.flexiv import integrate_delta_actions
from flexwam.datasets.lerobot.utils.rotation import rotation_6d_to_matrix
from flexwam.utils.misc import register_work_dir


def matrices(poses):
    return rotation_6d_to_matrix(torch.as_tensor(poses[..., 3:9], dtype=torch.float64)).numpy()


def degrees(matrix):
    shape = matrix.shape[:-2]
    return np.rad2deg(Rotation.from_matrix(matrix.reshape(-1, 3, 3)).magnitude()).reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint filename under <run-dir>/checkpoints/weights/.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    register_work_dir(out)
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    stats = args.run_dir / "dataset_stats.json"
    dataset = _build_dataset(cfg, stats, "train", ROOT)
    # Keep the train episode split while disabling train-time image transforms.
    processor = dataset.lerobot_dataset.processor.eval()
    model = _build_model("flexwam", cfg, args.run_dir / "checkpoints/weights" / args.checkpoint,
                         args.device, torch.bfloat16)
    bounds = dataset.lerobot_dataset.episode_data_index
    episode_ids = np.linspace(0, len(bounds["from"]) - 1, args.episodes, dtype=int)
    records = []
    # Action-only inference conditions on the current image, as deployment does.
    for ep in episode_ids:
        start, end = int(bounds["from"][ep]), int(bounds["to"][ep])
        for index in range(start, end - 31, args.stride):
            sample = dataset._get(index)  # Do not silently replace failed samples.
            gt_delta, state = _denormalize_action_state(processor, sample["action"], sample["proprio"])
            with torch.inference_mode():
                pred = model.infer_action(prompt=None, input_image=sample["video"][0],
                    action_horizon=32, proprio=sample["proprio"][0],
                    num_inference_steps=args.steps, seed=args.seed, rand_device="cpu",
                    tiled=False, context=sample["context"], context_mask=sample["context_mask"])
            pred_norm = pred["action"].detach().cpu().float().reshape(-1, 10)
            pred_delta = processor.normalizer.normalizers["action"]["default"].backward(pred_norm).numpy()
            pa = integrate_delta_actions(state[0], pred_delta)
            ga = integrate_delta_actions(state[0], gt_delta)
            valid = ~sample["action_is_pad"].cpu().numpy().astype(bool)
            records.append(dict(episode=int(ep), index=index, pred=pa, gt=ga, state=state[0],
                pred_delta=pred_delta, gt_delta=gt_delta, valid=valid))
            logging.info("episode=%d index=%d rotation_mean_deg=%.4f", ep, index,
                         degrees(matrices(ga).swapaxes(-1,-2) @ matrices(pa))[valid].mean())
    np.savez_compressed(out / "predictions.npz", **{
        k: np.stack([r[k] for r in records]) for k in records[0]})
    pa, ga = np.stack([r["pred"] for r in records]), np.stack([r["gt"] for r in records])
    pr, gr = matrices(pa), matrices(ga)
    rel = gr.swapaxes(-1,-2) @ pr
    angles = degrees(rel)
    bias = np.rad2deg(Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec()).reshape(*angles.shape,3)
    valid = np.stack([r["valid"] for r in records])
    poserr = np.linalg.norm(pa[...,:3]-ga[...,:3], axis=-1)*1000
    pd = np.stack([r["pred_delta"] for r in records])
    row0 = pd[...,3:6]
    row1 = pd[...,6:9]
    unit0 = row0 / np.linalg.norm(row0, axis=-1, keepdims=True)
    orth1 = row1 - (row1*unit0).sum(-1,keepdims=True)*unit0
    summary = dict(checkpoint=args.checkpoint, data_roots=list(cfg.data.train.dataset_dirs),
        seed=args.seed, steps=args.steps, windows=len(records), episodes=episode_ids.tolist(),
        rotation_mean_deg=float(angles[valid].mean()), rotation_p95_deg=float(np.percentile(angles[valid],95)),
        rotation_max_deg=float(angles[valid].max()),
        rotation_first_deg=float(angles[:,0].mean()), rotation_last_deg=float(angles[:,-1].mean()),
        local_rotvec_bias_deg=bias[valid].mean(0).tolist(),
        position_mean_mm=float(poserr[valid].mean()),
        rot6d_row0_norm_min=float(np.linalg.norm(row0,axis=-1).min()),
        rot6d_orthogonal_row1_norm_min=float(np.linalg.norm(orth1,axis=-1).min()),
        mean_angle_per_horizon_deg=angles.mean(0).tolist())
    (out / "metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(1,3,figsize=(16,4))
    axes[0].plot(angles.mean(0),label="mean")
    axes[0].plot(np.percentile(angles,95,axis=0),label="p95")
    axes[0].set(xlabel="Horizon index",ylabel="SO(3) error (degrees)")
    axes[1].plot(bias.mean(0),label=["local x","local y","local z"])
    axes[1].set(xlabel="Horizon index",ylabel="Signed rotation error (degrees)")
    axes[2].plot(poserr.mean(0))
    axes[2].set(xlabel="Horizon index",ylabel="Position error (mm)")
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(); axes[1].legend()
    fig.suptitle(f"{args.checkpoint}: {len(records)} train windows, seed {args.seed}")
    fig.tight_layout(); fig.savefig(out / "horizon_errors.png", dpi=160); plt.close(fig)
    for ep in episode_ids:
        ids = np.array([i for i,r in enumerate(records) if r["episode"] == ep])
        pp, gg = pa[ids].reshape(-1,10), ga[ids].reshape(-1,10)
        pe = np.rad2deg(np.unwrap(Rotation.from_matrix(matrices(pp)).as_euler("xyz"), axis=0))
        ge = np.rad2deg(np.unwrap(Rotation.from_matrix(matrices(gg)).as_euler("xyz"), axis=0))
        fig, axes = plt.subplots(3,2,figsize=(15,9))
        for d in range(3):
            axes[d,0].plot(gg[:,d]*1000,label="GT"); axes[d,0].plot(pp[:,d]*1000,label="prediction",alpha=.8)
            axes[d,0].set_ylabel(f"{'xyz'[d]} (mm)")
            axes[d,1].plot(ge[:,d],label="GT"); axes[d,1].plot(pe[:,d],label="prediction",alpha=.8)
            axes[d,1].set_ylabel(f"Euler {'xyz'[d]} (degrees)")
        for ax in axes.flat: ax.grid(alpha=.2)
        axes[0,0].legend(); axes[0,1].legend()
        fig.suptitle(f"Train split episode {ep}: independent 32-frame chunks; logged state reset each chunk")
        fig.tight_layout(); fig.savefig(out / f"episode_{ep:03d}.png", dpi=150); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
