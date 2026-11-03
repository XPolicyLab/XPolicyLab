"""Render the predicted future streams dumped by FLEXPI_DUMP_STREAMS as pred-vs-actual mp4s.

    bash render_streams.sh <streams_dir> --step 110000 [--out <dir>] [--fps 4]

<streams_dir>/ep<NNN>_env<i>/{call_*.pt, obs_*.npz} comes from a policy server run
with FLEXPI_DUMP_STREAMS set (model.py _StreamTrace). For every inference call the
predicted future -- 9 frames at steps t, t+4, ..., t+32 -- is decoded and laid out
like the training val video (trainer.evaluate): every row is
[prediction | actual future passed through the same VAE / encoder], rows top to
bottom:

    RGB video | DINO PCA | pointmap XYZ | pointmap depth | pointmap-latent PCA | video-latent PCA

"Actual" is the sim's own observation at those steps, built with the deploy's
per-frame builders, so it takes the same resize / compose chain as the model input.
Frame 0 of every prediction is the clean input frame (the model keeps it fixed).

Per episode, under <out>/ep<NNN>_env<i>/:
  call_<KKKK>.mp4   one inference; the header gives step t and the frame offset
  episode.mp4       every call back to back
  call_<KKKK>.npz   pred_rgb / actual_rgb_vae / actual_rgb_raw  uint8 [T, H, W, 3]
                    pred_xyz / actual_xyz_vae  decoded pointmap composite in [0, 1], float16 [3, T, H, W]
                    actual_dino  DINO features of the actual frames, float16
                    steps [T], valid [T] (False = past the episode end, last frame repeated)
The predicted latents themselves stay in the dump's call_*.pt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloop_replay import DEFAULT_DEPLOY, load_model  # noqa: E402

CONCAT_MODE = "tshape_robotwin_384x320_uniform"  # what trainer.evaluate passes the encoders
ROWS = "rows: RGB | DINO PCA | pointmap XYZ | depth | pointmap latent | video latent"
HEADER_H = 40


def load_obs(path: Path):
    d = np.load(path)
    observation = {}
    for key in d.files:
        if "." in key:
            obs_key, field = key.split(".", 1)
            observation.setdefault(obs_key, {})[field] = d[key]
    return {"observation": observation, "joint_action": {"vector": d["state"]}}


def actual_inputs(pol, obs_list):
    """The actual future frames as the tensors the dataset hands the model (see deploy_policy)."""
    from experiments.robotwin.flexpi_policy.deploy_policy import (
        _CAM_NAME_TO_OBS,
        _CAM_ORDER,
        _build_intrinsics_from_obs,
    )
    from flexpi.per_cam_compose import compose_robotwin_from_per_cam

    frames = [pol._build_robotwin_per_cam_tensors(o) for o in obs_list]          # cam -> [1, 3, h, w]
    per_cam = {k: torch.stack([f[k] for f in frames], dim=2) for k in frames[0]}  # cam -> [1, 3, T, h, w]
    if hasattr(pol, "_build_intrinsics"):
        # Same depth grid + K as the model input (obs_train_parity resamples both).
        depths = [pol._build_robotwin_per_cam_depth_tensors(o) for o in obs_list]  # cam -> [1, 1, H, W]
        per_cam_depth = {cam: torch.cat([d[cam] for d in depths], dim=1) for cam in _CAM_ORDER}
        K = pol._build_intrinsics(obs_list[0])
    else:
        per_cam_depth = {                                                         # cam -> [1, T, H, W] uint16 mm
            cam: torch.from_numpy(np.stack([
                np.asarray(o["observation"][_CAM_NAME_TO_OBS[cam]]["depth"], dtype=np.uint16) for o in obs_list
            ]))[None].to(pol.model.device)
            for cam in _CAM_ORDER
        }
        K = _build_intrinsics_from_obs(obs_list[0])
    composite = compose_robotwin_from_per_cam(per_cam)                            # [1, 3, T, 384, 320]
    return per_cam, per_cam_depth, composite, K


def _to_frames(n_target, row):
    """Repeat a row's frames to n_target, the way trainer.evaluate expands short rows."""
    n = row.shape[0]
    if n < n_target:
        row = row[np.clip(np.arange(n_target) * n // n_target, 0, n - 1)]
    return row[:n_target]


def _to_width(row, width):
    if row.shape[2] == width:
        return row
    h = max(int(round(row.shape[1] * width / row.shape[2])), 1)
    return np.stack([np.array(Image.fromarray(f).resize((width, h), Image.BILINEAR)) for f in row])


def _uint8(video):  # [3, T, H, W] in [0, 1] -> [T, H, W, 3]
    return (video.permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def render_call(fp, pred, per_cam, per_cam_depth, composite, K):
    from flexpi.utils.video_metrics import pil_frames_to_video_tensor
    from flexpi.vis import build_dino_row, build_pointmap_depth_row_vae, build_pointmap_row_vae, build_vae_row

    dev, dt = fp.device, fp.torch_dtype
    fp.set_camera_intrinsics(K.to(dev))
    arrays = {"actual_rgb_raw": _uint8(((composite[0].float().cpu() + 1) * 0.5))}

    pred_video = pil_frames_to_video_tensor(fp._decode_latents(pred["video_latents"].to(dev, dt), tiled=False))
    gt_lat = fp._encode_video_latents(composite.to(dev, dt), tiled=False)
    gt_video = pil_frames_to_video_tensor(fp._decode_latents(gt_lat, tiled=False))
    arrays["pred_rgb"], arrays["actual_rgb_vae"] = _uint8(pred_video), _uint8(gt_video)
    rows = [np.concatenate([arrays["pred_rgb"], arrays["actual_rgb_vae"]], axis=2)]
    T, width = rows[0].shape[0], rows[0].shape[2]

    if pred.get("dino_latents") is not None:
        gt_dino = fp.dino_encoder.encode_video(
            video=None, concat_mode=CONCAT_MODE, **fp._dino_encode_kwargs(),
            temporal_stride=fp.dino_temporal_stride,
            per_cam={k: v.to(dev, dt) for k, v in per_cam.items()},
        ).cpu()                                                                   # [1, 768, F, 294, 1]
        pd = pred["dino_latents"][0]                                              # [768, F, 294, 1]
        rows.append(build_dino_row(
            gt_features=gt_dino[0],
            pred_features=pd.squeeze(-1).permute(1, 2, 0).reshape(1, -1, pd.shape[0]),
            cam_patch_sizes=[tuple(p) for p in fp.dino_cam_patches],
            anchor_count=0,
            share_pca_basis=True,
            layout=getattr(fp, "_layout", None),
            pixel_unshuffle=getattr(fp, "_dino_pixel_unshuffle", 0),
        ))
        arrays["actual_dino"] = gt_dino[0].to(torch.float16).numpy()

    if pred.get("pointmap_latents") is not None:
        gt_pt = fp.pointmap_encoder.encode_composite(
            per_cam_depth=per_cam_depth, camera_intrinsics=fp._resolve_camera_intrinsics(),
            concat_mode=CONCAT_MODE, **fp._layout_kwargs(),
        )                                                                         # [1, 3, T, 384, 320] in [-1, 1]
        gt_pt_lat = fp._encode_video_latents(gt_pt, tiled=False)
        recon_pt = pil_frames_to_video_tensor(fp._decode_latents(gt_pt_lat, tiled=False))
        pred_pt = pil_frames_to_video_tensor(fp._decode_latents(pred["pointmap_latents"].to(dev, dt), tiled=False))
        pt_min, pt_max = fp.pointmap_encoder.pt_min, fp.pointmap_encoder.pt_max
        rows.append(build_pointmap_row_vae(pred=pred_pt, vae_recon=recon_pt))
        rows.append(build_pointmap_depth_row_vae(
            pred=pred_pt, vae_recon=recon_pt,
            z_min=float(pt_min[0, 2, 0, 0].item()), z_max=float(pt_max[0, 2, 0, 0].item()),
            vis_mode=getattr(fp, "pointmap_depth_vis_mode", "turbo"),
        ))
        rows.append(build_vae_row(gt_pt_lat[0].cpu(), pred["pointmap_latents"][0], target_hw=fp._layout.composite_hw))
        arrays["pred_xyz"] = pred_pt.to(torch.float16).numpy()
        arrays["actual_xyz_vae"] = recon_pt.to(torch.float16).numpy()

    rows.append(build_vae_row(gt_lat[0].cpu(), pred["video_latents"][0], target_hw=fp._layout.composite_hw))
    grid = np.concatenate([_to_width(_to_frames(T, r), width) for r in rows], axis=1)
    return grid, arrays


def add_header(grid, lines):
    out = []
    # Total height a multiple of 16, or the mp4 writer rescales every frame to fit its macroblocks.
    header_h = HEADER_H + (-(grid.shape[1] + HEADER_H)) % 16
    for f, text in zip(grid, lines):
        canvas = Image.new("RGB", (f.shape[1], f.shape[0] + header_h), (252, 252, 251))
        canvas.paste(Image.fromarray(f), (0, header_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), text, fill=(11, 11, 11))
        draw.text((6, 22), f"left: predicted | right: actual (through VAE/encoder)   {ROWS}", fill=(82, 81, 78))
        out.append(canvas)
    return out


def render_episode(model, ep_dir: Path, out_dir: Path, fps: int, label: str):
    from flexpi.utils.video_io import save_mp4

    calls = sorted(ep_dir.glob("call_*.pt"))
    if not calls:
        print(f"[render] {ep_dir.name}: no calls, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = model.model.model
    episode_frames = []
    for k, path in enumerate(calls):
        pred = torch.load(path, map_location="cpu", weights_only=False)
        if pred.get("video_latents") is None:
            raise SystemExit(f"{path}: no video latents (action-only path ran); nothing to render")
        t, stride = int(pred["step"]), int(pred["stride"])
        n = pred["video_latents"].shape[2]
        n = (n - 1) * 4 + 1  # WAN 2.2 VAE: 4x temporal compression plus the leading frame
        steps = t + stride * np.arange(n)
        valid = np.array([(ep_dir / f"obs_{s:05d}.npz").exists() for s in steps])
        last = max(int(s) for s, v in zip(steps, valid) if v)
        obs_list = [load_obs(ep_dir / f"obs_{(s if v else last):05d}.npz") for s, v in zip(steps, valid)]
        per_cam, per_cam_depth, composite, K = actual_inputs(model.model, obs_list)
        grid, arrays = render_call(fp, pred, per_cam, per_cam_depth, composite, K)
        lines = [
            f"{label}  {ep_dir.name}  call {k}  t={t}  frame {i}/{n - 1}  (+{stride * i} steps)"
            + ("" if valid[i] else "  [episode ended: actual = last frame]")
            for i in range(n)
        ]
        frames = add_header(grid, lines)
        save_mp4(frames, str(out_dir / f"call_{k:04d}.mp4"), fps=fps)
        np.savez_compressed(out_dir / f"call_{k:04d}.npz", steps=steps, valid=valid, **arrays)
        episode_frames += frames
        print(f"[render] {ep_dir.name} call {k} t={t}: {n} frames, {int(valid.sum())} with actual future")
    save_mp4(episode_frames, str(out_dir / "episode.mp4"), fps=fps)
    print(f"[render] wrote {out_dir}/episode.mp4 ({len(calls)} calls)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("streams_dir", type=Path)
    ap.add_argument("--step", type=int, required=True, help="ckpt step the run used (must match)")
    ap.add_argument("--ckpt-dir", required=True, help="run dir: config.yaml, dataset_stats.json, checkpoints/weights/")
    ap.add_argument("--deploy-yml", default=str(DEFAULT_DEPLOY))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, default=None, help="default: <streams_dir>/../render")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    out = args.out or args.streams_dir.parent / "render"
    t0 = time.time()
    model = load_model(args)
    print(f"[render] model loaded in {time.time() - t0:.0f}s")
    for ep_dir in sorted(p for p in args.streams_dir.glob("ep*_env*") if p.is_dir()):
        render_episode(model, ep_dir, out / ep_dir.name, args.fps, f"step {args.step}")


if __name__ == "__main__":
    main()
