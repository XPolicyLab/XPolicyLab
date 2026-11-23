"""Open-loop replay of RoboDojo-34D dataset episodes through the FlexPi eval pipeline.

Run it through ``openloop_replay.sh`` (same PYTHONPATH / env as the policy server).

From one dataset frame this builds the exact observation dict RoboDojo's
``obs_manager.get_obs()`` hands the policy server under ``env_cfg/arx_x5.yml``
(joint_states + world_ee_state + depth + intrinsic_matrix), then drives the
adapter through the same calls the ws server makes in batch eval:

    decode_obs_images(obs)            # server side, before every call
    Model.update_obs_batch([obs])     # _encode_obs: pack 14D -> build_yam32_state
    Model.get_action_batch([0])       # upstream _infer_action_chunk -> yam32_to_rd14

Only the websocket/msgpack hop is skipped, and it carries arrays losslessly.

There is no FK anywhere in this path: RoboDojo reports the EEF as the sim's
link6 ``body_link_pose_w`` (``get_real_endpose``), and the dataset's EEF block is
that same pose. So the replay feeds the dataset EEF back in the sim's own
``[x, y, z, qw, qx, qy, qz]`` format, and ``build_yam32_state`` must rebuild the
dataset's 32D state from it — checked on every frame (``state32 parity``).

GT for the chunk predicted at frame t is ``action34[t : t+H]`` (== state34[t+1 :
t+H+1]): the target Yam32DRelativeAction(anchor="first") encodes against the
anchor state34[t] in training.

YAM real-robot datasets (32D ``[L_pos3 L_rot6d R_pos3 R_rot6d L_grip R_grip
L_joint6 R_joint6]``, same LeRobot v2.1 + depth_ffv1 format) are read too: the
32D rows are widened to the 34D layout with zeros in the pad slots 26/33, so
everything below is unchanged and the model still sees exactly the stored 32D
state. Use ``--set eef_frame=world_link6`` (the default), which passes the stored
EEF pose through untouched.
"""

import argparse
import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

# The adapter's own deploy.yml (this file is policy/FlexPi/tools/openloop_replay.py).
DEFAULT_DEPLOY = Path(__file__).resolve().parents[1] / "deploy.yml"

# dataset camera -> RoboDojo camera name (model.py's _CAM_TO_OBS_KEY keys).
CAMS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
# YAM real-robot datasets name the same three cameras after the YAM rig.
YAM_CAMS = {"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
# 34D -> 32D eef layout (drop pads 26/33), and 34D -> RoboDojo 14D [L_arm6 L_grip R_arm6 R_grip].
KEEP32 = list(range(26)) + list(range(27, 33))
RD14_FROM_34 = [20, 21, 22, 23, 24, 25, 18, 27, 28, 29, 30, 31, 32, 19]
RD14_NAMES = [f"L_j{i}" for i in range(6)] + ["L_grip"] + [f"R_j{i}" for i in range(6)] + ["R_grip"]
JOINT_COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP_COLS = [6, 13]

INK, BLUE, ORANGE, SURFACE, MUTED = "#0b0b0b", "#2a78d6", "#eb6834", "#fcfcfb", "#8a8984"


# --------------------------------------------------------------------- geometry
def _rot6d_rows_to_mat(r6: np.ndarray) -> np.ndarray:
    """rot6d is the first two ROWS of R (robodojo_eef34 / yam_eef convention)."""
    r1 = r6[0:3] / np.linalg.norm(r6[0:3])
    r2 = r6[3:6] - r1 * np.dot(r1, r6[3:6])
    r2 = r2 / np.linalg.norm(r2)
    return np.stack([r1, r2, np.cross(r1, r2)])


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def pose9_to_pose7(p9: np.ndarray) -> np.ndarray:
    """[pos3, rot6d rows] -> the sim's [x, y, z, qw, qx, qy, qz]."""
    p9 = np.asarray(p9, dtype=np.float64)
    return np.concatenate([p9[:3], _mat_to_quat_wxyz(_rot6d_rows_to_mat(p9[3:9]))]).astype(np.float32)


def rot_err_deg(r6_a: np.ndarray, r6_b: np.ndarray) -> float:
    Ra, Rb = _rot6d_rows_to_mat(r6_a), _rot6d_rows_to_mat(r6_b)
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------------- dataset
class Episode:
    def __init__(self, root: Path, ep: int):
        import pandas as pd

        self.root, self.ep = root, ep
        self.info = json.loads((root / "meta/info.json").read_text())
        self.cams = CAMS if "observation.images.cam_high" in self.info["features"] else YAM_CAMS
        self.chunk = ep // int(self.info["chunks_size"])
        df = pd.read_parquet(root / self.info["data_path"].format(episode_chunk=self.chunk, episode_index=ep))
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        names = self.info["features"]["observation.state"]["names"]
        names = list(names[0] if isinstance(names[0], list) else names)
        if state.shape[1] == 32:  # YAM 32D -> 34D with zero pads (see module docstring)
            state, action = (np.insert(a, [26, 32], 0.0, axis=1) for a in (state, action))
            names = names[:26] + ["left_pad_26"] + names[26:] + ["right_pad_33"]
        if state.shape[1] != 34 or action.shape[1] != 34:
            raise ValueError(f"ep {ep}: expected 34D (or 32D YAM) state/action, got {state.shape} / {action.shape}")
        self.state34, self.action34, self.names34 = state, action, names
        self.n = len(df)
        self.task = None
        with open(root / "meta/episodes.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if row["episode_index"] == ep:
                    self.task = row["tasks"][0]
                    break
        intr = json.loads((root / "meta/camera_intrinsics.json").read_text())
        self.K = {
            cam: np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1]], dtype=np.float32)
            for cam, v in intr.items()
        }

    def _video_path(self, key: str) -> Path:
        rel = self.info["video_path"].format(episode_chunk=self.chunk, video_key=key, episode_index=self.ep)
        ext = (self.info["features"][key].get("info") or {}).get("depth_ext")
        path = self.root / rel
        return path.with_suffix(ext) if ext else path

    def frames(self, wanted):
        """{frame_idx: (rgb{cam: HxWx3 uint8}, depth_mm{cam: HxW uint16})} for the wanted frames only.

        A depth video that is not on disk is skipped (RGB-only replays need not
        download it); depth_mm then has no entry for that camera.
        """
        import av

        wanted = sorted(set(wanted))
        rgb = {t: {} for t in wanted}
        depth = {t: {} for t in wanted}
        for cam in self.cams:
            for key, sink, fmt in (
                (f"observation.images.{cam}", rgb, "rgb24"),
                (f"observation.depth_ffv1.{cam}", depth, None),
            ):
                if sink is depth and not self._video_path(key).is_file():
                    continue
                want = set(wanted)
                with av.open(str(self._video_path(key))) as c:
                    for i, fr in enumerate(c.decode(c.streams.video[0])):
                        if i in want:
                            arr = fr.to_ndarray(format=fmt) if fmt else fr.to_ndarray()
                            if fmt is None and arr.dtype != np.uint16:
                                raise TypeError(f"{key}: depth decoded as {arr.dtype}, want uint16 mm")
                            sink[i][cam] = arr
                            want.discard(i)
                        if not want:
                            break
                if want:
                    raise ValueError(f"{key}: frames {sorted(want)[:5]} missing (video shorter than parquet?)")
        return {t: (rgb[t], depth[t]) for t in wanted}


def build_robodojo_obs(s34, rgb, depth_mm, K, instruction, env_idx=0, cams=CAMS):
    """What obs_manager.get_obs() emits for one env under env_cfg arx_x5.yml."""
    vision = {}
    for ds_cam, rd_cam in cams.items():
        vision[rd_cam] = {"color": rgb[ds_cam], "shape": rgb[ds_cam].shape}
        if ds_cam in depth_mm:
            # The sim emits float32 METRES (distance_to_image_plane); the adapter
            # multiplies by 1000 and truncates to uint16. +0.5 mm makes that
            # truncation land back on the dataset's stored millimetres exactly.
            vision[rd_cam]["depth"] = ((depth_mm[ds_cam].astype(np.float64) + 0.5) / 1000.0).astype(np.float32)
            vision[rd_cam]["intrinsic_matrix"] = K[ds_cam].copy()
    state = {
        "left_arm_joint_state": s34[20:26].copy(),
        "right_arm_joint_state": s34[27:33].copy(),
        "left_ee_joint_state": [float(s34[18])],
        "right_ee_joint_state": [float(s34[19])],
        "left_ee_pose": pose9_to_pose7(s34[0:9]),
        "right_ee_pose": pose9_to_pose7(s34[9:18]),
    }
    return {
        "vision": vision,
        "state": state,
        "action": {},
        "instruction": instruction,
        "env_idx": env_idx,
        "data_format_version": "v1.0",
        "additional_info": {"frequency": 25},
    }


def pack14(step: dict) -> np.ndarray:
    return np.concatenate([
        np.asarray(step["left_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(step["left_ee_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(step["right_arm_joint_state"], dtype=np.float32).reshape(-1),
        np.asarray(step["right_ee_joint_state"], dtype=np.float32).reshape(-1),
    ])


# ------------------------------------------------------------------------ model
def load_model(args):
    """Model(cfg) with the same deploy.yml + overrides setup_eval_policy_server.sh passes."""
    from XPolicyLab.policy.FlexPi.model import Model

    cfg = yaml.safe_load(Path(args.deploy_yml).read_text())
    pattern = "step_*.pt" if args.step is None else f"step_{args.step:06d}.pt"
    weights = sorted((Path(args.ckpt_dir) / "checkpoints/weights").glob(pattern))
    if not weights:
        raise FileNotFoundError(f"no {pattern} under {args.ckpt_dir}/checkpoints/weights")
    print(f"[replay] weights {weights[-1]}")
    cfg.update(
        bench_name="RoboDojo",
        task_name="openloop_replay",
        ckpt_name=Path(args.ckpt_dir).name,
        env_cfg_type="arx_x5",
        action_type="joint",
        action_dim=14,
        seed=0,
        checkpoint_path=str(weights[-1]),
        dataset_stats_path=str(Path(args.ckpt_dir) / "dataset_stats.json"),
        allow_dummy_policy=False,
    )
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = ast.literal_eval(v)
        except Exception:
            cfg[k] = v
        print(f"[replay] override {k}={cfg[k]!r}")
    return Model(cfg)


class Spy:
    """Record what the upstream policy actually received and returned, per call."""

    def __init__(self, model):
        self.last = {}
        inner = model.model
        orig = inner._infer_action_chunk

        def wrapped(observation, instruction):
            out = orig(observation, instruction)
            self.last = {
                "state32": np.array(observation["joint_action"]["vector"], dtype=np.float32),
                "chunk32": np.array(out, dtype=np.float32),
                "obs": observation["observation"],
                "instruction": instruction,
            }
            return out

        inner._infer_action_chunk = wrapped


# ---------------------------------------------------------------------- metrics
def summarize(res: dict, H: int) -> dict:
    pred14, gt14, hold14, valid = res["pred14"], res["gt14"], res["hold14"], res["valid"]
    pred32, gt32, s34 = res["pred32"], res["gt32"], res["state34_t"]
    K = len(res["t"])
    m = np.zeros((K, H), dtype=bool)
    for i, v in enumerate(valid):
        m[i, :v] = True

    def per_step(err):  # err [K, H] -> mean over valid chunks per horizon step
        return np.array([err[m[:, k], k].mean() if m[:, k].any() else np.nan for k in range(H)])

    j_model = np.abs(pred14[..., JOINT_COLS] - gt14[..., JOINT_COLS]).mean(-1)
    j_hold = np.abs(hold14[..., JOINT_COLS] - gt14[..., JOINT_COLS]).mean(-1)
    g_model = np.abs(pred14[..., GRIP_COLS] - gt14[..., GRIP_COLS]).mean(-1)
    g_hold = np.abs(hold14[..., GRIP_COLS] - gt14[..., GRIP_COLS]).mean(-1)

    def eef_pos_err(p):  # cm, mean of both arms
        return 100 * 0.5 * (np.linalg.norm(p[..., 0:3] - gt32[..., 0:3], axis=-1)
                            + np.linalg.norm(p[..., 9:12] - gt32[..., 9:12], axis=-1))

    hold32 = np.repeat(s34[:, None, KEEP32], H, axis=1)
    e_model, e_hold = eef_pos_err(pred32), eef_pos_err(hold32)
    rot_model = np.zeros((K, H))
    for i in range(K):
        for k in range(valid[i]):
            rot_model[i, k] = 0.5 * (rot_err_deg(pred32[i, k, 3:9], gt32[i, k, 3:9])
                                     + rot_err_deg(pred32[i, k, 12:18], gt32[i, k, 12:18]))

    # Direction: does the chunk-end joint displacement point where GT goes?
    cos = []
    for i in range(K):
        k = valid[i] - 1
        d_gt = gt14[i, k, JOINT_COLS] - hold14[i, k, JOINT_COLS]
        d_pr = pred14[i, k, JOINT_COLS] - hold14[i, k, JOINT_COLS]
        if np.linalg.norm(d_gt) > 0.05:
            cos.append(float(d_gt @ d_pr / (np.linalg.norm(d_gt) * np.linalg.norm(d_pr) + 1e-9)))

    per_joint = np.array([
        np.abs(pred14[..., c] - gt14[..., c])[m].mean() for c in range(14)
    ])
    return {
        "n_chunks": int(K),
        "joint_mae_rad": {"model": float(j_model[m].mean()), "hold": float(j_hold[m].mean())},
        "joint_mae_rad_last_step": {"model": float(per_step(j_model)[-1]), "hold": float(per_step(j_hold)[-1])},
        "grip_mae": {"model": float(g_model[m].mean()), "hold": float(g_hold[m].mean())},
        "eef_pos_err_cm": {"model": float(e_model[m].mean()), "hold": float(e_hold[m].mean())},
        "eef_rot_err_deg_model": float(rot_model[m].mean()),
        "chunk_end_delta_cosine": {
            "mean": float(np.mean(cos)) if cos else None,
            "n_moving_chunks": len(cos),
        },
        "per_dim_mae": dict(zip(RD14_NAMES, np.round(per_joint, 4).tolist())),
        "curves": {
            "joint_model": per_step(j_model).tolist(), "joint_hold": per_step(j_hold).tolist(),
            "eef_model": per_step(e_model).tolist(), "eef_hold": per_step(e_hold).tolist(),
        },
    }


# ------------------------------------------------------------------------ plots
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=7)


def plot_all(ep: Episode, res: dict, summ: dict, out: Path, H: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(ep.n)
    gt_full14 = ep.state34[:, RD14_FROM_34]

    # 1) every 14D dim the sim executes: GT state trajectory + each predicted chunk.
    fig, axes = plt.subplots(7, 2, figsize=(13, 15), sharex=True, facecolor=SURFACE)
    for c in range(14):
        ax = axes[c % 7, c // 7]
        _style(ax)
        ax.plot(frames, gt_full14[:, c], color=INK, linewidth=1.5, label="GT (dataset)")
        for i, t in enumerate(res["t"]):
            v = res["valid"][i]
            xs = np.arange(t + 1, t + 1 + v)
            ax.plot(xs, res["pred14"][i, :v, c], color=BLUE, linewidth=1.2, alpha=0.85,
                    label="predicted chunk" if i == 0 else None)
            ax.plot([t], [gt_full14[t, c]], "o", color=BLUE, markersize=2.5)
        ax.set_title(RD14_NAMES[c], fontsize=9, color=INK, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("frame (25 Hz)", fontsize=8, color="#52514e")
    fig.suptitle(f"ep {ep.ep}: executed 14D (joints rad, gripper 0-1), open-loop chunks every {res['stride']} frames",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "joints.png", dpi=110)
    plt.close(fig)

    # 2) error vs horizon: model vs "hold current state".
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
    ks = np.arange(1, H + 1)
    for ax, key, ylabel in (
        (axes[0], "joint", "joint MAE (rad)"),
        (axes[1], "eef", "EEF position error (cm)"),
    ):
        _style(ax)
        ax.plot(ks, summ["curves"][f"{key}_model"], color=BLUE, linewidth=2, label="model")
        ax.plot(ks, summ["curves"][f"{key}_hold"], color=ORANGE, linewidth=2, label="hold current state")
        ax.set_xlabel("horizon step k", fontsize=8, color="#52514e")
        ax.set_ylabel(ylabel, fontsize=8, color="#52514e")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"ep {ep.ep}: error vs horizon over {summ['n_chunks']} chunks", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "error_vs_horizon.png", dpi=110)
    plt.close(fig)

    # 3) EEF xyz the model also predicts (world-frame link6; not executed, diagnostic only).
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True, facecolor=SURFACE)
    for arm, off in (("left", 0), ("right", 9)):
        for d, axis_name in enumerate("xyz"):
            ax = axes[d, 0 if arm == "left" else 1]
            _style(ax)
            ax.plot(frames, ep.state34[:, off + d], color=INK, linewidth=1.5, label="GT (dataset)")
            for i, t in enumerate(res["t"]):
                v = res["valid"][i]
                ax.plot(np.arange(t + 1, t + 1 + v), res["pred32"][i, :v, off + d], color=BLUE,
                        linewidth=1.2, alpha=0.85, label="predicted chunk" if i == 0 else None)
            ax.set_title(f"{arm} EEF {axis_name} (m)", fontsize=9, color=INK, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "eef_xyz.png", dpi=110)
    plt.close(fig)

    plot_34d(ep, res["t"], res["pred32"], res["valid"], out / "gt_vs_pred_34d.png")
    plot_34d(ep, res["t"], res["pred32"], res["valid"], out / "gt_vs_pred_34d_every32.png", every=H)


# One row per quantity, left arm | right arm; pads 26/33 sit in the joint_6 rows.
_ROWS34 = {
    "left": [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 20, 21, 22, 23, 24, 25, 26],
    "right": [9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 27, 28, 29, 30, 31, 32, 33],
}
PAD34 = (26, 33)


def _unit34(d: int) -> str:
    if d in (0, 1, 2, 9, 10, 11):
        return "m"
    if d < 18:
        return "rot6d"
    if d < 20:
        return "0-1"
    return "rad"


def plot_34d(ep: "Episode", ts, pred32, valid, out_png: Path, every=None):
    """GT state34 over the episode vs every predicted chunk, all 34 dims.

    The model's action head is 34 wide, but the deploy chain ends in
    AlignTo34DByWidth.backward, which gathers the 32 real slots; the pads 26/33
    are masked in the DiT and never leave it, so they have GT (exact 0) only.
    ``every`` keeps only chunks starting at multiples of it (the eval replan
    cadence, i.e. what an open-loop execution would stitch together).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ep.names34
    pred34 = np.zeros(pred32.shape[:2] + (34,), dtype=np.float32)
    pred34[..., KEEP32] = pred32
    keep = [i for i, t in enumerate(ts) if every is None or t % every == 0]
    mae = np.full(34, np.nan)
    for d in KEEP32:
        errs = [np.abs(pred34[i, :valid[i], d] - ep.action34[ts[i]:ts[i] + valid[i], d]) for i in keep]
        mae[d] = np.concatenate(errs).mean()

    frames = np.arange(ep.n)
    fig, axes = plt.subplots(17, 2, figsize=(14, 34), sharex=True, facecolor=SURFACE)
    for col, arm in enumerate(("left", "right")):
        for row, d in enumerate(_ROWS34[arm]):
            ax = axes[row, col]
            _style(ax)
            ax.plot(frames, ep.state34[:, d], color=INK, linewidth=1.5, label="GT (dataset)")
            if d in PAD34:
                ax.set_title(f"[{d}] {names[d]}  pad: always 0, no prediction leaves the model",
                             fontsize=8, color=MUTED, loc="left")
                continue
            for j, i in enumerate(keep):
                t, v = ts[i], valid[i]
                ax.plot(np.arange(t + 1, t + 1 + v), pred34[i, :v, d], color=BLUE, linewidth=1.1,
                        alpha=0.85 if every else 0.55, label="predicted chunk" if j == 0 else None)
                ax.plot([t], [ep.state34[t, d]], "o", color=BLUE, markersize=2.5)
            ax.set_title(f"[{d}] {names[d]} ({_unit34(d)})   MAE {mae[d]:.4f}", fontsize=8, color=INK, loc="left")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("frame (25 Hz)", fontsize=8, color="#52514e")
    cadence = f"chunks every {every} frames (eval replan cadence)" if every else f"all {len(keep)} chunks"
    fig.suptitle(f"ep {ep.ep} ({ep.task[:60]}...): 34D GT vs predicted, {cadence}", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def plot_executed(ep: "Episode", extra: dict, chunk_ts, out: Path):
    """GT vs the stitched executed trajectory, one panel per dim, in two figures:

    executed_vs_gt_14d.png  the 14 dims sent to the simulator (joints rad, grippers 0-1)
    executed_vs_gt_34d.png  all 34 dims the model predicts (EEF block is not executed;
                            pads 26/33 have GT only)

    x is the action index i: GT is action34[i] (== state34[i+1]), the executed value is
    step i - t of the chunk started at the last replan t -- the training target pairing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = extra["executed_idx"]
    frames = np.arange(ep.n)
    names34 = ep.names34

    def panel(ax, gt_full, ex, title, is_pad=False):
        _style(ax)
        ax.plot(frames, gt_full, color=INK, linewidth=1.5, label="GT (dataset)")
        for t in chunk_ts:
            ax.axvline(t, color=MUTED, linewidth=0.5, alpha=0.4, linestyle=":")
        if is_pad:
            ax.set_title(f"{title}  pad: always 0, not predicted", fontsize=8, color=MUTED, loc="left")
            return
        ax.plot(f, ex, color=BLUE, linewidth=1.3, label="executed (pred)")
        mae = np.abs(ex - gt_full[f]).mean()
        ax.set_title(f"{title}   MAE {mae:.4f}", fontsize=8, color=INK, loc="left")

    # 14D, the executed action: rows = L/R arm joints 0-5 then gripper
    fig, axes = plt.subplots(7, 2, figsize=(14, 15), sharex=True, facecolor=SURFACE)
    for col, cols14 in enumerate(([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13])):
        for row, c in enumerate(cols14):
            d34 = RD14_FROM_34[c]
            unit = "0-1" if c in GRIP_COLS else "rad"
            panel(axes[row, col], ep.action34[:, d34], extra["executed14"][:, c], f"{RD14_NAMES[c]} ({unit})")
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("action index (25 Hz)", fontsize=8, color="#52514e")
    fig.suptitle(f"ep {ep.ep} ({ep.task[:60]}): executed 14D vs GT, replan every "
                 f"{int(chunk_ts[1] - chunk_ts[0]) if len(chunk_ts) > 1 else '-'} frames (dotted)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "executed_vs_gt_14d.png", dpi=100)
    plt.close(fig)

    # 34D, everything the model predicts
    fig, axes = plt.subplots(17, 2, figsize=(14, 34), sharex=True, facecolor=SURFACE)
    for col, arm in enumerate(("left", "right")):
        for row, d in enumerate(_ROWS34[arm]):
            panel(axes[row, col], ep.action34[:, d], extra["executed34"][:, d],
                  f"[{d}] {names34[d]} ({_unit34(d)})", is_pad=d in PAD34)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("frame (25 Hz)", fontsize=8, color="#52514e")
    fig.suptitle(f"ep {ep.ep} ({ep.task[:60]}): executed 34D vs GT", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out / "executed_vs_gt_34d.png", dpi=90)
    plt.close(fig)


# -------------------------------------------------------------------------- run
def replay_episode(model, spy, ep: Episode, args, out: Path, dump: bool = False):
    from XPolicyLab.utils.process_data import decode_obs_images

    H = int(model.replan_steps)
    last = ep.n - 1 if args.max_frames is None else min(ep.n - 1, args.max_frames)
    if args.full_windows:
        # Only windows with a full H-step GT, t0 in range(0, n - H, stride): the
        # window list of the training-code golden harness (golden_run.py).
        last = min(last, ep.n - H)
    ts = list(range(0, last, args.stride))
    instruction = args.instruction or ep.task
    print(f"[replay] ep {ep.ep}: {ep.n} frames, {len(ts)} chunks (stride {args.stride}, H {H})")
    print(f"[replay] instruction: {instruction!r}")

    t0 = time.time()
    frames = ep.frames(ts)
    print(f"[replay] decoded {len(ts)} x 3 cams rgb+depth in {time.time() - t0:.1f}s")

    if dump:
        # model.py's FLEXPI_DUMP_OBS hook, read per call: one npz per inference holding
        # exactly what _infer_action_chunk received (rgb, depth, K, state32, instruction).
        os.environ["FLEXPI_DUMP_OBS"] = str(out / "model_inputs")
    model.reset()
    res = {k: [] for k in ("t", "pred14", "pred32", "gt14", "gt32", "hold14", "valid", "state34_t")}
    parity = {"state32": 0.0, "depth": 0, "rgb": 0, "K": 0.0}
    for t in ts:
        rgb, depth_mm = frames[t]
        s34 = ep.state34[t]
        obs = decode_obs_images(build_robodojo_obs(s34, rgb, depth_mm, ep.K, instruction, cams=ep.cams))
        model.update_obs_batch([obs])
        steps = model.get_action_batch([0])[0]
        pred14 = np.stack([pack14(s) for s in steps])
        seen = spy.last

        # Input parity: did the pipeline rebuild exactly what training saw?
        parity["state32"] = max(parity["state32"], float(np.abs(seen["state32"] - s34[KEEP32]).max()))
        for ds_cam, rd_cam in ep.cams.items():
            key = model_cam_key(rd_cam)
            parity["rgb"] = max(parity["rgb"], int(np.abs(
                seen["obs"][key]["rgb"].astype(np.int16) - rgb[ds_cam].astype(np.int16)).max()))
            if "depth" not in seen["obs"][key]:  # RGB-only run: the adapter passed no depth or K
                continue
            parity["depth"] = max(parity["depth"], int(np.abs(
                seen["obs"][key]["depth"].astype(np.int64) - depth_mm[ds_cam].astype(np.int64)).max()))
            parity["K"] = max(parity["K"], float(np.abs(seen["obs"][key]["intrinsic_cv"] - ep.K[ds_cam]).max()))

        gt34 = ep.action34[t:t + H]
        v = len(gt34)
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], H - len(a), 0)]) if len(a) < H else a
        res["t"].append(t)
        res["valid"].append(v)
        res["pred14"].append(pred14[:H])
        res["pred32"].append(seen["chunk32"][:H])
        res["gt14"].append(pad(gt34[:, RD14_FROM_34]))
        res["gt32"].append(pad(gt34[:, KEEP32]))
        res["hold14"].append(np.repeat(s34[RD14_FROM_34][None], H, 0))
        res["state34_t"].append(s34)
        jm = np.abs(pred14[:v, JOINT_COLS] - gt34[:, RD14_FROM_34][:, JOINT_COLS]).mean()
        jh = np.abs(s34[RD14_FROM_34][JOINT_COLS][None] - gt34[:, RD14_FROM_34][:, JOINT_COLS]).mean()
        print(f"[replay] t={t:4d}  joint MAE model {jm:.4f} | hold {jh:.4f} rad")

    os.environ.pop("FLEXPI_DUMP_OBS", None)
    res = {k: np.asarray(v) for k, v in res.items()}
    res["stride"] = args.stride
    summ = summarize(res, H)
    summ.update(episode=ep.ep, task=ep.task, instruction=instruction, parity=parity)

    extra = {}
    if args.stride == H:
        # Replanning every H steps: the chunks tile the episode, so concatenating
        # them is exactly the action stream eval would execute if the arm followed GT.
        n = [int(v) for v in res["valid"]]
        ex32 = np.concatenate([res["pred32"][i][:n[i]] for i in range(len(n))])
        ex34 = np.zeros((len(ex32), 34), dtype=np.float32)
        ex34[:, KEEP32] = ex32
        extra = {
            "executed14": np.concatenate([res["pred14"][i][:n[i]] for i in range(len(n))]),
            "executed34": ex34,
            # action index: chunk started at t, step k is the target for action34[t + k]
            "executed_idx": np.concatenate([np.arange(t, t + n[i]) for i, t in enumerate(res["t"])]),
            "gt_state34": ep.state34,
            "gt_action34": ep.action34,
        }
        f = extra["executed_idx"]
        summ["executed_mae_per_dim34"] = np.round(np.abs(ex34 - ep.action34[f]).mean(0), 5).tolist()
        summ["executed_mae_joints_rad"] = float(np.abs(extra["executed14"][:, JOINT_COLS]
                                                       - ep.action34[f][:, RD14_FROM_34][:, JOINT_COLS]).mean())

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "replay.npz", **{k: v for k, v in res.items() if k != "stride"}, **extra)
    (out / "metrics.json").write_text(json.dumps(summ, indent=1))
    if not args.no_plots:
        plot_all(ep, res, summ, out, H)
        if extra:
            plot_executed(ep, extra, res["t"], out)

    print(f"[replay] ---- ep {ep.ep} ----")
    print(f"[replay] input parity (max abs): state32 {parity['state32']:.2e}  depth {parity['depth']} mm  "
          f"rgb {parity['rgb']}  K {parity['K']:.1e}")
    for key in ("joint_mae_rad", "joint_mae_rad_last_step", "grip_mae", "eef_pos_err_cm"):
        print(f"[replay] {key:24s} model {summ[key]['model']:.4f}   hold {summ[key]['hold']:.4f}")
    print(f"[replay] eef_rot_err_deg (model)   {summ['eef_rot_err_deg_model']:.2f}")
    print(f"[replay] chunk-end delta cosine    {summ['chunk_end_delta_cosine']}")
    print(f"[replay] per-dim MAE {summ['per_dim_mae']}")
    print(f"[replay] wrote {out}/{{metrics.json,replay.npz,joints.png,error_vs_horizon.png,eef_xyz.png}}")


def model_cam_key(rd_cam: str) -> str:
    from XPolicyLab.policy.FlexPi.model import _CAM_TO_OBS_KEY

    return _CAM_TO_OBS_KEY[rd_cam]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--dataset-root", required=True, help="34D (or 32D YAM) LeRobot v2.1 dataset root")
    ap.add_argument("--ckpt-dir", required=True, help="run dir: config.yaml, dataset_stats.json, checkpoints/weights/")
    ap.add_argument("--step", type=int, default=None, help="weights step to load (default: the latest step_*.pt)")
    ap.add_argument("--no-plots", action="store_true", help="metrics + replay.npz only")
    ap.add_argument("--dump", action="store_true",
                    help="also save every model input (FLEXPI_DUMP_OBS) under <ep>/model_inputs/")
    ap.add_argument("--deploy-yml", default=str(DEFAULT_DEPLOY))
    ap.add_argument("--stride", type=int, default=8, help="frames between inference calls (eval replans every 32)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--full-windows", action="store_true",
                    help="only windows with a full H-step GT: t0 in range(0, n - H, stride)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="diffusion seeds to replay in turn with one model load (same as --set "
                         "diffusion_seed=S per run); outputs go to <out-dir>/seed<S>/, and --dump "
                         "writes the inputs once, under the first seed")
    ap.add_argument("--instruction", default=None, help="default: the episode's dataset task text")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="deploy.yml override, e.g. --set eef_frame=arm_base_grasp (negative control)")
    ap.add_argument("--out-dir", default="openloop_replay")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    t0 = time.time()
    model = load_model(args)
    spy = Spy(model)
    print(f"[replay] model loaded in {time.time() - t0:.0f}s")
    episodes = [Episode(root, e) for e in args.episodes]
    if args.seeds is None:
        for ep in episodes:
            replay_episode(model, spy, ep, args, Path(args.out_dir) / f"ep{ep.ep:06d}", dump=args.dump)
        return
    for i, seed in enumerate(args.seeds):
        # deploy.yml diffusion_seed ends up as the upstream policy's `seed`, passed to
        # every infer_action call (model.py _build_upstream_cfg).
        model.model.seed = seed
        print(f"[replay] ===== diffusion seed {seed} =====")
        for ep in episodes:
            replay_episode(model, spy, ep, args, Path(args.out_dir) / f"seed{seed}" / f"ep{ep.ep:06d}",
                           dump=args.dump and i == 0)


if __name__ == "__main__":
    main()
