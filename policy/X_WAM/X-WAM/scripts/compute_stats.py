"""Compute quantile normalization stats for a robot manipulation dataset.

Scans episode JSONs under ``<dataset_root>/data/``, computes per-key q01/q99
quantiles, and writes ``quantile_pretrain.yaml``. If ``metadata.json`` does not
exist it is created automatically by scanning all episode JSONs.

Computed keys:
  - proprio_{arm}_ee_xyz       : empirical q01/q99
  - gripper_pos                : empirical q01/q99 (left+right merged)
  - action_{arm}_ee_xyz        : symmetric [-M, +M]  (delta: a[t+offset] - p[t])
  - action_{arm}_ee_axisangle  : symmetric [-M, +M]  (delta rotation)
  - gripper_action             : symmetric [-M, +M]  (proprio delta, static-filtered)

Usage::

    python compute_stats.py \\
        --dataset-root /path/to/dataset \\
        [--output quantile_pretrain.yaml] \\
        [--action-skip 1] \\
        [--max-episodes N] \\
        [--workers 32] \\
        [--static-threshold 0.01]
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ARMS = ("left", "right")
STATIC_THRESHOLD = 0.01  # normalized gripper space


# ---------------------------------------------------------------------------
# Episode discovery & metadata
# ---------------------------------------------------------------------------

def _extract_num_frames(json_path: Path) -> int:
    """Fast extraction of num_frames from episode JSON (header peek first)."""
    with open(json_path, "r", encoding="utf-8") as f:
        head = f.read(8192)
    match = re.search(r'"num_frames"\s*:\s*(\d+)', head)
    if match is not None:
        return int(match.group(1))
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "num_frames" not in payload:
        raise ValueError(f"Missing num_frames in {json_path}")
    return int(payload["num_frames"])


def discover_episode_jsons(dataset_root: Path) -> list[Path]:
    """Find all episode JSONs under ``<dataset_root>/data/chunk-*/``."""
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        # Try sub-dataset layout: <root>/<sub>/data/chunk-*/...
        paths: list[Path] = []
        for sub in sorted(dataset_root.iterdir()):
            if not sub.is_dir():
                continue
            sub_data = sub / "data"
            if sub_data.is_dir():
                paths.extend(sorted(sub_data.glob("chunk-*/*.json")))
        return sorted(paths)
    return sorted(data_dir.glob("chunk-*/*.json"))


def _relative_key(json_path: Path, dataset_root: Path) -> str:
    """Build metadata key: relative path without .json suffix.

    For ``<root>/data/chunk-0000/episode_0000000.json`` →
    ``chunk-0000/episode_0000000``.

    For ``<root>/<sub>/data/chunk-0000/rosbag2_xxx.json`` →
    ``<sub>/chunk-0000/rosbag2_xxx``.
    """
    rel = json_path.relative_to(dataset_root)
    parts = rel.parts
    # Strip leading "data/" if present at second position
    if "data" in parts:
        idx = parts.index("data")
        parts = parts[:idx] + parts[idx + 1:]
    key = "/".join(parts)
    if key.endswith(".json"):
        key = key[:-5]
    return key


def load_or_create_metadata(dataset_root: Path, json_paths: list[Path], workers: int) -> dict:
    """Load metadata.json if it exists; otherwise create from scanned episodes."""
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Loaded existing metadata.json ({len(meta)} episodes)")
        return meta

    print(f"metadata.json not found, creating from {len(json_paths)} episode JSONs...")

    def _extract(p: Path) -> tuple[str, int]:
        key = _relative_key(p, dataset_root)
        nf = _extract_num_frames(p)
        return key, nf

    meta: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract, p): p for p in json_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building metadata", ncols=90):
            try:
                key, nf = fut.result()
                meta[key] = nf
            except Exception as exc:
                print(f"  Warning: {futures[fut]}: {exc}", file=sys.stderr)

    # Sort for determinism
    meta = dict(sorted(meta.items()))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata.json with {len(meta)} episodes")
    return meta


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _to_rotmat(raw) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return Rotation.from_rotvec(arr).as_matrix()
    if arr.ndim == 2 and arr.shape[1] == 9:
        return arr.reshape(-1, 3, 3)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    return None


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def _empty_buffers() -> dict[str, list[np.ndarray]]:
    keys = [
        "proprio_left_ee_xyz",
        "proprio_right_ee_xyz",
        "gripper_pos",
        "action_left_ee_xyz",
        "action_right_ee_xyz",
        "action_left_ee_axisangle",
        "action_right_ee_axisangle",
        "gripper_action",
    ]
    return {k: [] for k in keys}


def _has_raw_actions(episode: dict) -> bool:
    actions_dict = episode.get("actions")
    return isinstance(actions_dict, dict) and "raw_actions" in actions_dict


_RAW_DIM_PER_ARM = 7


def process_episode(
    json_path: Path,
    action_offset: int,
    action_skip: int,
    gripper_pos_q01: float | None,
    gripper_pos_q99: float | None,
    static_threshold: float,
) -> dict[str, list[np.ndarray]]:
    """Load one episode and collect all buffers."""
    buffers = _empty_buffers()
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            episode = json.load(f)
    except Exception:
        return buffers

    proprios = episode.get("proprios")
    actions = episode.get("actions")

    if not isinstance(proprios, dict):
        return buffers

    # --- Proprio & gripper_pos (absolute) ---
    for arm in ARMS:
        raw = proprios.get(f"{arm}_ee_pos")
        if raw is not None:
            pos = np.asarray(raw, dtype=np.float64)
            if pos.ndim == 1:
                pos = pos.reshape(-1, 3)
            buffers[f"proprio_{arm}_ee_xyz"].append(pos)

        raw_g = proprios.get(f"{arm}_gripper_pos")
        if raw_g is not None:
            g = np.asarray(raw_g, dtype=np.float64).reshape(-1, 1)
            buffers["gripper_pos"].append(g)

    # --- Action deltas ---
    if isinstance(actions, dict):
        if _has_raw_actions(episode):
            raw = np.asarray(actions["raw_actions"], dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] % _RAW_DIM_PER_ARM == 0:
                num_arms = raw.shape[1] // _RAW_DIM_PER_ARM
                for arm_idx, arm in enumerate(ARMS):
                    if arm_idx >= num_arms:
                        break
                    off = arm_idx * _RAW_DIM_PER_ARM
                    block = raw[:, off : off + _RAW_DIM_PER_ARM]
                    buffers[f"action_{arm}_ee_xyz"].append(block[:, :3])
                    buffers[f"action_{arm}_ee_axisangle"].append(block[:, 3:6])
        else:
            for arm in ARMS:
                raw_p_pos = proprios.get(f"{arm}_ee_pos")
                raw_a_pos = actions.get(f"{arm}_ee_pos")
                if raw_p_pos is not None and raw_a_pos is not None:
                    p_pos = np.asarray(raw_p_pos, dtype=np.float64)
                    a_pos = np.asarray(raw_a_pos, dtype=np.float64)
                    if p_pos.ndim == 1:
                        p_pos = p_pos.reshape(-1, 3)
                    if a_pos.ndim == 1:
                        a_pos = a_pos.reshape(-1, 3)
                    n = min(p_pos.shape[0], a_pos.shape[0] - action_offset)
                    if n > 0:
                        delta = a_pos[action_offset : action_offset + n] - p_pos[:n]
                        buffers[f"action_{arm}_ee_xyz"].append(delta)

                raw_p_rot = proprios.get(f"{arm}_ee_rotm")
                raw_a_rot = actions.get(f"{arm}_ee_rotm")
                p_rotm = _to_rotmat(raw_p_rot)
                a_rotm = _to_rotmat(raw_a_rot)
                if p_rotm is not None and a_rotm is not None:
                    m = min(p_rotm.shape[0], a_rotm.shape[0] - action_offset)
                    if m > 0:
                        delta_R = np.einsum(
                            "tij,tkj->tik",
                            a_rotm[action_offset : action_offset + m],
                            p_rotm[:m],
                        )
                        delta_aa = Rotation.from_matrix(delta_R).as_rotvec().astype(np.float64)
                        buffers[f"action_{arm}_ee_axisangle"].append(delta_aa)

    # --- Gripper action (proprio delta with static filtering) ---
    if gripper_pos_q01 is not None and gripper_pos_q99 is not None:
        denom = max(gripper_pos_q99 - gripper_pos_q01, 1e-6)
        for arm in ARMS:
            raw_grip = proprios.get(f"{arm}_gripper_pos")
            if raw_grip is None:
                continue
            grip = np.asarray(raw_grip, dtype=np.float64).reshape(-1, 1)
            n = grip.shape[0] - action_skip
            if n <= 0:
                continue
            raw_delta = grip[action_skip : action_skip + n] - grip[:n]
            norm_delta = 2.0 * raw_delta / denom
            keep_mask = np.abs(norm_delta[:, 0]) >= static_threshold
            if keep_mask.any():
                buffers["gripper_action"].append(raw_delta[keep_mask])

    return buffers


# ---------------------------------------------------------------------------
# Quantile computation
# ---------------------------------------------------------------------------

def _action_key_uses_symmetric_bounds(key: str) -> bool:
    return key.startswith("action_") or key == "gripper_action"


def _symmetric_q01_q99(q01: np.ndarray, q99: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per axis: M = max(|q01|, |q99|), return (-M, +M)."""
    m = np.maximum(np.abs(q01), np.abs(q99))
    return -m, m


def compute_quantiles(
    buffers: dict[str, list[np.ndarray]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Concatenate buffers and compute q01/q99 per key."""
    q01_out: dict[str, list[float]] = {}
    q99_out: dict[str, list[float]] = {}
    for key, parts in buffers.items():
        if len(parts) == 0:
            continue
        stacked = np.concatenate(parts, axis=0)
        q01 = np.quantile(stacked, 0.01, axis=0).astype(np.float64)
        q99 = np.quantile(stacked, 0.99, axis=0).astype(np.float64)
        if _action_key_uses_symmetric_bounds(key):
            q01, q99 = _symmetric_q01_q99(q01, q99)
        q01_out[key] = q01.reshape(-1).tolist()
        q99_out[key] = q99.reshape(-1).tolist()
    return q01_out, q99_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Dataset root containing data/ subfolder with chunk-*/*.json.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output YAML path (default: <dataset-root>/quantile_pretrain.yaml).")
    parser.add_argument("--action-skip", type=int, default=1,
                        help="Action skip (action_offset = action_skip - 1). Default: 1.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on number of episodes for faster debugging.")
    parser.add_argument("--workers", type=int, default=32,
                        help="Thread pool size for parallel episode loading.")
    parser.add_argument("--static-threshold", type=float, default=STATIC_THRESHOLD,
                        help=f"Gripper static-delta threshold in normalized space (default: {STATIC_THRESHOLD}).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    output_path: Path = (args.output or (dataset_root / "quantile_pretrain.yaml")).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    action_skip = int(args.action_skip)
    if action_skip < 1:
        raise ValueError(f"--action-skip must be >= 1, got {action_skip}.")
    action_offset = action_skip - 1

    print(f"{'=' * 60}")
    print(f"Dataset root : {dataset_root}")
    print(f"Output       : {output_path}")
    print(f"action_skip={action_skip}  action_offset={action_offset}")
    print(f"static_threshold={args.static_threshold}")
    print(f"{'=' * 60}")

    # --- Discover episodes ---
    json_paths = discover_episode_jsons(dataset_root)
    if len(json_paths) == 0:
        raise RuntimeError(f"No episode JSONs found under {dataset_root}")
    print(f"Found {len(json_paths)} episode JSONs")

    # --- Metadata ---
    load_or_create_metadata(dataset_root, json_paths, args.workers)

    # --- Subsample ---
    if args.max_episodes is not None and len(json_paths) > args.max_episodes:
        random.seed(args.seed)
        json_paths = random.sample(json_paths, args.max_episodes)
        print(f"Sub-sampled to {len(json_paths)} episodes (seed={args.seed})")

    # === Pass 1: compute proprio/gripper_pos/action quantiles (no gripper_action yet) ===
    print("\n--- Pass 1: proprio, gripper_pos, action deltas ---")
    buffers = _empty_buffers()
    workers = max(1, int(args.workers))

    def _worker_pass1(p: Path) -> dict[str, list[np.ndarray]]:
        return process_episode(p, action_offset, action_skip,
                               gripper_pos_q01=None, gripper_pos_q99=None,
                               static_threshold=args.static_threshold)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker_pass1, p): p for p in json_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pass 1", ncols=90):
            try:
                partial = fut.result()
            except Exception as exc:
                print(f"  Warning: {futures[fut]}: {exc}", file=sys.stderr)
                continue
            for key in buffers:
                buffers[key].extend(partial[key])

    # Compute quantiles for pass 1 keys (excluding gripper_action)
    q01_all, q99_all = compute_quantiles(
        {k: v for k, v in buffers.items() if k != "gripper_action"}
    )

    # --- Report pass 1 ---
    print("\nPass 1 results:")
    for key in sorted(q01_all.keys()):
        print(f"  {key}: q01={q01_all[key]}, q99={q99_all[key]}")

    # === Pass 2: gripper_action with static filtering ===
    if "gripper_pos" not in q01_all:
        print("\nWARNING: no gripper_pos data found, skipping gripper_action computation.")
    else:
        gp_q01 = q01_all["gripper_pos"][0]
        gp_q99 = q99_all["gripper_pos"][0]
        print(f"\n--- Pass 2: gripper_action (gripper_pos q01={gp_q01:.6g}, q99={gp_q99:.6g}) ---")

        gripper_buffers: list[np.ndarray] = []

        def _worker_pass2(p: Path) -> dict[str, list[np.ndarray]]:
            return process_episode(p, action_offset, action_skip,
                                   gripper_pos_q01=gp_q01, gripper_pos_q99=gp_q99,
                                   static_threshold=args.static_threshold)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker_pass2, p): p for p in json_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Pass 2", ncols=90):
                try:
                    partial = fut.result()
                except Exception as exc:
                    continue
                gripper_buffers.extend(partial["gripper_action"])

        if len(gripper_buffers) > 0:
            combined = np.concatenate(gripper_buffers, axis=0)
            ga_q01 = np.quantile(combined, 0.01, axis=0).astype(np.float64)
            ga_q99 = np.quantile(combined, 0.99, axis=0).astype(np.float64)
            ga_q01, ga_q99 = _symmetric_q01_q99(ga_q01, ga_q99)
            q01_all["gripper_action"] = ga_q01.reshape(-1).tolist()
            q99_all["gripper_action"] = ga_q99.reshape(-1).tolist()
            print(f"  gripper_action: q01={q01_all['gripper_action']}, q99={q99_all['gripper_action']}")
            print(f"  ({combined.shape[0]} non-static samples)")
        else:
            print("  WARNING: no non-static gripper samples found, skipping gripper_action.")

    # === Write output ===
    payload = {"q01": q01_all, "q99": q99_all}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), str(output_path))
    print(f"\nWrote {output_path}")
    print(f"Keys: {sorted(q01_all.keys())}")


if __name__ == "__main__":
    main()
