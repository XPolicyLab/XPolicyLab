"""Prepare MoPA datasets from episodes containing decoded RGB arrays."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np

from ..common import DATASET_FORMAT, joint_fields, rgb_image
from . import mobile


def validate_contract(metadata: dict) -> int:
    """Validate the arm-joint and image layout shared by training and inference."""
    if metadata.get("action_type", "joint") != "joint":
        raise ValueError("MoPA supports joint actions only.")
    if mobile.matches_layout(metadata):
        if metadata.get("state_dim") not in (None, mobile.MODEL_DIM):
            raise ValueError("Mobile state_dim must be 75")
        if metadata.get("action_dim") not in (None, mobile.MODEL_DIM):
            raise ValueError("Mobile action_dim must be 75")
        dim = mobile.MODEL_DIM
    else:
        dim = sum(size for _, size in joint_fields(metadata["robot_action_dim_info"]))
    for name in ("state_dim", "action_dim"):
        if name in metadata and metadata[name] != dim:
            raise ValueError(f"{name} does not match robot_action_dim_info ({dim}).")
    cameras = metadata["cameras"]
    if (not isinstance(cameras, (list, tuple)) or not cameras
            or any(not isinstance(name, str) or not name.strip() for name in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError("Camera names must be nonempty, unique strings.")
    image_size = metadata["image_size"]
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in image_size)):
        raise ValueError("image_size must contain positive integer height and width.")
    return dim


def prepare_episode(episode: dict, metadata: dict) -> dict:
    """Validate one episode and resize uint8 RGB frames without channel conversion."""
    dim = validate_contract(metadata)
    state = np.asarray(episode["state"], dtype=np.float32)
    action = np.asarray(episode["action"], dtype=np.float32)
    if state.ndim != 2 or not len(state) or state.shape[1] != dim or action.shape != state.shape:
        raise ValueError(f"Expected matching nonempty [T,{dim}] state/action arrays.")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State and action must contain finite values.")
    instruction = np.asarray(episode["instruction"])
    if instruction.ndim != 0 or instruction.dtype.kind not in ("U", "S"):
        raise ValueError("instruction must be a scalar string.")
    instruction = instruction.item()
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    if not instruction.strip():
        raise ValueError("instruction must be nonempty.")
    result = {"state": state, "action": action, "instruction": np.asarray(instruction.strip())}
    target_size = tuple(metadata["image_size"])
    for index in range(len(metadata["cameras"])):
        key = f"image_{index}"
        images = np.asarray(episode[key])
        if images.ndim != 4 or images.shape[0] != len(state) or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"{key} must contain uint8 RGB [T,H,W,3] matching the episode length.")
        if min(images.shape[1:3]) <= 0:
            raise ValueError(f"{key} has an empty image dimension.")
        result[key] = (images if images.shape[1:3] == target_size
                       else np.stack([np.asarray(rgb_image(frame, target_size)) for frame in images]))
    return result


def quantile_statistics(values: np.ndarray) -> dict[str, list[float]]:
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def prepare_dataset(episodes: Iterable[dict], metadata: dict, output_dir: str | Path) -> dict:
    """Write normalized-training inputs and statistics from real, unpadded frames.

    Episodes contain state/action arrays, a scalar instruction, and image_0,
    image_1, ... in the camera order declared by metadata. Images are RGB pixels,
    never encoded image buffers. The output directory must not already exist.
    """
    dim = validate_contract(metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records, states, actions = [], [], []
    for index, raw_episode in enumerate(episodes):
        episode = prepare_episode(raw_episode, metadata)
        filename = f"episode_{index:07d}.npz"
        np.savez_compressed(output_dir / filename, **episode)
        records.append({"file": filename, "length": len(episode["state"])})
        states.append(episode["state"])
        actions.append(episode["action"])
        print(f"[MoPA] prepared episode {index + 1}: {len(episode['state'])} frames", flush=True)
    if not records:
        raise ValueError("A dataset must contain at least one nonempty episode.")
    statistics = {"state": quantile_statistics(np.concatenate(states)),
                  "action": quantile_statistics(np.concatenate(actions))}
    result = dict(metadata)
    result.update(format_version=DATASET_FORMAT, action_type="joint", action_dim=dim, state_dim=dim,
                  cameras=list(metadata["cameras"]), image_size=list(metadata["image_size"]),
                  episodes=records, num_frames=sum(record["length"] for record in records))
    for name, value in (("metadata.json", result), ("dataset_statistics.json", statistics)):
        (output_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return result


def _decode_scalar(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    text = str(value).strip()
    if not text:
        raise ValueError("instruction must be nonempty")
    return text


def _reservoir_update(reservoir: np.ndarray, seen: int, values: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Bound statistics memory while retaining an unbiased row sample."""
    if values.ndim != 2:
        raise ValueError(f"Expected [T,D] values, got {values.shape}")
    limit = reservoir.shape[0]
    for row in values:
        seen += 1
        if seen <= limit:
            reservoir[seen - 1] = row
        else:
            slot = int(rng.integers(0, seen))
            if slot < limit:
                reservoir[slot] = row
    return reservoir, seen


def _update_moments(count, mean, m2, lower, upper, values):
    """Merge one frame block into exact per-dimension moments and extrema."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"Expected a nonempty [T,D] block, got {values.shape}")
    block_count = values.shape[0]
    block_mean = values.mean(axis=0)
    centered = values - block_mean
    block_m2 = np.sum(centered * centered, axis=0)
    total = count + block_count
    delta = block_mean - mean
    correction = delta * delta * count * block_count / max(total, 1)
    mean = mean + delta * block_count / max(total, 1)
    m2 = m2 + block_m2 + correction
    lower = np.minimum(lower, values.min(axis=0))
    upper = np.maximum(upper, values.max(axis=0))
    return total, mean, m2, lower, upper


def _minmax_statistics(count, mean, m2, lower, upper, sample, seen):
    """Serialize exact min/max statistics and sampled quantile diagnostics."""
    variance = m2 / max(count - 1, 1)
    return {
        "count": [int(count)] * len(mean),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
        "min": lower.tolist(),
        "max": upper.tolist(),
        "q01": np.quantile(sample[:min(seen, len(sample))], 0.01, axis=0).tolist(),
        "q99": np.quantile(sample[:min(seen, len(sample))], 0.99, axis=0).tolist(),
    }


def prepare_mobile_dataset(source_dir: str | Path, metadata: dict,
                            output_dir: str | Path, *, episode_limit: int | None = None,
                            statistics_samples: int = 200_000) -> dict:
    """Index source HDF5 episodes for lazy frame reads and bounded-memory statistics."""
    try:
        import h5py  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only reached in a broken env
        raise RuntimeError("Mobile conversion requires h5py") from exc
    dim = validate_contract(metadata)
    if not mobile.matches_layout(metadata) or dim != mobile.MODEL_DIM:
        raise ValueError("prepare_mobile_dataset requires the Mobile contract")
    source_dir = Path(source_dir).expanduser()
    files = sorted(set(source_dir.rglob("*.hdf5")) | set(source_dir.rglob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No Mobile HDF5 episodes found below {source_dir}")
    if episode_limit is not None:
        if not 1 <= episode_limit <= len(files):
            raise ValueError(f"episode_limit must be in [1, {len(files)}]")
        files = files[:episode_limit]
    if statistics_samples <= 0:
        raise ValueError("statistics_samples must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    state_sample = np.empty((statistics_samples, dim), dtype=np.float32)
    action_sample = np.empty((statistics_samples, dim), dtype=np.float32)
    state_mean = np.zeros(dim, dtype=np.float64)
    action_mean = np.zeros(dim, dtype=np.float64)
    state_m2 = np.zeros(dim, dtype=np.float64)
    action_m2 = np.zeros(dim, dtype=np.float64)
    state_min = np.full(dim, np.inf, dtype=np.float64)
    action_min = np.full(dim, np.inf, dtype=np.float64)
    state_max = np.full(dim, -np.inf, dtype=np.float64)
    action_max = np.full(dim, -np.inf, dtype=np.float64)
    state_count = action_count = 0
    state_seen = action_seen = 0
    records = []
    import h5py
    for index, path in enumerate(files):
        with h5py.File(path, "r") as handle:
            if "state" not in handle or "action" not in handle:
                raise ValueError(f"{path}: expected state/ and action/ groups")
            state = mobile.pack({key: handle[f"state/{key}"][()] for key, _, _ in mobile.KEYS})
            action = mobile.pack({key: handle[f"action/{key}"][()] for key, _, _ in mobile.KEYS})
            if state.ndim != 2 or action.shape != state.shape or not len(state):
                raise ValueError(f"{path}: invalid state/action shapes {state.shape}/{action.shape}")
            instruction = _decode_scalar(handle["instruction"][()])

            for camera in metadata["cameras"]:
                if f"vision/{camera}/colors" not in handle:
                    raise ValueError(f"{path}: missing vision/{camera}/colors")
                if len(handle[f"vision/{camera}/colors"]) != len(state):
                    raise ValueError(f"{path}: camera {camera} length differs from state")
        state_sample, state_seen = _reservoir_update(state_sample, state_seen, state, rng)
        action_sample, action_seen = _reservoir_update(action_sample, action_seen, action, rng)
        state_count, state_mean, state_m2, state_min, state_max = _update_moments(
            state_count, state_mean, state_m2, state_min, state_max, state
        )
        action_count, action_mean, action_m2, action_min, action_max = _update_moments(
            action_count, action_mean, action_m2, action_min, action_max, action
        )
        records.append({"file": str(path.resolve()), "length": int(len(state)),
                        "instruction": instruction})
        print(f"[MoPA] indexed Mobile episode {index + 1}/{len(files)}: {len(state)} frames", flush=True)
    statistics = {
        "normalization": "mmabc_minmax",
        "state": _minmax_statistics(state_count, state_mean, state_m2, state_min,
                                     state_max, state_sample, state_seen),
        "action": _minmax_statistics(action_count, action_mean, action_m2, action_min,
                                      action_max, action_sample, action_seen),
    }
    result = dict(metadata)
    result.update(format_version=DATASET_FORMAT, action_type="joint", action_dim=dim,
                  state_dim=dim, storage_format="mobile_hdf5", episodes=records,
                  num_frames=sum(record["length"] for record in records),
                  statistics_samples=statistics_samples)
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    return result


def load_episodes(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            yield {key: archive[key] for key in archive.files}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="An RGB-array NPZ episode or directory of episodes.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path,
                        help="JSON with robot_action_dim_info, cameras, and image_size.")
    args = parser.parse_args(argv)
    if args.source.is_file():
        files = [args.source]
    else:
        files = sorted(args.source.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ episodes found at {args.source}.")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = prepare_dataset(load_episodes(files), metadata, args.output)
    print(f"[MoPA] saved {result['num_frames']} frames to {args.output}")


if __name__ == "__main__":
    main()
