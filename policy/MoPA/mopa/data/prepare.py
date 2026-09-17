"""Prepare MoPA datasets from episodes containing decoded RGB arrays."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np

from ..common import DATASET_FORMAT, joint_fields, rgb_image


def validate_contract(metadata: dict) -> int:
    """Validate the arm-joint and image layout shared by training and inference."""
    if metadata.get("action_type", "joint") != "joint":
        raise ValueError("MoPA supports joint actions only.")
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
