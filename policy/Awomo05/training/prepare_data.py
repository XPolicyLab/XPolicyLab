# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Build a compact trajectory index from local RoboDojo HDF5 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

FIELDS = (
    ("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
    ("right_arm_joint_states", 6), ("right_ee_joint_states", 1),
)
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def read_episode(path: Path) -> tuple[str, torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as handle:
        trajectories = []
        for group in ("action", "state"):
            parts = []
            for name, width in FIELDS:
                part = np.asarray(handle[f"{group}/{name}"], dtype=np.float32)
                if width == 1 and part.ndim == 1:
                    part = part[:, None]
                if part.ndim != 2 or part.shape[1] != width:
                    raise ValueError(f"{path}: {group}/{name} must have shape [N,{width}]")
                parts.append(part)
            if len({part.shape[0] for part in parts}) != 1:
                raise ValueError(f"{path}: inconsistent {group} trajectory lengths")
            values = np.concatenate(parts, axis=1)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {group} trajectory")
            trajectories.append(torch.from_numpy(values))
        lengths = [len(values) for values in trajectories]
        lengths.extend(len(handle[f"vision/{camera}/colors"]) for camera in CAMERAS)
        if len(set(lengths)) != 1 or lengths[0] < 2:
            raise ValueError(f"{path}: inconsistent or too-short episode lengths: {lengths}")
        if "additional_info/frequency" in handle:
            frequency = float(handle["additional_info/frequency"][()])
            if frequency != 25.0:
                raise ValueError(f"{path}: expected 25 Hz, got {frequency}")
        instruction = handle["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{path}: missing nonempty instruction")
    return instruction, trajectories[0], trajectories[1]


def build_index(data_root: Path, episodes_file: Path, output: Path) -> dict:
    data_root, output = data_root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    entries = [line.strip() for line in episodes_file.read_text().splitlines() if line.strip()]
    if not entries or len(set(entries)) != len(entries):
        raise ValueError("Episode list must be nonempty and contain no duplicates")
    episodes, actions, states = [], [], []
    seen, offset = set(), 0
    for entry in entries:
        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 4:
            raise ValueError(f"Expected task/robot/data/episode.hdf5: {entry}")
        if relative.parts[2] != "data" or relative.suffix != ".hdf5":
            raise ValueError(f"Unsupported episode path: {entry}")
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or path in seen:
            raise ValueError(f"Duplicate episode or path outside data root: {entry}")
        seen.add(path)
        instruction, action, state = read_episode(path)
        episodes.append(dict(task=relative.parts[0], path=os.path.relpath(path, output.parent),
                             instruction=instruction, n_frames=len(action), offset=offset))
        actions.append(action)
        states.append(state)
        offset += len(action)
    payload = dict(episodes=episodes, actions=torch.cat(actions), states=torch.cat(states),
                   action_dim=14, state_dim=14, gripper_dims=[6, 13])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=True)
        if restored["episodes"] != episodes or any(
            not torch.equal(restored[key], payload[key]) for key in ("actions", "states")
        ):
            raise RuntimeError("Index serialization verification failed")
        os.link(temporary, output)
    finally:
        temporary.unlink()
    return dict(episodes=len(episodes), tasks=len({e["task"] for e in episodes}), frames=offset,
                trajectory_shape=list(payload["actions"].shape), dtype="float32",
                windows_stride1=sum(e["n_frames"] - 1 for e in episodes),
                episode_list_sha256=hashlib.sha256(episodes_file.read_bytes()).hexdigest(),
                index_bytes=output.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root containing task/robot/data/*.hdf5")
    parser.add_argument("--episodes", type=Path, default=Path(__file__).with_name("episodes.txt"),
                        help="Ordered list of paths relative to --data-root")
    parser.add_argument("--output", type=Path, required=True, help="New index.pt; never overwritten")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(build_index(args.data_root, args.episodes, args.output), indent=2))


if __name__ == "__main__":
    main()
