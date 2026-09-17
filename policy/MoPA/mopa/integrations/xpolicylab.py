"""Optional XPolicyLab observation and command-line integration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ..common import DEFAULT_CAMERAS, joint_fields, pack_joint, unpack_joint
from ..data.prepare import prepare_dataset
from ..runtime import Policy


class ModelAdapter:
    """Translate XPolicyLab observation dictionaries to the standalone policy API."""

    def __init__(self, checkpoint, dim_info, model_cfg):
        self.dim_info = dim_info
        self.policy = Policy(checkpoint, device=model_cfg.get("device", "cuda"),
                             base_vlm=model_cfg.get("base_vlm"), dtype=model_cfg.get("dtype"))
        dim = sum(size for _, size in joint_fields(dim_info))
        for key, expected in (("env_cfg_type", model_cfg["env_cfg_type"]), ("action_type", "joint"),
                              ("action_dim", dim), ("state_dim", dim), ("robot_action_dim_info", dim_info)):
            if self.policy.config.get(key) != expected:
                raise ValueError(f"Checkpoint {key}={self.policy.config.get(key)!r}, runtime requires {expected!r}")
        horizon = int(self.policy.config["action_horizon"])
        steps = model_cfg.get("execute_steps")
        self.execute_steps = horizon if steps is None else int(steps)
        if not 1 <= self.execute_steps <= horizon:
            raise ValueError(f"execute_steps must be in [1, {horizon}]")
        self.reset()

    def reset(self):
        self._observation = None
        self._batch = []

    def update_obs(self, obs):
        self._observation = obs

    def update_obs_batch(self, obs_list):
        if not isinstance(obs_list, (list, tuple)):
            raise TypeError("obs_list must be a list of observations")
        self._batch = list(obs_list)

    def _predict(self, observations):
        if not observations:
            return []
        states = np.stack([pack_joint(obs["state"], self.dim_info) for obs in observations])
        images = [[obs["vision"][camera]["color"] for camera in self.policy.cameras] for obs in observations]
        instructions = [instruction_text(obs) for obs in observations]
        actions = self.policy.predict(images, instructions, states)
        return [[unpack_joint(step, self.dim_info) for step in chunk[:self.execute_steps]] for chunk in actions]

    def get_action(self):
        if self._observation is None:
            raise RuntimeError("Call update_obs before get_action")
        return self._predict([self._observation])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._batch:
            if env_idx_list is not None and len(env_idx_list) == 0:
                return []
            raise RuntimeError("Call update_obs_batch before get_action_batch")
        observations = self._batch
        if env_idx_list is not None:
            indices = list(env_idx_list)
            if not indices:
                return []
            if len(set(indices)) != len(indices):
                raise ValueError("env_idx_list contains duplicate environment indices")
            has_indices = ["env_idx" in obs for obs in observations]
            if any(has_indices) and not all(has_indices):
                raise ValueError("Batch observations must either all carry env_idx or all omit it")
            if all(has_indices):
                by_index = {obs["env_idx"]: obs for obs in observations}
                if len(by_index) != len(observations):
                    raise ValueError("Observations contain duplicate env_idx values")
                observations = [by_index[index] for index in indices]
            elif len(indices) != len(observations):
                raise ValueError("Without env_idx, observations must match the requested batch order and length")
        return self._predict(observations)


def instruction_text(data, fallback=None):
    """Select the first instruction variant for both conversion and inference."""
    for value in (data.get("instruction"), data.get("instructions"), fallback):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    value = parsed
            except json.JSONDecodeError:
                pass
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "No nonempty instruction; provide instruction or instructions "
        "(or --instruction during data conversion)"
    )


def convert_episode(source, dim_info, cameras, instruction=None):
    # The shared decoder owns XPolicyLab's image-buffer formats and RGB handling.
    from XPolicyLab.utils.data_loader import load_xspark_v1
    from XPolicyLab.utils.process_data import decode_image_bit

    data = load_xspark_v1(str(source), decode_images=False)
    episode = {"state": pack_joint(data["state"], dim_info, plural=True),
               "action": pack_joint(data["action"], dim_info, plural=True),
               "instruction": np.asarray(instruction_text(data, instruction))}
    for index, camera in enumerate(cameras):
        colors = decode_image_bit(data["vision"][camera]["colors"])
        if isinstance(colors, np.ndarray) and colors.ndim == 3:
            colors = colors[None]
        episode[f"image_{index}"] = np.asarray(colors)
    return episode


def convert_dataset(source_dir, output_dir, env_cfg_type, *, cameras=None,
                    image_size=(224, 224), episode_limit=None, instruction=None):
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    source_dir = Path(source_dir)
    files = sorted(set(source_dir.rglob("*.hdf5")) | set(source_dir.rglob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 episodes found under {source_dir}")
    if episode_limit is not None:
        if not 1 <= episode_limit <= len(files):
            raise ValueError(f"episode_limit must be in [1, {len(files)}]")
        files = files[:episode_limit]
    dim_info = get_robot_action_dim_info(env_cfg_type)
    cameras = list(DEFAULT_CAMERAS if cameras is None else cameras)
    metadata = {"env_cfg_type": env_cfg_type, "action_type": "joint",
                "robot_action_dim_info": dim_info, "cameras": cameras,
                "image_size": list(image_size), "instruction_selection": "first_variant"}
    episodes = (convert_episode(source, dim_info, cameras, instruction) for source in files)
    return prepare_dataset(episodes, metadata, output_dir)


def process_data_main(argv=None, *, policy_dir, xpl_root):
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

    parser = argparse.ArgumentParser(description="Convert XPolicyLab HDF5 trajectories to MoPA RGB episodes")
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("expert_data_num", type=int, nargs="?")
    parser.add_argument("--source", type=Path, default=os.environ.get("SOURCE_DATA"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--instruction", help="Fallback when the source has no instruction")
    args = parser.parse_args(argv)
    source = args.source or Path(xpl_root).parent / "data" / args.bench_name / args.ckpt_name / args.env_cfg_type / "data"
    output = args.output or Path(policy_dir) / "data" / build_run_dir_name(vars(args), include_seed=False)
    metadata = convert_dataset(source, output, args.env_cfg_type, cameras=args.cameras,
                               image_size=args.image_size, episode_limit=args.expert_data_num,
                               instruction=args.instruction)
    print(f"[MoPA] saved {metadata['num_frames']} frames to {output}")


def train_main(argv=None, *, policy_dir, xpl_root):
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    from ..data.dataset import EpisodeDataset
    from ..training.cli import build_parser, run_training

    parser = argparse.ArgumentParser(parents=[build_parser()], conflict_handler="resolve",
                                     description="Train MoPA with XPolicyLab run naming and robot dimensions")
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("seed", type=int)
    # Replace the native optional seed with the standard positional run seed.
    parser.add_argument("--seed", dest="seed_override", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--action-dim", type=int, help="From the shared get_action_dim.sh helper")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.seed_override is not None:
        parser.error("Use the positional seed argument")
    args.dataset = args.dataset or Path(policy_dir) / "data" / build_run_dir_name(vars(args), include_seed=False)
    args.output = args.output or Path(policy_dir) / "checkpoints" / build_run_dir_name(vars(args))
    dim = args.action_dim
    if dim is None:
        dim = int(subprocess.check_output(["bash", str(Path(xpl_root) / "utils/get_action_dim.sh"),
                                          str(Path(xpl_root).parent), args.env_cfg_type], text=True).strip())
    dataset = EpisodeDataset(args.dataset, cache_episodes=args.cache_episodes)
    if dataset.metadata.get("env_cfg_type") != args.env_cfg_type or dataset.dim != dim:
        raise ValueError("Dataset robot/dimensions disagree with the training environment and shared dimension helper")
    run_training(args)
