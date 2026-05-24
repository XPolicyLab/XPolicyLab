"""XPolicyLab → TinyVLA training adapter.

The official entrypoint is ``policy/TinyVLA/tinyvla/train_tinyvla.py``. We do
not modify it; instead this wrapper:

  1. Reads the XPolicyLab v1.0 hdf5 episodes directly (no ACT-style conversion).
  2. Patches ``train_tinyvla.TASK_CONFIGS`` / ``load_data`` / ``parse_pythia``
     to redirect data loading and propagate the env-cfg-derived action_dim.
  3. Hands control to ``train_tinyvla.main``.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

POLICY_DIR = Path(__file__).resolve().parent
TINYVLA_DIR = POLICY_DIR / "tinyvla"
if str(TINYVLA_DIR) not in sys.path:
    sys.path.append(str(TINYVLA_DIR))


# llava_pythia.process_batch_to_llava splits the camera dim with
# ``torch.chunk(curr_image, 2, dim=0)``, so exactly two cameras are required.
CAMERA_KEYS = ("cam_left_wrist", "cam_right_wrist")
TARGET_SIZE = (320, 240)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class XPolicyLabTinyVLADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_paths,
        episode_ids,
        episode_lens,
        states,
        actions,
        raw_langs,
        norm_stats,
        chunk_size,
        llava_pythia_process,
    ):
        super().__init__()
        self.episode_paths = episode_paths
        self.episode_ids = list(episode_ids)
        self.episode_lens = list(episode_lens)
        self.cumulative_len = np.cumsum(self.episode_lens)
        self.max_episode_len = max(self.episode_lens)
        self.states = states
        self.actions = actions
        self.raw_langs = raw_langs
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.llava_pythia_process = llava_pythia_process

    def __len__(self):
        return int(self.cumulative_len[-1])

    def _locate_transition(self, index):
        offset = int(np.argmax(self.cumulative_len > index))
        start_ts = index - (self.cumulative_len[offset] - self.episode_lens[offset])
        return self.episode_ids[offset], int(start_ts)

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)

        qpos = self.states[episode_id][start_ts]
        # ACT-style "+1 action lag" hack from upstream EpisodicDataset.
        action_start = max(0, start_ts - 1)
        action = self.actions[episode_id][action_start:]
        action_len = len(action)

        padded_action = np.zeros(
            (self.max_episode_len, action.shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len, dtype=bool)
        is_pad[action_len:] = True

        image_data = load_images(self.episode_paths[episode_id], start_ts)
        image_data = torch.einsum("k h w c -> k c h w", torch.from_numpy(image_data))
        image_data = image_data.float() / 255.0

        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        padded_action = (
            padded_action - self.norm_stats["action_mean"]
        ) / self.norm_stats["action_std"]

        sample = {
            "image": image_data,
            "state": torch.from_numpy(qpos).float(),
            "action": torch.from_numpy(padded_action[: self.chunk_size]).float(),
            "is_pad": torch.from_numpy(is_pad[: self.chunk_size]),
            "raw_lang": self.raw_langs[episode_id],
        }
        return self.llava_pythia_process.forward_process(sample)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--xpl_dataset_name", required=True)
    parser.add_argument("--xpl_task_name", required=True)
    parser.add_argument("--xpl_env_cfg_type", required=True)
    parser.add_argument("--xpl_expert_data_num", required=True, type=int)
    parser.add_argument("--xpl_action_type", required=True, choices=["joint", "ee"])
    # Forwarded by train.sh but not declared in TinyVLA's HfArgumentParser.
    parser.add_argument("--use_state")
    parser.add_argument("--window_size")
    wrapper_args, remaining = parser.parse_known_args(argv[1:])
    sys.argv = [argv[0], *remaining]
    return wrapper_args


def raw_data_dir(args):
    return (
        POLICY_DIR.parents[2]
        / "data"
        / args.xpl_dataset_name
        / args.xpl_task_name
        / args.xpl_env_cfg_type
        / "data"
    )


def list_episode_paths(args):
    data_dir = raw_data_dir(args)
    return [data_dir / f"episode_{i:07d}.hdf5" for i in range(args.xpl_expert_data_num)]


def load_episode(path, action_type, robot_action_dim_info):
    with h5py.File(path, "r") as root:
        data = {
            "state": {k: root["state"][k][()] for k in root["state"].keys()},
            "action": {k: root["action"][k][()] for k in root["action"].keys()},
        }
        state = pack_robot_state(
            data,
            action_type=action_type,
            robot_action_dim_info=robot_action_dim_info,
            source_type="dataset",
            state_type="state",
        ).astype(np.float32)
        action = pack_robot_state(
            data,
            action_type=action_type,
            robot_action_dim_info=robot_action_dim_info,
            source_type="dataset",
            state_type="action",
        ).astype(np.float32)
        raw_lang = json.loads(root["instructions"][()].decode("utf-8"))[0]
    return state, action, raw_lang


def load_images(path, start_ts):
    images = []
    with h5py.File(path, "r") as root:
        for cam_key in CAMERA_KEYS:
            raw = root["vision"][cam_key]["colors"][start_ts]
            bgr = raw if raw.ndim == 3 else decode_image_bit(raw)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            images.append(rgb)
    return np.stack(images, axis=0)


def load_all_episodes(episode_paths, action_type, robot_action_dim_info):
    """Single pass over disk: returns per-episode state/action/lang and norm stats."""
    states, actions, raw_langs, episode_lens = {}, {}, {}, []
    for ep_id, path in enumerate(episode_paths):
        state, action, raw_lang = load_episode(path, action_type, robot_action_dim_info)
        states[ep_id] = state
        actions[ep_id] = action
        raw_langs[ep_id] = raw_lang
        episode_lens.append(len(state))

    all_qpos = np.concatenate([states[i] for i in range(len(episode_paths))], axis=0)
    all_actions = np.concatenate([actions[i] for i in range(len(episode_paths))], axis=0)
    eps = 1e-4
    norm_stats = {
        "qpos_mean": all_qpos.mean(axis=0).astype(np.float32),
        "qpos_std": np.clip(all_qpos.std(axis=0), 1e-2, np.inf).astype(np.float32),
        "action_mean": all_actions.mean(axis=0).astype(np.float32),
        "action_std": np.clip(all_actions.std(axis=0), 1e-2, np.inf).astype(np.float32),
        "action_min": (all_actions.min(axis=0) - eps).astype(np.float32),
        "action_max": (all_actions.max(axis=0) + eps).astype(np.float32),
        "example_qpos": all_qpos[-1],
    }
    return states, actions, raw_langs, episode_lens, norm_stats


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------
def make_load_data(episode_paths, states, actions, raw_langs, episode_lens, norm_stats, action_dim):
    def load_xpolicylab_data(
        dataset_dir_l, name_filter, camera_names,
        batch_size_train, batch_size_val, chunk_size,
        config, skip_mirrored_data=False, policy_class=None, stats_dir_l=None,
        sample_weights=None, train_ratio=0.95, return_dataset=False,
        llava_pythia_process=None,
    ):
        n = len(episode_paths)
        shuffled = np.random.permutation(n)
        split = int(train_ratio * n)
        train_ids, val_ids = shuffled[:split], shuffled[split:]

        def make_dataset(ids):
            return XPolicyLabTinyVLADataset(
                episode_paths=episode_paths,
                episode_ids=ids,
                episode_lens=[episode_lens[i] for i in ids],
                states=states,
                actions=actions,
                raw_langs=raw_langs,
                norm_stats=norm_stats,
                chunk_size=chunk_size,
                llava_pythia_process=llava_pythia_process,
            )

        train_dataset = make_dataset(train_ids)
        val_dataset = make_dataset(val_ids)

        sampler_params = {
            "train": {
                "batch_size": batch_size_train,
                "episode_len_l": [train_dataset.episode_lens],
                "sample_weights": sample_weights,
            },
            "eval": {
                "batch_size": batch_size_val,
                "episode_len_l": [val_dataset.episode_lens],
                "sample_weights": None,
            },
        }
        print(
            f"\n[XPolicyLab→TinyVLA] {n} episodes | "
            f"train={len(train_ids)} val={len(val_ids)} | action_dim={action_dim}\n"
        )
        return train_dataset, val_dataset, norm_stats, sampler_params

    return load_xpolicylab_data


def get_forwarded_arg(name):
    for idx, arg in enumerate(sys.argv):
        if arg == name:
            return sys.argv[idx + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    raise ValueError(f"{name} must be forwarded to TinyVLA training.")


def patch_tinyvla(wrapper_args):
    import train_tinyvla
    import policy_heads.models.detr_vae as detr_vae

    robot_action_dim_info = get_robot_action_dim_info(wrapper_args.xpl_env_cfg_type)
    action_dim = sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])

    episode_paths = list_episode_paths(wrapper_args)
    states, actions, raw_langs, episode_lens, norm_stats = load_all_episodes(
        episode_paths, wrapper_args.xpl_action_type, robot_action_dim_info
    )

    task_name = get_forwarded_arg("--task_name")
    train_tinyvla.TASK_CONFIGS[task_name] = {
        "dataset_dir": [str(raw_data_dir(wrapper_args))],
        "episode_len": int(max(episode_lens)),
        "camera_names": list(CAMERA_KEYS),
    }
    train_tinyvla.load_data = make_load_data(
        episode_paths, states, actions, raw_langs, episode_lens, norm_stats, action_dim
    )
    detr_vae.IN_DIM_STATE = action_dim
    detr_vae.IN_DIM_ACTION = action_dim

    # Splice action_dim through every layer the official train code reads it from.
    original_parse = train_tinyvla.parse_pythia

    def parse_pythia_with_xpolicylab_dims():
        model_args, data_args, training_args, action_args, config, bnb = original_parse()
        action_args.action_dim = action_dim
        action_args.state_dim = action_dim
        config.action_dim = action_dim
        config.state_dim = action_dim
        config.act["act"]["action_dim"] = action_dim
        config.act["act"]["camera_names"] = list(CAMERA_KEYS)
        config.act["act"]["chunk_size"] = action_args.chunk_size
        return model_args, data_args, training_args, action_args, config, bnb

    train_tinyvla.parse_pythia = parse_pythia_with_xpolicylab_dims
    return train_tinyvla


def main():
    wrapper_args = parse_wrapper_args(sys.argv)
    train_tinyvla = patch_tinyvla(wrapper_args)
    model_args, data_args, training_args, action_args, llava_pythia_config, bnb = (
        train_tinyvla.parse_pythia()
    )
    train_tinyvla.main(
        config={
            "model_args": model_args,
            "data_args": data_args,
            "training_args": training_args,
            "action_args": action_args,
            "bnb_model_from_pretrained_args": bnb,
        },
        llava_pythia_config=llava_pythia_config,
    )


if __name__ == "__main__":
    main()
