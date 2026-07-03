"""XPolicyLab -> TinyVLA training adapter."""
import argparse
import json
import random
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


POLICY_DIR = Path(__file__).resolve().parent
if str(POLICY_DIR / "tinyvla") not in sys.path:
    sys.path.append(str(POLICY_DIR / "tinyvla"))


CAMERA_KEYS = ("cam_left_wrist", "cam_right_wrist")  # no need to change
TARGET_SIZE = (640, 480)  # (W, H) for cv2.resize


SENIOR_TRAIN_ARGS = {
    # optimizer / LoRA / data loading
    "learning_rate":             "2e-4",
    "non_lora_lr":               "2e-5",
    "lr_scheduler_type":         "cosine",
    "warmup_ratio":              "0.005",
    "weight_decay":              "0.",
    "lora_enable":               "True",
    "lora_module":               "vit llm",
    "lora_r":                    "64",
    "lora_alpha":                "256",
    "dataloader_num_workers":    "8",
    "gradient_checkpointing":    "True",
    # model architecture / framework constants
    "save_strategy":             "steps",
    "bf16":                      "True",
    "tf32":                      "True",
    "model_max_length":          "2048",
    "lazy_preprocess":           "True",
    "action_head_type":          "droid_diffusion",
    "concat":                    "token_cat",
    "pretrain_image_size":       "320",
    "load_pretrain":             "False",
    "tune_mm_mlp_adapter":       "True",
    "freeze_vision_tower":       "True",
    "freeze_backbone":           "True",
    "mm_use_im_start_end":       "False",
    "mm_use_im_patch_token":     "False",
    "image_aspect_ratio":        "pad",
    "group_by_modality_length":  "False",
    "version":                   "v0",
    "report_to":                 "tensorboard",
}


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
        instructions,
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
        self.instructions = instructions
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
            "raw_lang": random.choice(self.instructions[episode_id]),
        }
        return self.llava_pythia_process.forward_process(sample)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def parse_wrapper_args(argv):
    p = argparse.ArgumentParser(add_help=False)

    # XPolicyLab entry-point args
    p.add_argument("--xpl_dataset_name",    required=True)
    p.add_argument("--xpl_ckpt_name",       required=True,
                   help="Reused as the TinyVLA TASK_CONFIGS key (cotrain-friendly).")
    p.add_argument("--xpl_env_cfg_type",    required=True)
    p.add_argument("--xpl_expert_data_num", required=True, type=int)
    p.add_argument("--xpl_action_type",     required=True, choices=["joint", "ee"])
    p.add_argument("--xpl_seed",            required=True, type=int)

    # training schedule 
    p.add_argument("--max_steps",                   required=True)
    p.add_argument("--per_device_train_batch_size", required=True)
    p.add_argument("--gradient_accumulation_steps", required=True)
    p.add_argument("--save_steps",                  required=True)
    p.add_argument("--save_total_limit",            required=True)
    p.add_argument("--logging_steps",               required=True)

    args, _ = p.parse_known_args(argv[1:])
    return args


def processed_dataset_dir(args):
    ckpt_setting = (
        f"{args.xpl_dataset_name}-{args.xpl_ckpt_name}-{args.xpl_env_cfg_type}"
        f"-{args.xpl_expert_data_num}-{args.xpl_action_type}"
    )
    return POLICY_DIR / "data" / ckpt_setting


def list_episode_paths(args):
    data_dir = processed_dataset_dir(args)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Processed data dir not found: {data_dir}. "
            "Run process_data.sh before training."
        )
    paths = sorted(data_dir.glob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(
            f"No episode_*.hdf5 found under {data_dir}. "
            "Did process_data.sh complete successfully?"
        )
    return paths


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

        if "instructions" in root:
            instructions = json.loads(root["instructions"][()].decode("utf-8"))
        else:
            raw = root["instruction"][()]
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            instructions = [text]
            
    return state, action, instructions


def load_images(path, start_ts):
    images = []
    with h5py.File(path, "r") as root:
        for cam_key in CAMERA_KEYS:
            raw = root["vision"][cam_key]["colors"][start_ts]
            if isinstance(rgb, (bytes, bytearray, np.bytes_)):
                rgb = decode_image_bit(rgb)
            elif rgb.ndim == 1:
                rgb = decode_image_bit(rgb)
            else:
                rgb = raw
            rgb = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            images.append(rgb)
    return np.stack(images, axis=0)


def load_all_episodes(episode_paths, action_type, robot_action_dim_info):
    """Single pass over disk: returns per-episode state/action/instructions and norm stats."""
    states, actions, instructions, episode_lens = {}, {}, {}, []
    for ep_id, path in enumerate(episode_paths):
        state, action, ep_instructions = load_episode(
            path, action_type, robot_action_dim_info
        )
        states[ep_id] = state
        actions[ep_id] = action
        instructions[ep_id] = ep_instructions
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
    return states, actions, instructions, episode_lens, norm_stats


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------
def make_load_data(episode_paths, states, actions, instructions, episode_lens, norm_stats, action_dim):
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
                instructions=instructions,
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
    states, actions, instructions, episode_lens, norm_stats = load_all_episodes(
        episode_paths, wrapper_args.xpl_action_type, robot_action_dim_info
    )

    task_name = get_forwarded_arg("--task_name")
    train_tinyvla.TASK_CONFIGS[task_name] = {
        "dataset_dir": [str(processed_dataset_dir(wrapper_args))],
        "episode_len": int(max(episode_lens)),
        "camera_names": list(CAMERA_KEYS),
    }
    train_tinyvla.load_data = make_load_data(
        episode_paths, states, actions, instructions, episode_lens, norm_stats, action_dim
    )
    detr_vae.IN_DIM_STATE = action_dim
    detr_vae.IN_DIM_ACTION = action_dim

    # Splice action_dim through every layer the official train code reads it from.
    original_parse = train_tinyvla.parse_pythia

    def parse_pythia_with_xpolicylab_dims():
        model_args, data_args, training_args, action_args, config, bnb = original_parse()
        # We found that action_head_type="act" conflicts with the released pre-trained Llava-Pythia VLM configs
        # lock the action head type to droid_diffusion
        action_args.action_head_type = "droid_diffusion"
        config.action_head_type = "droid_diffusion"
        action_args.action_dim = action_dim
        action_args.state_dim = action_dim
        config.action_dim = action_dim
        config.state_dim = action_dim
        return model_args, data_args, training_args, action_args, config, bnb

    train_tinyvla.parse_pythia = parse_pythia_with_xpolicylab_dims
    return train_tinyvla


def main():
    args = parse_wrapper_args(sys.argv)

    # path: training output directory
    output_dir = POLICY_DIR / "checkpoints" / (
        f"{args.xpl_dataset_name}-{args.xpl_ckpt_name}-{args.xpl_env_cfg_type}"
        f"-{args.xpl_expert_data_num}-{args.xpl_action_type}-{args.xpl_seed}"
    )


    train_args = {
        **SENIOR_TRAIN_ARGS,
        "max_steps":                   args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_steps":                  args.save_steps,
        "save_total_limit":            args.save_total_limit,
        "logging_steps":               args.logging_steps,
        "output_dir":                  str(output_dir),
        "logging_dir":                 str(output_dir / "log"),
        "model_name_or_path":          str(output_dir / "pretrained_vlm"),
        "deepspeed":                   str(POLICY_DIR / "tinyvla" / "llava-pythia" / "scripts" / "zero2.json"),
        "task_name":                   args.xpl_ckpt_name,
        "seed":                        str(args.xpl_seed),
    }

    # Flatten {"--key": "val", ...} into the sys.argv form HfArgumentParser expects.
    sys.argv = [sys.argv[0]]
    for key, val in train_args.items():
        sys.argv += [f"--{key}", val]

    train_tinyvla = patch_tinyvla(args)
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
