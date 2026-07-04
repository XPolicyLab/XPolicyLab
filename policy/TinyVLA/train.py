"""XPolicyLab -> TinyVLA training entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = POLICY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(POLICY_DIR / "tinyvla") not in sys.path:
    sys.path.append(str(POLICY_DIR / "tinyvla"))

from XPolicyLab.utils.process_data import get_robot_action_dim_info  # noqa: E402


# Order matters: index 0/1/2 map to TinyVLA's image / image_r / image_top inputs.
# Must stay in sync with process_data.py and deploy.yml.
CAMERA_KEYS = ("cam_left_wrist", "cam_right_wrist", "cam_head")


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
    "pretrain_image_size":       "640",
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
    "evaluation_strategy":        "no",
}


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)

    # XPolicyLab entry-point args
    parser.add_argument("--xpl_dataset_name",    required=True)
    parser.add_argument("--xpl_ckpt_name",       required=True,
                        help="Reused as the TinyVLA TASK_CONFIGS key.")
    parser.add_argument("--xpl_env_cfg_type",    required=True)
    parser.add_argument("--xpl_expert_data_num", required=True, type=int)
    parser.add_argument("--xpl_action_type",     required=True, choices=["joint", "ee"])
    parser.add_argument("--xpl_seed",            required=True, type=int)

    # training schedule
    parser.add_argument("--max_steps",                   required=True)
    parser.add_argument("--per_device_train_batch_size", required=True)
    parser.add_argument("--gradient_accumulation_steps", required=True)
    parser.add_argument("--save_steps",                  required=True)
    parser.add_argument("--save_total_limit",            required=True)
    parser.add_argument("--logging_steps",               required=True)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", default=None)

    args = parser.parse_args(argv[1:])
    return args


def patch_tinyvla(wrapper_args):
    import train_tinyvla

    robot_action_dim_info = get_robot_action_dim_info(wrapper_args.xpl_env_cfg_type)
    action_dim = sum(robot_action_dim_info["arm_dim"]) + sum(robot_action_dim_info["ee_dim"])

    ckpt_setting = (
        f"{wrapper_args.xpl_dataset_name}-{wrapper_args.xpl_ckpt_name}"
        f"-{wrapper_args.xpl_env_cfg_type}-{wrapper_args.xpl_expert_data_num}"
        f"-{wrapper_args.xpl_action_type}"
    )
    data_dir = POLICY_DIR / "data" / ckpt_setting

    train_tinyvla.TASK_CONFIGS[wrapper_args.xpl_ckpt_name] = {
        "dataset_dir": [str(data_dir)],
        # Official TinyVLA load_data() derives true per-episode lengths from HDF5.
        "episode_len": 0,
        "camera_names": list(CAMERA_KEYS),
        "train_ratio": 1.0,
    }

    # Splice action_dim through every layer the official train code reads it from.
    original_parse = train_tinyvla.parse_pythia

    def parse_pythia_with_xpolicylab_dims():
        model_args, data_args, training_args, action_args, config, bnb = original_parse()
        # This adapter only supports the droid_diffusion head (the act head conflicts
        # with the released pre-trained Llava-Pythia VLM configs).
        action_args.action_head_type = "droid_diffusion"
        config.action_head_type = "droid_diffusion"
        action_args.action_dim = action_dim
        action_args.state_dim = action_dim
        config.action_dim = action_dim
        config.state_dim = action_dim
        return model_args, data_args, training_args, action_args, config, bnb

    train_tinyvla.parse_pythia = parse_pythia_with_xpolicylab_dims
    print(
        f"\n[XPolicyLab->TinyVLA] data_dir={data_dir} | "
        f"action_dim={action_dim} | image=640x480\n"
    )
    return train_tinyvla


def main():
    args = parse_wrapper_args(sys.argv)

    ckpt_setting = (
        f"{args.xpl_dataset_name}-{args.xpl_ckpt_name}-{args.xpl_env_cfg_type}"
        f"-{args.xpl_expert_data_num}-{args.xpl_action_type}-{args.xpl_seed}"
    )
    output_dir = POLICY_DIR / "checkpoints" / ckpt_setting

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
