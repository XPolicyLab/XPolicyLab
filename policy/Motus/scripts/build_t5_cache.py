#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import torch


POLICY_DIR = Path(__file__).resolve().parents[1]
MOTUS_ROOT = POLICY_DIR / "motus"
ROBOTWIN_MOTUS_ROOT = MOTUS_ROOT / "inference" / "robotwin" / "Motus"
BAK_ROOT = ROBOTWIN_MOTUS_ROOT / "bak"

for path in (ROBOTWIN_MOTUS_ROOT, BAK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wan.modules.t5 import T5EncoderModel  # noqa: E402


SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def normalize_instruction(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return str(value).strip()


def read_unique_instructions(data_root: Path) -> list[dict[str, str]]:
    """Collect every unique instruction string found under each task directory."""
    items: list[dict[str, str]] = []
    for task_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        episode_files = sorted(task_dir.glob("*/data/episode_*.hdf5"))
        if not episode_files:
            continue

        seen: set[str] = set()
        for episode_file in episode_files:
            with h5py.File(episode_file, "r") as h5_file:
                instruction = normalize_instruction(h5_file["instruction"][()])
            if instruction in seen:
                continue
            seen.add(instruction)
            items.append(
                {
                    "task": task_dir.name,
                    "instruction": instruction,
                    "source": str(episode_file.relative_to(data_root)),
                }
            )
    return items


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt_variants(instruction: str, include_raw: bool, include_scene_prefix: bool) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    if include_raw:
        variants.append({"variant": "raw", "prompt": instruction})
    if include_scene_prefix:
        variants.append({"variant": "scene_prefix", "prompt": f"{SCENE_PREFIX}{instruction}"})
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Motus pre-encoded T5 cache from RoboDojo HDF5 instructions.")
    parser.add_argument(
        "--data-root",
        default="/mnt/xspark-data/final_data/RoboDojo_first100",
        help="Read-only RoboDojo data root. All unique instructions per task are collected.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(POLICY_DIR / "t5_cache" / "RoboDojo_first100"),
        help="Project-local directory where embeddings and manifest.json are written.",
    )
    parser.add_argument(
        "--wan-path",
        default="/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B",
        help="WAN model directory containing the T5 checkpoint and tokenizer.",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device used to run the T5 encoder (must be a GPU).")
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--raw-only", action="store_true", help="Only encode raw instructions.")
    parser.add_argument("--scene-prefix-only", action="store_true", help="Only encode scene-prefixed prompts.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_only and args.scene_prefix_only:
        raise ValueError("--raw-only and --scene-prefix-only are mutually exclusive")

    include_raw = not args.scene_prefix_only
    include_scene_prefix = not args.raw_only

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for T5 encoding but torch.cuda.is_available() is False.")

    gpu_device = torch.device(args.device)
    if gpu_device.type != "cuda":
        raise ValueError(f"--device must be a CUDA device, got {args.device!r}")

    task_items = read_unique_instructions(data_root)
    if not task_items:
        raise RuntimeError(f"No task instructions found under {data_root}")

    # Build and load the T5 encoder on CPU first. The WAN wrapper constructs the full
    # umt5-xxl in fp32 under `with torch.device(device)` before casting to bf16, so
    # initializing directly on the GPU peaks at the fp32 footprint (~22GB) and OOMs a
    # 24GB card. Constructing on CPU keeps the fp32 peak in host RAM; we then move the
    # bf16 model (~11GB) onto a single GPU, which fits comfortably.
    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=os.path.join(args.wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.wan_path, "google", "umt5-xxl"),
    )
    encoder.model = encoder.model.to(gpu_device)
    encoder.device = gpu_device
    print(f"T5 encoder moved to {gpu_device}; running encoding on GPU.", flush=True)

    manifest: dict[str, Any] = {
        "data_root": str(data_root),
        "wan_path": str(Path(args.wan_path).expanduser().resolve()),
        "text_len": args.text_len,
        "entries": {},
        "tasks": task_items,
    }

    seen_prompts: dict[str, dict[str, str]] = {}
    for item in task_items:
        for variant in build_prompt_variants(item["instruction"], include_raw, include_scene_prefix):
            prompt = variant["prompt"]
            key = cache_key(prompt)
            if key in seen_prompts:
                continue

            encoded = encoder([prompt], gpu_device)
            if isinstance(encoded, torch.Tensor):
                embedding = encoded.squeeze(0) if encoded.dim() == 3 else encoded
            elif isinstance(encoded, list) and len(encoded) == 1:
                embedding = encoded[0]
            else:
                raise ValueError(f"Unexpected T5 output for task={item['task']}: {type(encoded)}")

            filename = f"{key}.pt"
            torch.save(embedding.cpu(), output_dir / filename)
            seen_prompts[key] = {
                "file": filename,
                "task": item["task"],
                "variant": variant["variant"],
                "instruction": item["instruction"],
                "prompt": prompt,
                "source": item["source"],
                "shape": list(embedding.shape),
            }
            print(f"cached {item['task']} [{variant['variant']}] -> {filename} shape={tuple(embedding.shape)}", flush=True)

    manifest["entries"] = seen_prompts
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    unique_tasks = len({item["task"] for item in task_items})
    print(
        f"wrote {len(seen_prompts)} embeddings for {len(task_items)} unique instructions "
        f"across {unique_tasks} tasks to {output_dir}"
    )


if __name__ == "__main__":
    main()
