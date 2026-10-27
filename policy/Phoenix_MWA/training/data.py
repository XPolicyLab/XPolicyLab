"""Validate the native RoboTwin v2.1 export and prepare its indexed T5 cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ..runtime.loader import activate_source, POLICY_DIR


def tasks(root):
    rows = [
        json.loads(s)
        for s in (Path(root) / "meta/tasks.jsonl").read_text().splitlines()
        if s.strip()
    ]
    rows = sorted(rows, key=lambda x: int(x["task_index"]))
    if [int(r["task_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Task indices must be contiguous from zero")
    return rows


def validate_dataset(root):
    root = Path(root)
    activate_source()
    from wam.datasets.robotwin_joint96_robot37 import clean_metadata

    info, episodes = clean_metadata(root)
    for k in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        if k not in info["features"]:
            raise ValueError("Missing native RoboTwin camera key: " + k)
    if not (root / "data").is_dir() or not (root / "videos").is_dir():
        raise FileNotFoundError("Expected data/ and videos/ in the LeRobot export")
    rows = tasks(root)
    return dict(
        total_episodes=info["total_episodes"],
        selected_clean_episodes=len(episodes),
        tasks=len(rows),
        fps=info["fps"],
    )


def validate_cache(root, cache):
    activate_source()
    from wam.datasets.text_embedding_cache import TaskIndexShardedTextEmbeddingCache
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    bank = TaskIndexShardedTextEmbeddingCache(cache, expected_context_len=128)
    rows = tasks(root)
    if (
        bank.total_tasks != len(rows)
        or bank.embedding_dim != 4096
        or bank.manifest.get("prompt_template") != DEFAULT_PROMPT
    ):
        raise ValueError("Incompatible task-indexed T5 cache")
    # Verify every indexed prompt without reading hundreds of GiB of embedding payload.
    for start in range(0, len(rows), bank.shard_size):
        shard = bank._open_shard(start // bank.shard_size)
        expected = np.array(
            [
                hashlib.sha256(
                    DEFAULT_PROMPT.format(task=r["task"]).encode()
                ).hexdigest()
                for r in rows[start : shard.end]
            ],
            dtype="S64",
        )
        if not np.array_equal(np.asarray(shard.prompt_hashes).astype("S64"), expected):
            raise ValueError("Cache prompt index differs from dataset")
        lengths = np.asarray(shard.lengths)
        offsets = np.asarray(shard.offsets)
        if (
            np.any(lengths < 1)
            or np.any(lengths > 128)
            or not np.array_equal(np.diff(offsets), lengths)
        ):
            raise ValueError("Invalid cache token index")
        bank._open_shards.clear()
    return bank.manifest


def write_shard(cache, start, prompts, contexts, masks):
    """Serialize exact BF16 valid tokens in the upstream reader's layout."""
    import torch

    end = start + len(prompts)
    folder = Path(cache) / "shards" / f"shard_{start:07d}_{end:07d}"
    folder.mkdir(parents=True)
    lengths = np.array([int(m.sum()) for m in masks], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    hashes = np.array(
        [hashlib.sha256(p.encode()).hexdigest() for p in prompts], dtype="S64"
    )
    with (folder / "embeddings.uint16.mmap").open("wb") as f:
        for c, m in zip(contexts, masks):
            n = int(m.sum())
            if n < 1 or n > 128 or not bool(m[:n].all()) or bool(m[n:].any()):
                raise ValueError("Expected prefix-valid T5 padding")
            c[:n].to(dtype=torch.bfloat16, device="cpu").contiguous().view(
                torch.uint16
            ).numpy().tofile(f)
    for name, a in [
        ("lengths", lengths),
        ("offsets", offsets),
        ("prompt_sha256", hashes),
    ]:
        np.save(folder / (name + ".npy"), a)
    (folder / "manifest.json").write_text(json.dumps(dict(start=start, end=end)))


def build_cache(root, cache, weights, device):
    import torch

    activate_source()
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from wam.model.backbones.wan22.helpers.loader import (
        _resolve_configs,
        _load_registered_model,
    )
    from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    cache = Path(cache)
    if cache.exists():
        raise FileExistsError(
            "Use a new cache directory; partial/existing caches are never overwritten"
        )
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(weights).resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    _, text, _, tok = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    text.download_if_necessary()
    tok.download_if_necessary()
    encoder = _load_registered_model(
        text.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tok.path, seq_len=128, clean="whitespace")
    rows = tasks(root)
    shard_size = 64
    for start in range(0, len(rows), shard_size):
        prompts = [
            DEFAULT_PROMPT.format(task=r["task"])
            for r in rows[start : start + shard_size]
        ]
        contexts = []
        masks = []
        with torch.no_grad():
            for prompt in prompts:
                ids, mask = tokenizer(
                    [prompt], return_mask=True, add_special_tokens=True
                )
                mask = mask.to(device=device, dtype=torch.bool)
                c = encoder(ids.to(device), mask)
                contexts.append(c[0].detach().cpu())
                masks.append(mask[0].cpu())
        write_shard(cache, start, prompts, contexts, masks)
        print(f"Text cache: {min(start+shard_size,len(rows))}/{len(rows)}", flush=True)
    metadata = dict(
        format="task_index_sharded_mmap_v1",
        total_tasks=len(rows),
        context_len=128,
        embedding_dim=4096,
        shard_size=shard_size,
        logical_dtype="bfloat16",
        storage_dtype="uint16",
        prompt_template=DEFAULT_PROMPT,
        tasks_file_sha256=hashlib.sha256(
            (Path(root) / "meta/tasks.jsonl").read_bytes()
        ).hexdigest(),
    )
    (cache / "manifest.json").write_text(json.dumps(metadata, indent=2))
    validate_cache(root, cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bench_name", "ckpt_name", "env_cfg_type", "action_type"):
        p.add_argument(key)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--build-text-cache", action="store_true")
    p.add_argument("--weights", type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if (args.bench_name, args.env_cfg_type, args.action_type) != (
        "RoboTwin",
        "arx_x5",
        "joint",
    ):
        raise ValueError("Expected RoboTwin arx_x5 joint")
    result = validate_dataset(args.data)
    if args.build_text_cache:
        if args.weights is None:
            p.error("--build-text-cache requires --weights")
        build_cache(args.data, args.text_cache, args.weights, args.device)
    validate_cache(args.data, args.text_cache)
    print(json.dumps(result, indent=2))
    print("Native dataset and indexed text cache verified")


if __name__ == "__main__":
    main()
