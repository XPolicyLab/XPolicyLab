#!/usr/bin/env python3
"""
Add episode-level T5 embedding cache to an existing LeRobot dataset.

What it does
------------
- For each episode in `meta/episodes.jsonl`, ensure it has a `t5_embedding_path` field
  pointing to a cached embedding file under `{dataset_root}/{t5_folder_name}/episode_XXXXXX.pt`.
- If the cache is missing, encode the episode instruction using WAN's `T5EncoderModel`,
  write the `.pt` file, and update `meta/episodes.jsonl` in-place (atomically).

Why
---
Motus' `LeRobotMotusDataset` can load T5 embeddings from:
1) frame-level `language_embedding` (parquet), or
2) episode-level external pt referenced by `meta/episodes.jsonl:t5_embedding_path`, or
3) on-the-fly fallback (optional).

This script upgrades an existing LeRobot dataset to (2), so training can run without
on-the-fly encoding.

An example of using this script:

export WAN_PATH=/share/home/bhz/pretrained_models

python /share/home/lht/Motus/data/lerobot/add_t5_cache_to_lerobot_dataset.py \
  --repo_id randomized_only50/beat_block_hammer \
  --root /share/home/lht/.cache/huggingface/lerobot/robotwin/randomized_only50/beat_block_hammer \
  --device cuda \
  --t5_folder_name t5_embedding
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def _load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonlines_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _resolve_dataset_root(repo_id: str, root: Optional[str]) -> Path:
    # Use LeRobot metadata to resolve default cache dir if root is not provided.
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return Path(meta.root)


def _init_wan_t5_encoder(
    wan_path: str,
    device: str,
    text_len: int = 512,
) -> Any:
    """
    Initialize WAN T5EncoderModel.
    We mirror the initialization used in Motus inference scripts.
    """
    try:
        from Motus.bak.wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception:
        # Fallback: add bak path similarly to inference scripts
        bak_root = str((Path(__file__).resolve().parents[2] / "bak").resolve())
        if bak_root not in sys.path:
            sys.path.insert(0, bak_root)
        from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt = os.path.join(wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"T5 checkpoint not found: {ckpt}")
    if not os.path.exists(tok):
        raise FileNotFoundError(f"T5 tokenizer dir not found: {tok}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def _encode_t5(encoder: Any, instruction: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        out = encoder([instruction], device)
    if isinstance(out, list):
        emb = out[0]
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        raise ValueError(f"Unexpected T5 encoder output type: {type(out)}")

    # Normalize to [S, D] tensor on CPU
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()


def _episode_instruction_from_meta(ep_row: Dict[str, Any]) -> str:
    # episodes.jsonl stores "tasks": [<task string>, ...]
    tasks = ep_row.get("tasks", None)
    if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], str):
        return tasks[0]
    # Fallback: try "task"
    task = ep_row.get("task", "")
    if isinstance(task, str):
        return task
    return str(task)


def main():
    parser = argparse.ArgumentParser(
        description="Add WAN T5 embedding cache to a LeRobot dataset (episode-level pt + episodes.jsonl pointer)"
    )
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo_id (identifier)")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root (contains meta/data/videos). If omitted, LeRobot default cache is used.",
    )
    parser.add_argument(
        "--wan_path",
        type=str,
        default=None,
        help="Base path that contains Wan2.2-TI2V-5B/ (can also use env WAN_PATH/WAN_ROOT).",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--text_len", type=int, default=512, help="T5 text_len (default: 512)")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding", help="Cache folder name under dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files and meta pointers")
    parser.add_argument("--max_episodes", type=int, default=0, help="Process at most N episodes (0 = all)")
    parser.add_argument(
        "--strip_parquet_metadata",
        action="store_true",
        help="Also strip parquet schema metadata (fix datasets>=3.x 'List' feature issues).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    wan_path = args.wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
    if not wan_path:
        raise ValueError("WAN path not provided. Use --wan_path or set WAN_PATH/WAN_ROOT.")

    dataset_root = _resolve_dataset_root(args.repo_id, args.root)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    out_dir = dataset_root / args.t5_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_jsonlines(episodes_path)
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    encoder = None
    updated = 0
    skipped = 0

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        rel = f"{args.t5_folder_name}/episode_{ep_idx:06d}.pt"
        abs_pt = dataset_root / rel

        has_ptr = ("t5_embedding_path" in ep) and isinstance(ep.get("t5_embedding_path"), str)
        ptr_ok = has_ptr and (Path(dataset_root) / str(ep["t5_embedding_path"])).exists()

        if not args.overwrite:
            if ptr_ok or abs_pt.exists():
                # Ensure pointer exists if file exists
                if not has_ptr and abs_pt.exists():
                    ep["t5_embedding_path"] = rel
                    updated += 1
                else:
                    skipped += 1
                continue

        instr = _episode_instruction_from_meta(ep)
        if encoder is None:
            print(f"Loading WAN T5 encoder from {wan_path} on {device} ...")
            encoder = _init_wan_t5_encoder(wan_path=wan_path, device=device, text_len=int(args.text_len))

        emb = _encode_t5(encoder, instr, device=device)
        torch.save(emb, abs_pt)
        ep["t5_embedding_path"] = rel
        updated += 1

        if updated % 10 == 0:
            print(f"Processed {updated} episodes (latest: {ep_idx})")

    # Write back episodes.jsonl atomically
    # Note: we rewrote only the (potentially truncated) prefix if max_episodes is set.
    # To preserve remaining episodes, reload full file and patch by episode_index.
    if args.max_episodes and args.max_episodes > 0:
        full = _load_jsonlines(episodes_path)
        patch_map = {int(ep["episode_index"]): ep for ep in episodes}
        for i, ep in enumerate(full):
            ep_idx = int(ep["episode_index"])
            if ep_idx in patch_map:
                full[i] = patch_map[ep_idx]
        _write_jsonlines_atomic(episodes_path, full)
    else:
        _write_jsonlines_atomic(episodes_path, episodes)

    print(f"Done. updated={updated}, skipped={skipped}.")
    print(f"Dataset root: {dataset_root}")

    if args.strip_parquet_metadata:
        from Motus.data.lerobot.strip_parquet_hf_metadata import main as strip_main  # type: ignore

        # emulate calling the stripping utility
        sys.argv = [
            sys.argv[0],
            "--dataset_root",
            str(dataset_root),
        ]
        strip_main()


if __name__ == "__main__":
    main()


