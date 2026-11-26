"""Validate local four-suite data and populate the original T5 prompt cache."""

import json
from pathlib import Path


def check_data(cfg):
    expected = {
        "libero_spatial_no_noops_lerobot": 53229,
        "libero_object_no_noops_lerobot": 67309,
        "libero_goal_no_noops_lerobot": 52895,
        "libero_10_no_noops_lerobot": 104280,
    }
    rows = []
    for path in cfg.data.train.dataset_dirs:
        root = Path(path)
        meta = json.loads((root / "meta/info.json").read_text())
        if root.name not in expected or int(meta["total_frames"]) != expected[root.name]:
            raise ValueError(f"Dataset frame inventory differs from paper: {root}")
        if not list(root.glob("data/chunk-*/*.parquet")):
            raise ValueError(f"Missing LeRobot parquet shards: {root}")
        if not list(root.glob("videos/chunk-*/*/*.mp4")):
            raise ValueError(f"Missing camera video shards: {root}")
        for key in ("meta/tasks.jsonl", "meta/episodes.jsonl"):
            with (root / key).open() as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
        rows.append(
            dict(suite=root.name, frames=meta["total_frames"], episodes=meta["total_episodes"])
        )
    if len({r["suite"] for r in rows}) != 4 or sum(r["episodes"] for r in rows) != 1712:
        raise ValueError("Expected exactly four suites and 1712 episodes")
    from hydra.utils import instantiate
    from recoverwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    processor = instantiate(cfg.data.train.processor)
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(cfg.data.train.pretrained_norm_stats)
    )
    return dict(
        suites=rows, frames=sum(r["frames"] for r in rows), episodes=1712, normalization="min/max"
    )


def text_cache(cfg, device="cuda", batch_size=16):
    import hashlib  # Prompt-to-cache addressing, never a file integrity hash.
    import torch
    from recoverwam.models.wan21_14b.helpers.io import load_state_dict
    from recoverwam.models.wan21_14b.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    from recoverwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    check_data(cfg)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prompts = sorted(
        {
            DEFAULT_PROMPT.format(task=json.loads(line)["task"])
            for directory in cfg.data.train.dataset_dirs
            for line in (Path(directory) / "meta/tasks.jsonl").read_text().splitlines()
            if line.strip()
        }
    )
    root = Path(cfg.data.train.text_embedding_cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    wan = Path(cfg.model.model_id)
    encoder = WanTextEncoder()
    encoder.load_state_dict(
        load_state_dict(str(wan / "models_t5_umt5-xxl-enc-bf16.pth"), torch_dtype=torch.bfloat16),
        strict=True,
    )
    encoder = encoder.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    tokenizer = HuggingfaceTokenizer(
        name=str(wan / "google/umt5-xxl"), seq_len=128, clean="whitespace"
    )
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device, dtype=torch.bool)
            context = encoder(ids, mask)
            for i, prompt in enumerate(batch):
                path = (
                    root
                    / f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len128.{cfg.data.train.text_embedding_cache_tag}.pt"
                )
                temp = path.with_suffix(".tmp")
                torch.save(
                    dict(context=context[i].cpu().contiguous(), mask=mask[i].cpu().contiguous()),
                    temp,
                )
                temp.replace(path)
    return dict(prompts=len(prompts), context_length=128, output=str(root))
