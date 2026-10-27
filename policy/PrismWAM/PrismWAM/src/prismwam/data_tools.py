"""Explicit, non-overwriting preparation of portable PrismWAM data assets.

These tools read raw training data. They never fit statistics on validation data
implicitly, mutate source indexes, download weights, or create visual caches.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _destination(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(value: Any, path: str | Path) -> Path:
    path = _destination(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    return path


def save_tensor_asset(value: dict, path: str | Path) -> Path:
    path = _destination(path)
    # Exclusive creation also prevents a concurrent worker from replacing it.
    with path.open("xb") as handle:
        torch.save(value, handle)
    return path


def episode_paths(index: dict, root: str | Path | None = None) -> list[Path]:
    root = Path(root or index["root_dir"]).expanduser().resolve()
    paths = []
    for row in index["episodes"]:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe episode path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if (
            row.get("size_bytes") is not None
            and path.stat().st_size != row["size_bytes"]
        ):
            raise ValueError(f"Episode size differs from manifest: {path}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate episode paths")
    return paths


def relocate_index(source: str | Path, root: str | Path, output: str | Path) -> dict:
    """Change only the absolute dataset root; preserve partition identity."""
    payload = json.loads(Path(source).read_text())
    episode_paths(payload, root)
    payload["root_dir"] = str(Path(root).expanduser().resolve())
    write_json(payload, output)
    return payload


def build_robodojo_index(
    root: str | Path,
    output: str | Path,
    cameras=("cam_head", "cam_left_wrist", "cam_right_wrist"),
    embodiment: str = "arx_x5",
) -> dict:
    """Scan frame-aligned native HDF5 metadata without decoding RGB/depth."""
    from .datasets.robodojo_rgb_depth_action_dataset import (
        ROBODOJO_INDEX_VERSION,
        RoboDojoRGBDepthActionVideoDataset,
        _natural_key,
    )

    root = Path(root).expanduser().resolve()
    reader = RoboDojoRGBDepthActionVideoDataset.__new__(
        RoboDojoRGBDepthActionVideoDataset
    )
    reader.cameras = tuple(cameras)
    rows = []
    identity = hashlib.sha256()
    for task in sorted(root.iterdir(), key=_natural_key):
        if not task.is_dir() or task.name.startswith("."):
            continue
        for path in sorted(
            (task / embodiment / "data").glob("episode_*.hdf5"), key=_natural_key
        ):
            metadata = reader._read_episode_metadata(path)
            stat = path.stat()
            row = dict(
                relative_path=path.relative_to(root).as_posix(),
                task=task.name,
                embodiment=embodiment,
                episode_index=reader._episode_index(path),
                length=metadata["length"],
                instruction=metadata["instruction"],
                frequency=metadata["frequency"],
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
            identity.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            )
            rows.append(row)
    if not rows:
        raise ValueError("No RoboDojo episodes found")
    payload = dict(
        version=ROBODOJO_INDEX_VERSION,
        root_dir=str(root),
        episodes=rows,
        num_episodes=len(rows),
        num_frames=sum(r["length"] for r in rows),
        manifest_fingerprint=identity.hexdigest(),
    )
    write_json(payload, output)
    return payload


def build_holdout(
    index_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    expected_tasks: int = 35,
) -> dict:
    """Per-task 90/10 partition used by the released action normalization."""
    from .robodojo_split import split_index

    source = json.loads(Path(index_path).read_text())
    episode_paths(source)
    partitions = split_index(source, seed=seed, expected_tasks=expected_tasks)
    destinations = {role: Path(output_dir) / f"{role}.json" for role in partitions}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(path)
    for role, partition in partitions.items():
        write_json(partition, destinations[role])
    return {role: str(path) for role, path in destinations.items()}


def joint14_statistics(
    index_path: str | Path, output: str | Path, std_floor: float = 1e-6
) -> dict:
    """Fit native joint14 statistics on an explicit episode index.

    The RoboDojo recipes train on every episode but normalize with statistics
    fitted on ``holdout/train.json``, the 90% partition that excludes the
    validation episodes.
    """
    from .datasets.robodojo_rgb_depth_action_dataset import (
        RoboDojoRGBDepthActionVideoDataset,
    )

    index = json.loads(Path(index_path).read_text())
    paths = episode_paths(index)
    total = np.zeros(14, dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = RoboDojoRGBDepthActionVideoDataset._joint_vector(
                handle, "action"
            ).astype(np.float64)
        if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
            raise ValueError(f"Invalid native joint14 vectors: {path}")
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += len(values)
    if count == 0:
        raise ValueError("Cannot fit empty training data")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(square / count - mean**2, 0)), std_floor)
    payload = dict(
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
        action_key="action/joint_vector",
        action_dim=14,
        num_episodes=len(paths),
        num_frames=count,
        train_manifest_fingerprint=index["manifest_fingerprint"],
        normalization="zscore",
        normalization_source="explicit_native_training_partition",
    )
    save_tensor_asset(payload, output)
    return payload


def _config_node(config: str | Path, node: str, overrides=()) -> dict:
    from .config import load_config

    value = OmegaConf.select(load_config(config, overrides), node)
    if value is None:
        raise KeyError(f"Configuration has no {node}")
    return OmegaConf.to_container(value, resolve=True)


def dataset_statistics(
    config: str | Path, output: str | Path, node: str = "data.train", overrides=()
) -> dict:
    """Use the selected dataset's own vector conversion and statistics method.

    For mixed real-robot data select one source, e.g. data.train.sources.0.
    A validation node is rejected to prevent accidental holdout fitting.
    """
    if node == "data.val" or node.startswith("data.val."):
        raise ValueError("Fit training statistics only; reuse them for validation")
    spec = copy.deepcopy(_config_node(config, node, overrides))
    target = spec.get("_target_", "")
    if "sources" in spec:
        raise ValueError("Mixed data requires one --node data.train.sources.N per task")
    if not target and "task_names" in spec:
        target = "prismwam.datasets.real_world_mixed_dataset.RealWorldTaskDataset"
        spec["_target_"] = target
    _destination(output)
    if "normalize_actions" in spec:
        spec["normalize_actions"] = False
    if "action_stats_cache_path" in spec:
        spec["action_stats_cache_path"] = None
        spec["action_stats_cache_policy"] = (
            "compute" if "real_world" in target else "strict"
        )
    if "pretrained_norm_stats" in spec:
        spec["pretrained_norm_stats"] = None
    dataset = instantiate(OmegaConf.create(spec))
    if hasattr(dataset, "save_dataset_stats"):
        dataset.save_dataset_stats(output)
        return {"path": str(output), "format": "dataset_stats_json"}
    mean, std = dataset._compute_action_stats()
    signature_fn = (
        getattr(dataset, "action_stats_signature", None)
        or dataset._action_stats_signature
    )
    payload = dict(
        mean=mean.cpu(),
        std=std.cpu(),
        signature=signature_fn(),
        action_dim=dataset.action_dim,
        num_episodes=len(dataset.episodes),
    )
    if hasattr(dataset, "action_key"):
        payload["action_key"] = dataset.action_key
    if hasattr(dataset, "task_names") and dataset.task_names is not None:
        payload["tasks"] = list(dataset.task_names)
    if spec.get("episode_index_path"):
        index = json.loads(Path(spec["episode_index_path"]).read_text())
        payload["train_manifest_fingerprint"] = index["manifest_fingerprint"]
    save_tensor_asset(payload, output)
    return payload


def _prompts(dataset) -> list[str]:
    if callable(getattr(dataset, "prompts", None)):
        return list(dataset.prompts())
    prompts = set()
    for episode in dataset.episodes:
        candidates = episode.get("instruction_candidates")
        if candidates:
            prompts.update(
                dataset.prompt_template.format(task=value) for value in candidates
            )
        elif hasattr(dataset, "_prompt_for_episode"):
            prompts.add(dataset._prompt_for_episode(episode, start=0))
        elif episode.get("instruction"):
            from .datasets.robotwin_rgb_dataset import PROMPT_TEMPLATE

            prompts.add(PROMPT_TEMPLATE.format(task=episode["instruction"]))
        else:
            raise ValueError("Dataset does not expose its exact language prompts")
    return sorted(prompts)


def text_embeddings(
    config: str | Path,
    *,
    checkpoint: str | Path,
    tokenizer: str | Path,
    device: str = "cuda",
    batch_size: int = 8,
    overrides=(),
) -> dict:
    """Generate the full-length context+mask format required by recipe readers."""
    from .models.wan22.helpers.loader import _load_registered_model
    from .models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not Path(checkpoint).is_file() or not Path(tokenizer).is_dir():
        raise FileNotFoundError(
            "Provide local text-encoder weights and tokenizer directory"
        )
    cache_prompts = {}
    for split in ("train", "val"):
        try:
            spec = _config_node(config, f"data.{split}", overrides)
        except KeyError:
            continue
        sources = spec.get("sources")
        nodes = sources if sources is not None else [spec]
        for source in nodes:
            cache = source.get("text_embedding_cache_dir")
            if cache is None:
                continue
            source = copy.deepcopy(source)
            source["text_embedding_cache_dir"] = None
            if sources is not None:
                source["_target_"] = (
                    "prismwam.datasets.real_world_mixed_dataset.RealWorldTaskDataset"
                )
            dataset = instantiate(OmegaConf.create(source))
            key = (
                str(cache),
                int(source.get("context_len", 128)),
                source.get("text_encoder_id", "wan22ti2v5b"),
            )
            cache_prompts.setdefault(key, set()).update(_prompts(dataset))
    if not cache_prompts:
        raise ValueError("No text-cache consumers found in config")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    encoder = _load_registered_model(
        str(checkpoint), "wan_video_text_encoder", torch_dtype=dtype, device=device
    ).eval()
    written = 0
    skipped = 0
    with torch.no_grad():
        for (directory, length, encoder_id), prompts in cache_prompts.items():
            tok = HuggingfaceTokenizer(
                name=str(tokenizer), seq_len=length, clean="whitespace"
            )
            pending = []
            for prompt in sorted(prompts):
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                path = Path(directory) / f"{digest}.t5_len{length}.{encoder_id}.pt"
                if path.exists():
                    skipped += 1
                else:
                    pending.append((prompt, path))
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                ids, mask = tok(
                    [row[0] for row in batch], return_mask=True, add_special_tokens=True
                )
                mask = mask.to(device=device, dtype=torch.bool)
                context = encoder(ids.to(device), mask)
                for i, (_, path) in enumerate(batch):
                    save_tensor_asset(
                        {
                            "context": context[i]
                            .detach()
                            .cpu()
                            .to(torch.bfloat16)
                            .contiguous(),
                            "mask": mask[i].detach().cpu().contiguous(),
                        },
                        path,
                    )
                    written += 1
    return dict(
        written=written, skipped_existing=skipped, cache_consumers=len(cache_prompts)
    )
