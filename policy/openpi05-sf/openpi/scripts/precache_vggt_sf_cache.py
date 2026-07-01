"""Precompute chunked VGGT feature cache for Pi05SF JAX alignment training."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import platform
import time
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import tyro

from openpi.models_pytorch import sf_offline_cache
import openpi.training.align_utils as _align_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def _replace_config(config: Any, **updates):
    if dataclasses.is_dataclass(config):
        return dataclasses.replace(config, **updates)
    values = vars(config).copy()
    values.update(updates)
    return SimpleNamespace(**values)


def prepare_precache_config(config: _config.TrainConfig) -> _config.TrainConfig:
    cache_dir = os.environ.get("SF_CACHE_DIR", getattr(config, "sf_cache_dir", None))
    if not cache_dir:
        raise ValueError("VGGT SF precache requires sf_cache_dir or SF_CACHE_DIR.")
    cache_mode = os.environ.get("SF_CACHE_MODE", "readwrite")
    if cache_mode != "readwrite":
        raise ValueError(f"VGGT SF precache requires SF_CACHE_MODE=readwrite, got {cache_mode!r}.")

    return _replace_config(
        config,
        sf_cache_enable=True,
        sf_cache_mode=cache_mode,
        sf_cache_miss_policy="online_compute",
        sf_cache_dir=cache_dir,
        sf_cache_save_dtype=os.environ.get("SF_CACHE_SAVE_DTYPE", getattr(config, "sf_cache_save_dtype", "bf16")),
        sf_cache_chunk_size=env_int("SF_CACHE_CHUNK_SIZE", int(getattr(config, "sf_cache_chunk_size", 128))),
        sf_cache_strict_shape=env_bool(
            "SF_CACHE_STRICT_SHAPE",
            bool(getattr(config, "sf_cache_strict_shape", True)),
        ),
        sf_cache_overwrite=env_bool("PRECACHE_OVERWRITE", bool(getattr(config, "sf_cache_overwrite", False))),
        sf_dataset_uid=env_int("SF_DATASET_UID", int(getattr(config, "sf_dataset_uid", 0))),
    )


def _precache_config_for_cli(config: _config.TrainConfig) -> _config.TrainConfig:
    return _replace_config(config, exp_name="precache")


def _cli() -> _config.TrainConfig:
    configs = {
        name: (name, _precache_config_for_cli(config))
        for name, config in _config._CONFIGS_DICT.items()  # noqa: SLF001
    }
    return tyro.extras.overridable_config_cli(configs)


def setup_distributed() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def resolve_precache_devices() -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT precache requires CUDA because TorchAlignFeatureExtractor loads VGGT on cuda devices.")

    visible_count = torch.cuda.device_count()
    explicit = os.environ.get("PRECACHE_VGGT_DEVICES")
    if explicit:
        devices = [int(part) for part in explicit.split(",") if part.strip()]
        if not devices:
            raise ValueError("PRECACHE_VGGT_DEVICES is set but no device id was parsed.")
        for device in devices:
            if device < 0 or device >= visible_count:
                raise ValueError(f"Invalid CUDA device id {device}; visible CUDA device count is {visible_count}.")
        return devices

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        return [local_rank % visible_count]

    count = env_int("PRECACHE_VGGT_DEVICE_COUNT", 1)
    if count < 1:
        raise ValueError(f"PRECACHE_VGGT_DEVICE_COUNT must be >= 1, got {count}.")
    return list(range(min(count, visible_count)))


def _unpack_precache_batch(batch):
    if len(batch) != 3:
        raise ValueError("VGGT SF precache requires a data loader batch with (observation, actions, sf_identity).")
    observation, _actions, sf_identity = batch
    return observation, sf_identity


def _infer_num_batches(loader) -> int:
    torch_loader = getattr(getattr(loader, "_data_loader", None), "torch_loader", None)
    if torch_loader is None:
        raise ValueError("Unable to infer number of batches; set PRECACHE_NUM_BATCHES.")
    return len(torch_loader)


def _reference_tokens(observation) -> int:
    return int(sum((image.shape[1] // 14) * (image.shape[2] // 14) for image in observation.images.values()))


def _cache_key(identity: dict[str, np.ndarray], index: int) -> sf_offline_cache.SFCacheKey:
    return sf_offline_cache.make_cache_key(
        identity["dataset_uid"][index],
        identity["episode_index"][index],
        identity["step_index"][index],
    )


def _is_cached(config, key: sf_offline_cache.SFCacheKey, expected_shape: tuple[int, int]) -> bool:
    tensor = sf_offline_cache.load_cached_tensor(
        config.sf_cache_dir,
        key,
        config.sf_cache_save_dtype,
        int(config.sf_cache_chunk_size),
        expected_shape=expected_shape,
        strict_shape=bool(getattr(config, "sf_cache_strict_shape", True)),
    )
    return tensor is not None


def precache_batch(
    *,
    extractor,
    observation,
    sf_identity,
    config,
    reference_tokens: int,
    overwrite: bool,
) -> dict[str, int]:
    batch_size = next(iter(observation.images.values())).shape[0]
    identity = _align_utils._normalize_sf_identity(config, observation, sf_identity, batch_size)  # noqa: SLF001
    expected_shape = (int(reference_tokens), 2 * int(config.vggt_dim))

    missing_indices = []
    existing = 0
    for index in range(batch_size):
        key = _cache_key(identity, index)
        if not overwrite and _is_cached(config, key, expected_shape):
            existing += 1
        else:
            missing_indices.append(index)

    if not missing_indices:
        return {"existing": existing, "written": 0, "failed": 0}

    features = extractor(
        _align_utils.prepare_align_observation(observation),
        reference_tokens=reference_tokens,
        sf_identity=sf_identity,
    )
    written = 0
    failed = 0
    for index in missing_indices:
        key = _cache_key(identity, index)
        try:
            ok = sf_offline_cache.save_cached_tensor(
                config.sf_cache_dir,
                key,
                torch.as_tensor(features.targets[index]),
                cache_dtype=config.sf_cache_save_dtype,
                chunk_size=int(config.sf_cache_chunk_size),
                overwrite=overwrite,
            )
            written += int(ok)
            existing += int(not ok)
        except Exception:
            failed += 1
            logging.exception(
                "Failed to write VGGT SF cache for dataset_uid=%s episode_index=%s step_index=%s",
                key.dataset_uid,
                key.episode_index,
                key.step_index,
            )

    return {"existing": existing, "written": written, "failed": failed}


def precache_loop(config: _config.TrainConfig) -> dict[str, int | float | str]:
    config = prepare_precache_config(config)
    pathlib.Path(config.sf_cache_dir).mkdir(parents=True, exist_ok=True)
    rank, world_size = setup_distributed()

    num_batches_env = env_int("PRECACHE_NUM_BATCHES", 0)
    loader = _data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=num_batches_env if num_batches_env > 0 else None,
    )
    num_batches = num_batches_env if num_batches_env > 0 else _infer_num_batches(loader)

    devices = resolve_precache_devices()
    extractor = _align_utils.TorchAlignFeatureExtractor(config, devices=devices)
    overwrite = bool(getattr(config, "sf_cache_overwrite", False))

    totals = {"existing": 0, "written": 0, "failed": 0}
    start_time = time.time()
    logging.info(
        "Starting VGGT SF cache precache on %s: rank=%d/%d batches=%d cache_dir=%s dtype=%s chunk_size=%d devices=%s overwrite=%s",
        platform.node(),
        rank,
        world_size,
        num_batches,
        config.sf_cache_dir,
        config.sf_cache_save_dtype,
        int(config.sf_cache_chunk_size),
        devices,
        overwrite,
    )

    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(tqdm.tqdm(loader, total=num_batches, desc="Precache VGGT SF")):
                if batch_index >= num_batches:
                    break
                observation, sf_identity = _unpack_precache_batch(batch)
                stats = precache_batch(
                    extractor=extractor,
                    observation=observation,
                    sf_identity=sf_identity,
                    config=config,
                    reference_tokens=_reference_tokens(observation),
                    overwrite=overwrite,
                )
                for key in totals:
                    totals[key] += stats[key]
    finally:
        extractor.close()

    summary = {
        "cache_dir": config.sf_cache_dir,
        "sf_cache_save_dtype": config.sf_cache_save_dtype,
        "sf_cache_chunk_size": int(config.sf_cache_chunk_size),
        "rank": int(rank),
        "world_size": int(world_size),
        "num_batches": int(num_batches),
        "existing": int(totals["existing"]),
        "written": int(totals["written"]),
        "failed": int(totals["failed"]),
        "duration_sec": round(time.time() - start_time, 3),
    }
    summary_name = "_precache_summary.json" if world_size == 1 else f"_precache_summary.rank{rank}.json"
    summary_path = pathlib.Path(config.sf_cache_dir) / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("VGGT SF cache precache summary: %s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    init_logging()
    config = _cli()
    try:
        precache_loop(config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
