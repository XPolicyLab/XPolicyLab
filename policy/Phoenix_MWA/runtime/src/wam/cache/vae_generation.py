"""Offline VAE and frozen-VLM artifact generation."""

from __future__ import annotations

from bisect import bisect_left
import datetime
import json
import logging
import os
import socket
import time
import uuid
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from wam.model.backbones.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)

from .contracts import ArtifactContract
from .errors import CacheError
from .fields import cache_key_for_sample, vlm_artifact_id
from .fingerprint import vae_fingerprint_from_path, vlm_fingerprint_from_path
from .vlm_latent import (
    VlmLatentSource,
    VlmLatentGenerationDataset,
    encode_vlm_artifacts,
)
from .manager import CacheManager
from .tensor_shard import atomic_json_dump
from .vae_latent import (
    VaeLatentGenerationDataset,
    VaeLatentSource,
    artifact_ids,
    encode_vae_artifacts,
)


logger = logging.getLogger(__name__)


# CacheManager-owned defaults for offline VAE generation.  Keep resource and
# scheduling knobs here so every entrypoint gets the same behavior without
# wrapper-specific environment variables.  Explicit ``precompute.*`` config
# always takes precedence.
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 12
DEFAULT_DATALOADER_TIMEOUT = 3600
DEFAULT_FRAME_CACHE_CAPACITY = 96
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_IN_ORDER = False


class _UnifiedGenerationDataset(Dataset):
    """Merge producer inputs while keeping each domain adapter independent."""

    def __init__(self, vae_source: VaeLatentSource, vlm_source: VlmLatentSource):
        if vae_source.indices != vlm_source.indices:
            raise ValueError("VAE and VLM generation sources must share one keyspace.")
        self.vae_source = vae_source
        self.vlm_source = vlm_source

    def __len__(self) -> int:
        return len(self.vae_source.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.vae_source.indices[int(position)]
        result = self.vae_source.get_sample(index)
        vlm_sample = self.vlm_source.get_sample(index)
        if int(vlm_sample["sample_id"]) != int(result["sample_id"]):
            raise CacheError(
                "VAE/VLM generation adapters resolved different sample ids."
            )
        result.update(
            {name: value for name, value in vlm_sample.items() if name != "sample_id"}
        )
        return result


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "", "none", "null"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}.")


def _distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return False, 0, 1, local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_seconds = max(30, int(os.environ.get("WAM_CACHE_DIST_TIMEOUT", "3600")))
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _broadcast_error(active: bool, rank: int, error: str | None) -> None:
    if not active:
        if error is not None:
            raise CacheError(error)
        return
    payload = [error if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise CacheError(str(payload[0]))


def _parse_dtype(value: Any, *, default: torch.dtype) -> torch.dtype:
    if value in (None, "", "null"):
        return default
    text = str(value).replace("torch.", "").strip().lower()
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return values[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder cache dtype: {value!r}.") from exc


def _load_vae(cfg: Any, *, device: str, dtype: torch.dtype) -> tuple[Any, str]:
    model_cfg = _get(cfg, "model")
    if model_cfg is None:
        raise ValueError("`cfg.model` is required for VAE cache generation.")
    model_id = str(_get(model_cfg, "model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=str(
            _get(model_cfg, "tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B")
        ),
        redirect_common_files=bool(_get(model_cfg, "redirect_common_files", True)),
    )
    vae_config.download_if_necessary()
    if vae_config.path is None:
        raise RuntimeError("Wan VAE config resolved without a local path.")
    logger.info("Loading cache producer VAE only from %s", vae_config.path)
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=device,
    ).eval()
    return vae, str(vae_config.path)


def _load_vlm(
    cfg: Any, *, device: str, dtype: torch.dtype
) -> tuple[Any, str, dict[str, Any]]:
    from wam.model.modules.understanding import QwenVLUnderstandingEncoder

    understanding_cfg = _get(_get(cfg, "model"), "understanding")
    if not bool(_get(understanding_cfg, "enabled", False)):
        raise ValueError(
            "VLM cache generation requires model.understanding.enabled=true."
        )
    if bool(_get(understanding_cfg, "train_vlm", False)):
        raise ValueError("VLM cache generation requires train_vlm=false.")
    model_path = str(_get(understanding_cfg, "vlm_model_path", ""))
    if not model_path:
        raise ValueError("VLM cache generation requires a vlm_model_path.")
    resolved = {
        key: _get(understanding_cfg, key, default)
        for key, default in {
            "enabled": True,
            "vlm_model_path": model_path,
            "trust_remote_code": True,
            "train_vlm": False,
            "vlm_gradient_checkpointing": False,
            "save_vlm_weights": None,
            "vlm_batch_size": 0,
            "precompute_metadata": True,
            "max_pixels": 65536,
            "prompt": None,
            "default_view_names": "",
        }.items()
    }
    encoder = QwenVLUnderstandingEncoder(
        vlm_model_path=model_path,
        device=device,
        dtype=dtype,
        prompt=resolved["prompt"],
        train_vlm=False,
        vlm_gradient_checkpointing=False,
        save_vlm_weights=False,
        trust_remote_code=bool(resolved["trust_remote_code"]),
        vlm_batch_size=int(resolved["vlm_batch_size"] or 0),
        precompute_metadata=bool(resolved["precompute_metadata"]),
        max_pixels=int(resolved["max_pixels"]),
    ).eval()
    return encoder, model_path, resolved


def _configure_rank_log(log_dir: Path, rank: int) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_log = log_dir / f"rank_{rank:05d}.log"
    progress_path = log_dir / f"rank_{rank:05d}_progress.json"
    handler = logging.FileHandler(rank_log, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler._wam_cache_rank_handler = True  # type: ignore[attr-defined]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        getattr(existing, "_wam_cache_rank_handler", False)
        for existing in root_logger.handlers
    ):
        root_logger.addHandler(handler)
    return rank_log, progress_path


def _write_progress(path: Path, **values: Any) -> None:
    atomic_json_dump(
        {
            "schema_version": 1,
            "updated_at": time.time(),
            **values,
        },
        path,
    )


def _timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    total = sum(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "mean": total / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _batch_counter(batch: dict[str, Any], key: str) -> int:
    value = batch.get(key)
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.sum().item())
    if isinstance(value, (list, tuple)):
        return sum(int(item) for item in value)
    return int(value)


def _log_aggregate_progress(log_dir: Path, world_size: int) -> None:
    rows = []
    for rank in range(world_size):
        path = log_dir / f"rank_{rank:05d}_progress.json"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                rows.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return
    logger.info(
        "Cache aggregate progress: ranks=%d/%d encoded=%d/%d batches=%d/%d",
        len(rows),
        world_size,
        sum(int(row.get("encoded_samples", 0)) for row in rows),
        sum(int(row.get("total_samples", 0)) for row in rows),
        sum(int(row.get("batch", 0)) for row in rows),
        sum(int(row.get("total_batches", 0)) for row in rows),
    )


def _episode_block_partitions(
    indices: Sequence[int],
    episode_starts: Sequence[int],
    episode_ends: Sequence[int],
    world_size: int,
) -> tuple[range, ...]:
    """Split ordered sample positions into balanced contiguous episode blocks."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not indices:
        raise ValueError("Episode-block sharding requires at least one sample.")
    if len(episode_starts) != len(episode_ends):
        raise ValueError("Episode start/end metadata lengths do not match.")
    if any(int(left) >= int(right) for left, right in pairwise(indices)):
        raise ValueError("Episode-block sharding requires sorted unique indices.")

    spans: list[tuple[int, int]] = []
    covered_until = 0
    for raw_start, raw_end in zip(episode_starts, episode_ends):
        start, end = int(raw_start), int(raw_end)
        if start >= end:
            raise ValueError(f"Invalid episode range [{start}, {end}).")
        left = bisect_left(indices, start, lo=covered_until)
        right = bisect_left(indices, end, lo=left)
        if left == right:
            continue
        if left != covered_until:
            raise ValueError(
                "Selected samples are not fully covered by episode metadata."
            )
        spans.append((left, right))
        covered_until = right
        if covered_until == len(indices):
            break

    if covered_until != len(indices):
        raise ValueError("Selected samples are not fully covered by episode metadata.")
    if len(spans) < world_size:
        raise ValueError(
            "Episode-block sharding requires at least one selected episode "
            f"per rank; got episodes={len(spans)} world_size={world_size}."
        )

    cuts = [0]
    previous_episode_count = 0
    for boundary_rank in range(1, world_size):
        desired_position = len(indices) * boundary_rank / world_size
        minimum_episode_count = previous_episode_count + 1
        maximum_episode_count = len(spans) - (world_size - boundary_rank)
        episode_count = min(
            range(minimum_episode_count, maximum_episode_count + 1),
            key=lambda count: abs(spans[count - 1][1] - desired_position),
        )
        cuts.append(spans[episode_count - 1][1])
        previous_episode_count = episode_count
    cuts.append(len(indices))
    return tuple(range(cuts[rank], cuts[rank + 1]) for rank in range(world_size))


def _wait_for_rank_fragments(
    repository: Any,
    contracts: dict[str, ArtifactContract],
    *,
    build_id: str,
    world_size: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Wait for rank fragments produced by independent distributed jobs."""

    expected = [
        repository.temporary_artifact_root(artifact_id, build_id)
        / "fragments"
        / f"rank{rank:05d}.json"
        for artifact_id in contracts
        for rank in range(world_size)
    ]
    deadline = time.monotonic() + timeout_seconds
    last_reported_missing = None
    while True:
        missing = [path for path in expected if not path.is_file()]
        if not missing:
            logger.info(
                "All external cache fragments are ready: build=%s files=%d",
                build_id,
                len(expected),
            )
            return
        if timeout_seconds <= 0 or time.monotonic() >= deadline:
            raise CacheError(
                "Timed out waiting for external cache fragments: "
                f"build={build_id} missing={len(missing)}/{len(expected)} "
                f"examples={[str(path) for path in missing[:4]]}"
            )
        if len(missing) != last_reported_missing:
            logger.info(
                "Waiting for external cache fragments: build=%s "
                "ready=%d/%d missing_examples=%s",
                build_id,
                len(expected) - len(missing),
                len(expected),
                [str(path) for path in missing[:2]],
            )
            last_reported_missing = len(missing)
        time.sleep(max(0.1, poll_seconds))


def generate_vae_cache(
    *,
    manager: CacheManager,
    cfg: Any,
    distributed: Any = None,
    progress: Any = None,
) -> Any:
    """Generate selected artifacts and atomically publish the catalog."""

    del distributed, progress  # Reserved observer/context extension points.
    if manager.repository is None:
        raise CacheError("VAE cache generation has no configured root.")
    active, rank, world_size, local_rank = _distributed()
    precompute_cfg = _get(cfg, "precompute")
    job_count = max(1, int(_get(precompute_cfg, "job_count", 1)))
    job_index = int(_get(precompute_cfg, "job_index", 0))
    if not 0 <= job_index < job_count:
        raise ValueError(
            f"precompute.job_index must be in [0, {job_count}), got {job_index}."
        )
    global_world_size = world_size * job_count
    global_rank = job_index * world_size + rank
    batch_size = max(1, int(_get(precompute_cfg, "batch_size", DEFAULT_BATCH_SIZE)))
    num_workers = max(
        0,
        int(_get(precompute_cfg, "num_workers", DEFAULT_NUM_WORKERS)),
    )
    shard_size = max(1, int(_get(precompute_cfg, "shard_size", 256)))
    overwrite = _as_bool(
        _get(precompute_cfg, "overwrite", False),
        name="precompute.overwrite",
    )
    max_samples_value = _get(precompute_cfg, "max_samples")
    max_samples = (
        None
        if max_samples_value in (None, "", "null")
        else max(1, int(max_samples_value))
    )
    timeout = max(
        0,
        int(
            _get(
                precompute_cfg,
                "dataloader_timeout",
                DEFAULT_DATALOADER_TIMEOUT,
            )
        ),
    )
    progress_every = max(
        1,
        int(
            _get(
                precompute_cfg,
                "progress_every",
                os.environ.get("WAM_CACHE_PROGRESS_EVERY", 10),
            )
        ),
    )
    frame_cache_capacity = max(
        0,
        int(
            _get(
                precompute_cfg,
                "frame_cache_capacity",
                DEFAULT_FRAME_CACHE_CAPACITY,
            )
        ),
    )
    prefetch_factor = max(
        1,
        int(
            _get(
                precompute_cfg,
                "prefetch_factor",
                DEFAULT_PREFETCH_FACTOR,
            )
        ),
    )
    in_order = _as_bool(
        _get(
            precompute_cfg,
            "in_order",
            DEFAULT_IN_ORDER,
        ),
        name="precompute.in_order",
    )
    if _as_bool(_get(precompute_cfg, "tiled", False), name="precompute.tiled"):
        raise ValueError("VAE latent cache generation requires tiled=false.")

    precision = str(_get(cfg, "mixed_precision", "bf16")).strip().lower()
    default_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(precision, torch.bfloat16)
    if not torch.cuda.is_available() and default_dtype == torch.float16:
        default_dtype = torch.float32
    dtype = _parse_dtype(_get(precompute_cfg, "dtype"), default=default_dtype)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(str(_get(cfg, "output_dir", manager.repository.root)))
    log_dir = (
        Path(os.environ.get("WAM_CACHE_LOG_DIR", str(output_dir / "cache_logs")))
        .expanduser()
        .resolve()
    )
    rank_log, progress_path = _configure_rank_log(log_dir, global_rank)
    started = time.perf_counter()
    logger.info(
        "Encoder cache generation start: rank=%d/%d local_rank=%d host=%s "
        "job=%d/%d local_process_rank=%d/%d "
        "device=%s root=%s fields=%s batch=%d workers=%d prefetch=%d "
        "in_order=%s frame_cache=%d shard=%d validation_workers=%d log=%s",
        global_rank,
        global_world_size,
        local_rank,
        socket.gethostname(),
        job_index,
        job_count,
        rank,
        world_size,
        device,
        manager.repository.root,
        [field.value for field in manager.config.fields],
        batch_size,
        num_workers,
        prefetch_factor,
        in_order,
        frame_cache_capacity,
        shard_size,
        manager.config.validation_workers,
        rank_log,
    )
    _write_progress(
        progress_path,
        rank=global_rank,
        world_size=global_world_size,
        phase="initializing",
        batch=0,
        total_batches=0,
        encoded_samples=0,
        total_samples=0,
    )

    target_error = None
    if rank == 0:
        try:
            manager.repository.validate_write_target(
                (
                    *artifact_ids(manager.config.fields),
                    *((vlm_artifact_id(),) if manager.config.vlm_enabled else ()),
                ),
                overwrite=overwrite,
            )
        except Exception as exc:
            target_error = f"{type(exc).__name__}: {exc}"
    _broadcast_error(active, rank, target_error)

    dataset_cfg = _get(_get(cfg, "data"), "train")
    if dataset_cfg is None:
        raise ValueError("`data.train` is required for VAE cache generation.")
    dataset_started = time.perf_counter()
    dataset = instantiate(
        dataset_cfg, return_vlm_current_images=manager.config.vlm_enabled
    )
    global_indices = tuple(range(len(dataset)))
    if max_samples is not None:
        global_indices = global_indices[:max_samples]
    if len(global_indices) < global_world_size:
        raise ValueError(
            f"VAE cache samples={len(global_indices)} must be >= "
            f"world_size={global_world_size}."
        )
    vae_source = (
        VaeLatentSource.create(dataset, manager.config.fields, indices=global_indices)
        if manager.config.fields
        else None
    )
    vlm_source = (
        VlmLatentSource.create(dataset, indices=global_indices)
        if manager.config.vlm_enabled
        else None
    )
    source = vae_source or vlm_source
    if frame_cache_capacity and vae_source is not None:
        vae_source.configure_frame_cache(frame_cache_capacity)
    logger.info(
        "Cache dataset ready: rank=%d samples=%d seconds=%.2f",
        rank,
        len(global_indices),
        time.perf_counter() - dataset_started,
    )
    if global_world_size == 1:
        partitions = (range(len(global_indices)),)
        partition_strategy = "single_rank"
    else:
        partitions = _episode_block_partitions(
            global_indices,
            dataset._episode_starts,
            dataset._episode_ends,
            global_world_size,
        )
        partition_strategy = "episode_block"
    local_positions = partitions[global_rank]
    if global_rank == 0:
        logger.info(
            "Cache rank partition: strategy=%s samples_per_rank=%s",
            partition_strategy,
            [len(partition) for partition in partitions],
        )
    logger.info(
        "Cache local partition: rank=%d/%d strategy=%s positions=[%d,%d) "
        "samples=%d sample_ids=[%d,%d]",
        global_rank,
        global_world_size,
        partition_strategy,
        local_positions.start,
        local_positions.stop,
        len(local_positions),
        global_indices[local_positions.start],
        global_indices[local_positions.stop - 1],
    )
    local_dataset = Subset(
        (
            _UnifiedGenerationDataset(vae_source, vlm_source)
            if vae_source is not None and vlm_source is not None
            else VaeLatentGenerationDataset(vae_source)
            if vae_source is not None
            else VlmLatentGenerationDataset(vlm_source)
        ),
        local_positions,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "in_order": in_order,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = prefetch_factor
        if timeout:
            loader_kwargs["timeout"] = timeout
    loader = DataLoader(local_dataset, **loader_kwargs)

    model_proxy = None
    vae_path = None
    if vae_source is not None:
        vae, vae_path = _load_vae(cfg, device=device, dtype=dtype)
        model_proxy = SimpleNamespace(vae=vae, device=torch.device(device))
    understanding = None
    vlm_path = None
    understanding_cfg: dict[str, Any] = {}
    if vlm_source is not None:
        understanding, vlm_path, understanding_cfg = _load_vlm(
            cfg, device=device, dtype=dtype
        )

    contract_payload: list[Any] = [None]
    contract_error = None
    if rank == 0:
        try:
            contracts: dict[str, ArtifactContract] = {}
            fingerprints: dict[str, str] = {}
            if vae_source is not None:
                assert model_proxy is not None and vae_path is not None
                fingerprint = vae_fingerprint_from_path(vae_path)
                fingerprints["vae"] = fingerprint
                contracts.update(
                    vae_source.contracts(
                        vae=model_proxy.vae,
                        vae_fingerprint=fingerprint,
                        dtype=dtype,
                        value_codec=manager.config.vae_value_codec,
                    )
                )
            if vlm_source is not None:
                assert understanding is not None and vlm_path is not None
                fingerprint = vlm_fingerprint_from_path(vlm_path)
                fingerprints["vlm"] = fingerprint
                contract = vlm_source.contract(
                    producer_fingerprint=fingerprint,
                    context_dim=int(understanding.context_dim),
                    dtype=dtype,
                    understanding_config=understanding_cfg,
                    value_codec=manager.config.vlm_value_codec,
                )
                contracts[contract.artifact_id] = contract
            contract_payload[0] = {
                "fingerprints": fingerprints,
                "contracts": {
                    name: contract.to_dict() for name, contract in contracts.items()
                },
            }
        except Exception as exc:
            contract_error = f"{type(exc).__name__}: {exc}"
    _broadcast_error(active, rank, contract_error)
    if active:
        dist.broadcast_object_list(contract_payload, src=0)
    payload = contract_payload[0]
    if not isinstance(payload, dict) or not isinstance(payload.get("contracts"), dict):
        raise CacheError("Rank 0 did not publish valid artifact contracts.")
    contracts = {
        str(name): ArtifactContract.from_dict(contract)
        for name, contract in payload["contracts"].items()
        if isinstance(contract, dict)
    }
    logger.info(
        "Cache producers ready: rank=%d fingerprints=%s contracts=%s",
        rank,
        payload["fingerprints"],
        list(contracts),
    )

    configured_build_id = str(_get(precompute_cfg, "build_id", "") or "").strip()
    if job_count > 1 and not configured_build_id:
        raise ValueError("Independent cache jobs require a shared precompute.build_id.")
    build_id = configured_build_id or (uuid.uuid4().hex if rank == 0 else "")
    if active:
        build_payload = [build_id]
        dist.broadcast_object_list(build_payload, src=0)
        build_id = str(build_payload[0])
    session = manager.repository.begin_build(
        contracts,
        build_id=build_id,
        rank=global_rank,
        shard_size=shard_size,
    )

    encoded = 0
    fetch_timings: list[float] = []
    encode_timings: list[float] = []
    write_timings: list[float] = []
    fetch_seconds_total = 0.0
    encode_seconds_total = 0.0
    write_seconds_total = 0.0
    frame_cache_hits = 0
    frame_cache_misses = 0
    total_batches = len(loader)
    _write_progress(
        progress_path,
        rank=global_rank,
        world_size=global_world_size,
        phase="encoding",
        batch=0,
        total_batches=total_batches,
        encoded_samples=0,
        total_samples=len(local_dataset),
        build_id=build_id,
    )
    with (
        torch.no_grad(),
        tqdm(
            total=total_batches,
            desc=f"Encoder cache rank {global_rank}/{global_world_size}",
            unit="batch",
            dynamic_ncols=True,
            disable=False,
        ) as bar,
    ):
        iterator = iter(loader)
        for batch_index in range(1, total_batches + 1):
            fetch_started = time.perf_counter()
            batch = next(iterator)
            fetch_seconds = time.perf_counter() - fetch_started
            sample_ids = batch["sample_id"]
            if isinstance(sample_ids, torch.Tensor):
                ids = [int(value) for value in sample_ids.tolist()]
            else:
                ids = [int(value) for value in sample_ids]
            keys = [cache_key_for_sample(value) for value in ids]
            encode_started = time.perf_counter()
            values: dict[str, dict[str, torch.Tensor]] = {}
            if vae_source is not None:
                assert model_proxy is not None
                values = encode_vae_artifacts(
                    model_proxy,
                    batch,
                    fields=manager.config.fields,
                    keep=tuple(range(len(keys))),
                    dtype=dtype,
                    value_codec=manager.config.vae_value_codec,
                )
            vlm_rows = (
                encode_vlm_artifacts(understanding, batch, dtype=dtype)
                if vlm_source is not None
                else []
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(torch.device(device))
            encode_seconds = time.perf_counter() - encode_started
            write_started = time.perf_counter()
            for artifact_id, bundle in values.items():
                session.put_artifact(artifact_id, keys, bundle)
            if vlm_source is not None:
                artifact_id = vlm_artifact_id(manager.config.vlm_value_codec)
                for key, row in zip(keys, vlm_rows):
                    session.put_artifact(artifact_id, [key], row)
            write_seconds = time.perf_counter() - write_started
            fetch_timings.append(fetch_seconds)
            encode_timings.append(encode_seconds)
            write_timings.append(write_seconds)
            fetch_seconds_total += fetch_seconds
            encode_seconds_total += encode_seconds
            write_seconds_total += write_seconds
            frame_cache_hits += _batch_counter(batch, "frame_cache_hits")
            frame_cache_misses += _batch_counter(batch, "frame_cache_misses")
            encoded += len(keys)
            _write_progress(
                progress_path,
                rank=global_rank,
                world_size=global_world_size,
                phase="encoding",
                batch=batch_index,
                total_batches=total_batches,
                encoded_samples=encoded,
                total_samples=len(local_dataset),
                build_id=build_id,
                last_keys=keys[:2],
                fetch_seconds=fetch_seconds,
                encode_seconds=encode_seconds,
                write_seconds=write_seconds,
                fetch_seconds_total=fetch_seconds_total,
                encode_seconds_total=encode_seconds_total,
                write_seconds_total=write_seconds_total,
                frame_cache_hits=frame_cache_hits,
                frame_cache_misses=frame_cache_misses,
            )
            bar.update(1)
            bar.set_postfix(
                encoded=encoded,
                fetch=f"{fetch_seconds:.2f}s",
                encode=f"{encode_seconds:.2f}s",
                write=f"{write_seconds:.2f}s",
                refresh=False,
            )
            if global_rank == 0 and batch_index % progress_every == 0:
                _log_aggregate_progress(log_dir, global_world_size)

    fragments = session.finalize_rank()
    timing_summary = {
        "fetch": _timing_summary(fetch_timings),
        "encode": _timing_summary(encode_timings),
        "write": _timing_summary(write_timings),
    }
    frame_cache_requests = frame_cache_hits + frame_cache_misses
    frame_cache_hit_rate = (
        frame_cache_hits / frame_cache_requests if frame_cache_requests else 0.0
    )
    _write_progress(
        progress_path,
        rank=global_rank,
        world_size=global_world_size,
        phase="rank_finalized",
        batch=total_batches,
        total_batches=total_batches,
        encoded_samples=encoded,
        total_samples=len(local_dataset),
        build_id=build_id,
        fragments=fragments,
        timings=timing_summary,
        in_order=in_order,
        frame_cache_capacity=frame_cache_capacity,
        frame_cache_hits=frame_cache_hits,
        frame_cache_misses=frame_cache_misses,
        frame_cache_hit_rate=frame_cache_hit_rate,
    )
    logger.info(
        "Cache rank finalized: rank=%d samples=%d fetch_total=%.3fs "
        "encode_total=%.3fs write_total=%.3fs frame_cache=%d/%d (%.2f%%) "
        "fragments=%s",
        global_rank,
        encoded,
        timing_summary["fetch"]["total"],
        timing_summary["encode"]["total"],
        timing_summary["write"]["total"],
        frame_cache_hits,
        frame_cache_requests,
        frame_cache_hit_rate * 100.0,
        fragments,
    )
    if active:
        dist.barrier()
    publish_error = None
    report = None
    if global_rank == 0:
        try:
            if job_count > 1:
                _wait_for_rank_fragments(
                    manager.repository,
                    contracts,
                    build_id=build_id,
                    world_size=global_world_size,
                    timeout_seconds=max(
                        0.0,
                        float(
                            _get(
                                precompute_cfg,
                                "publish_timeout",
                                86400,
                            )
                        ),
                    ),
                    poll_seconds=max(
                        0.1,
                        float(
                            _get(
                                precompute_cfg,
                                "publish_poll_interval",
                                30,
                            )
                        ),
                    ),
                )
            report = session.publish(
                world_size=global_world_size,
                expected_keys=source.expected_keys(),
            )
        except Exception as exc:
            publish_error = f"{type(exc).__name__}: {exc}"
    if job_count == 1:
        _broadcast_error(active, rank, publish_error)
        if active:
            dist.barrier()
    elif global_rank == 0 and publish_error is not None:
        raise CacheError(publish_error)
    _write_progress(
        progress_path,
        rank=global_rank,
        world_size=global_world_size,
        phase=(
            "complete" if job_count == 1 or global_rank == 0 else "fragment_complete"
        ),
        batch=total_batches,
        total_batches=total_batches,
        encoded_samples=encoded,
        total_samples=len(local_dataset),
        build_id=build_id,
        fragments=fragments,
        timings=timing_summary,
        in_order=in_order,
        frame_cache_capacity=frame_cache_capacity,
        frame_cache_hits=frame_cache_hits,
        frame_cache_misses=frame_cache_misses,
        frame_cache_hit_rate=frame_cache_hit_rate,
    )
    if job_count > 1 and global_rank != 0:
        logger.info(
            "External cache fragment complete: build=%s rank=%d/%d; "
            "global rank 0 will publish the assembled cache.",
            build_id,
            global_rank,
            global_world_size,
        )
        return None
    if global_rank == 0:
        assert report is not None
        elapsed = max(time.perf_counter() - started, 1e-6)
        for artifact_id, artifact_report in report.artifacts.items():
            logger.info(
                "Published cache artifact validated: artifact=%s build=%s "
                "samples=%d shards=%d bytes=%.2f GiB manifest=%.3fs "
                "index=%.3fs structure=%.3fs shard_data=%.3fs workers=%d "
                "stat_sum=%.3fs read_sum=%.3fs hash_sum=%.3fs "
                "load_sum=%.3fs check_sum=%.3fs total=%.3fs",
                artifact_id,
                artifact_report.build_id,
                artifact_report.sample_count,
                artifact_report.shard_count,
                artifact_report.total_bytes / (1024**3),
                artifact_report.manifest_seconds,
                artifact_report.index_seconds,
                artifact_report.structure_seconds,
                artifact_report.shard_seconds,
                artifact_report.validation_workers,
                artifact_report.shard_stat_seconds,
                artifact_report.shard_read_seconds,
                artifact_report.shard_hash_seconds,
                artifact_report.shard_load_seconds,
                artifact_report.shard_check_seconds,
                artifact_report.total_seconds,
            )
        logger.info(
            "VAE cache published: build=%s artifacts=%s samples=%d "
            "bytes=%.2f GiB elapsed=%.1fs throughput=%.2f samples/s catalog=%s",
            report.build_id,
            list(report.artifacts),
            len(global_indices),
            report.total_bytes / (1024**3),
            elapsed,
            len(global_indices) / elapsed,
            report.catalog,
        )
    return report
