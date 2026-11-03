"""Deterministic, field-scoped cache fingerprints."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ._format import revision_tag
from .fields import CacheField


_SOURCE_INVENTORY_REVISION = 1
_DATASET_REVISION = 2
_IMAGE_PREPROCESS_REVISION = 2
_FIELD_SELECTION_REVISION = 2
_SAMPLE_KEY_REVISION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _dataset_source_inventory(root_text: str) -> str:
    """Conservative identity for LeRobot metadata and source-file inventory."""

    root = Path(root_text)
    if not root.is_dir():
        return stable_fingerprint({"root": root_text, "exists": False})
    records: list[dict[str, Any]] = []
    # Only metadata that controls video layout and episode/sample mapping is a
    # VAE dependency.  Task text and action/statistics metadata deliberately
    # stay out so downstream conditioning changes do not invalidate latents.
    meta_root = root / "meta"
    if meta_root.is_dir():
        for path in sorted(
            candidate for candidate in meta_root.rglob("*") if candidate.is_file()
        ):
            relative_meta_path = path.relative_to(meta_root).as_posix().lower()
            is_info = relative_meta_path == "info.json"
            is_episode_mapping = (
                "episode" in relative_meta_path
                and "stats" not in relative_meta_path
            )
            if not is_info and not is_episode_mapping:
                continue
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    # Full video hashing would reread the entire dataset at every launch.  For
    # large immutable payloads, bind the cache to the complete relative-path,
    # size and nanosecond-mtime inventory.  Replacing source data through the
    # normal filesystem workflows therefore invalidates the cache cheaply.
    payload_suffixes = {
        ".avi",
        ".jpeg",
        ".jpg",
        ".mkv",
        ".mov",
        ".mp4",
        ".parquet",
        ".png",
        ".webm",
    }
    for scope in ("videos", "data", "images"):
        scope_root = root / scope
        if not scope_root.is_dir():
            continue
        for path in sorted(
            candidate for candidate in scope_root.rglob("*") if candidate.is_file()
        ):
            if path.suffix.lower() not in payload_suffixes:
                continue
            stat = path.stat()
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "ctime_ns": int(stat.st_ctime_ns),
                }
            )
    return stable_fingerprint(
        {
            "version": revision_tag(
                "lerobot_source_inventory", _SOURCE_INVENTORY_REVISION
            ),
            "records": records,
        }
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_dataset_fingerprint(
    *,
    dataset_dirs: Iterable[str | Path],
    dataset_length: int,
    episode_ranges: Iterable[tuple[int, int]],
    image_shape_meta: Any,
    global_sample_stride: int,
    val_set_proportion: float,
    is_training_set: bool,
    episode_selection: Mapping[str, Any] | None,
) -> str:
    """Fingerprint RGB sample identity without action/proprio training settings."""

    normalized_dirs = [
        str(Path(path).expanduser().resolve()) for path in dataset_dirs
    ]
    payload = {
        "version": revision_tag("wam_vae_dataset", _DATASET_REVISION),
        "dataset_dirs": normalized_dirs,
        "source_inventories": [
            _dataset_source_inventory(path) for path in normalized_dirs
        ],
        "dataset_length": int(dataset_length),
        "episode_ranges": [
            [int(start), int(end)] for start, end in episode_ranges
        ],
        "image_shape_meta": _jsonable(image_shape_meta),
        "global_sample_stride": int(global_sample_stride),
        "val_set_proportion": float(val_set_proportion),
        "is_training_set": bool(is_training_set),
        "episode_selection": _jsonable(episode_selection),
        "sample_key": revision_tag(
            "global_post_retry_index", _SAMPLE_KEY_REVISION
        ),
    }
    return stable_fingerprint(payload)


def build_common_image_fingerprint(
    *,
    video_size: Iterable[int],
    video_view_names: Iterable[str],
    concat_multi_camera: str | None,
    single_canvas: bool,
    image_color_jitter: Mapping[str, Any] | None,
    image_camera_jitter: Mapping[str, Any] | None,
    image_sensor_noise: Mapping[str, Any] | None,
    processor_video_preprocess: Any,
) -> str:
    payload = {
        "version": revision_tag(
            "robot_video_common_image", _IMAGE_PREPROCESS_REVISION
        ),
        "video_size": [int(value) for value in video_size],
        "video_view_names": [str(value) for value in video_view_names],
        "concat_multi_camera": (
            None if concat_multi_camera is None else str(concat_multi_camera)
        ),
        "single_canvas": bool(single_canvas),
        "image_color_jitter": _jsonable(image_color_jitter),
        "image_camera_jitter": _jsonable(image_camera_jitter),
        "image_sensor_noise": _jsonable(image_sensor_noise),
        "processor_video_preprocess": _jsonable(processor_video_preprocess),
        "resize_transform": "ResizeSmallestSideAspectPreserving",
        "crop_transform": "CenterCrop",
        "normalize": {"mean": 0.5, "std": 0.5},
    }
    return stable_fingerprint(payload)


def build_vae_field_fingerprints(
    *,
    num_frames: int,
    video_sample_indices: Iterable[int],
    action_video_freq_ratio: int,
    memory_video_anchor_size: int,
    memory_recent_frame_offset: int,
    video_layout: str,
) -> dict[CacheField, str]:
    return {
        CacheField.CURRENT: stable_fingerprint(
            {
                "version": revision_tag(
                    "current_window", _FIELD_SELECTION_REVISION
                ),
                "num_frames": int(num_frames),
                "video_sample_indices": [
                    int(value) for value in video_sample_indices
                ],
                "action_video_freq_ratio": int(action_video_freq_ratio),
                "padding": "lerobot_offset_factory",
                "video_layout": str(video_layout),
                "tiled": False,
            }
        ),
        CacheField.MEMORY_ANCHOR: stable_fingerprint(
            {
                "version": revision_tag(
                    "episode_anchor", _FIELD_SELECTION_REVISION
                ),
                "memory_video_anchor_size": int(memory_video_anchor_size),
                "selection": "episode_start_known_frame_repeat",
                "padding": "copy_prior_or_first_valid",
                "video_layout": str(video_layout),
                "tiled": False,
            }
        ),
        CacheField.MEMORY_RECENT: stable_fingerprint(
            {
                "version": revision_tag(
                    "recent_frame", _FIELD_SELECTION_REVISION
                ),
                "memory_recent_frame_offset": int(memory_recent_frame_offset),
                "episode_boundary": "episode_start_with_recent_is_real_false",
                "video_layout": str(video_layout),
                "tiled": False,
            }
        ),
    }


def vae_fingerprint_from_path(path: str | Path | Iterable[str | Path]) -> str:
    """Hash VAE checkpoint bytes, rather than only tensor names and shapes."""

    def checkpoint_files(item: str | Path) -> list[Path]:
        resolved = Path(item).expanduser().resolve()
        if resolved.is_file():
            return [resolved]
        if resolved.is_dir():
            suffixes = {".bin", ".pth", ".pt", ".safetensors"}
            files = sorted(
                candidate
                for candidate in resolved.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in suffixes
            )
            if files:
                return files
        raise FileNotFoundError(f"Unable to fingerprint VAE checkpoint: {resolved}")

    paths = [path] if isinstance(path, (str, Path)) else list(path)
    files = [file for item in paths for file in checkpoint_files(item)]
    if not files:
        raise ValueError("VAE checkpoint fingerprint requires at least one file.")
    file_digests = []
    for file in files:
        file_digests.append(_sha256_file(file))
    wam_root = Path(__file__).resolve().parents[1]
    implementation_paths = (
        wam_root / "model/backbones/wan22/wan_video_vae.py",
        wam_root / "model/backbones/wan22/helpers/loader.py",
        wam_root / "model/backbones/wan22/helpers/state_dict_converters.py",
        wam_root / "model/modules/codecs/video_latent_codec.py",
    )
    implementation_digests = {
        path.relative_to(wam_root).as_posix(): _sha256_file(path)
        for path in implementation_paths
    }
    # Sorting makes a sharded checkpoint independent of path and caller order,
    # while source digests also bind scaling, conversion and encode semantics.
    composite = stable_fingerprint(
        {
            "algorithm": "sha256",
            "file_digests": sorted(file_digests),
            "implementation_digests": implementation_digests,
        }
    )
    return f"wan_video_vae:sha256:{composite}"


def vae_fingerprint_from_model(model: Any) -> str | None:
    model_paths = getattr(model, "model_paths", None) or {}
    vae_path = model_paths.get("vae") if isinstance(model_paths, Mapping) else None
    if vae_path:
        return vae_fingerprint_from_path(vae_path)
    # A class/config signature cannot distinguish two checkpoints with the
    # same tensor structure.  Strict read mode therefore refuses models that
    # do not expose their VAE checkpoint path.
    return None


def vlm_fingerprint_from_path(path: str | Path) -> str:
    """Fingerprint VLM weights, processor config and the InternW0-delta extraction code."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Unable to fingerprint VLM checkpoint: {root}")
    suffixes = {".bin", ".json", ".model", ".py", ".safetensors", ".txt"}
    files = [root] if root.is_file() else sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in suffixes
    )
    if not files:
        raise ValueError(f"VLM checkpoint contains no fingerprintable files: {root}")
    records = [
        {
            "name": file.name if root.is_file() else file.relative_to(root).as_posix(),
            "bytes": int(file.stat().st_size),
            "sha256": _sha256_file(file),
        }
        for file in files
    ]
    implementation = (
        Path(__file__).resolve().parents[1]
        / "model/modules/understanding/qwen_vl_encoder.py"
    )
    digest = stable_fingerprint(
        {
            "algorithm": "sha256",
            "files": records,
            "implementation_sha256": _sha256_file(implementation),
        }
    )
    return f"qwen_vl:sha256:{digest}"

