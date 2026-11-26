from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path, PurePath
from typing import Any

from torch import Tensor

MODEL_FILENAME = "model.safetensors"
MODEL_INDEX_FILENAME = "model.safetensors.index.json"
LOGGER = logging.getLogger(__name__)


def checkpoint_weight_files(directory: str | Path) -> list[Path]:
    """Return and validate the files comprising one HF safetensors checkpoint."""

    root = Path(directory)
    if root.is_file():
        if root.suffix != ".safetensors":
            raise ValueError(f"unsupported model weights file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")

    index_path = root / MODEL_INDEX_FILENAME
    single_path = root / MODEL_FILENAME
    if index_path.exists() and single_path.exists():
        raise ValueError("checkpoint contains both single-file and sharded weights")
    if single_path.is_file():
        return [single_path]
    if not index_path.is_file():
        raise FileNotFoundError(
            f"checkpoint has neither {MODEL_FILENAME} nor {MODEL_INDEX_FILENAME}: {root}"
        )

    index = _read_index(index_path)
    shard_names = sorted(set(index["weight_map"].values()))
    files = [index_path]
    for name in shard_names:
        shard = root / name
        if not shard.is_file() or shard.is_symlink():
            raise FileNotFoundError(f"checkpoint shard is missing: {shard}")
        files.append(shard)
    return files




def load_checkpoint_state_dict(path: str | Path) -> dict[str, Tensor]:
    """Load old single-file or HF-sharded safetensors with strict index checks."""

    source = Path(path)
    files = checkpoint_weight_files(source)
    if len(files) == 1:
        return _load_safetensors_file(files[0])

    index = _read_index(files[0])
    expected = index["weight_map"]
    merged: dict[str, Tensor] = {}
    tensor_files: dict[str, str] = {}
    for shard in files[1:]:
        values = _load_safetensors_file(shard)
        duplicate = set(values).intersection(merged)
        if duplicate:
            preview = ", ".join(sorted(duplicate)[:5])
            raise ValueError(f"duplicate tensor keys across checkpoint shards: {preview}")
        merged.update(values)
        tensor_files.update(dict.fromkeys(values, shard.name))

    missing = sorted(set(expected) - set(merged))
    unexpected = sorted(set(merged) - set(expected))
    misplaced = sorted(
        key
        for key, filename in expected.items()
        if key in tensor_files and tensor_files[key] != filename
    )
    if missing or unexpected or misplaced:
        raise ValueError(
            "checkpoint index does not match shard contents: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"misplaced={misplaced[:5]}"
        )
    return merged


def _load_safetensors_file(path: Path) -> dict[str, Tensor]:
    """Retry newly published shard headers before using a buffered read."""

    from safetensors import SafetensorError
    from safetensors.torch import load, load_file

    for attempt in range(3):
        try:
            return dict(load_file(str(path), device="cpu"))
        except SafetensorError as error:
            if "header" not in str(error).lower():
                raise SafetensorError(f"Failed to load {path}: {error}") from error
            if attempt < 2:
                LOGGER.warning(
                    "Cannot read safetensors header in %s (%s); retrying (%d/2)",
                    path, error, attempt + 1,
                )
                time.sleep(float(attempt + 1))

    # Some shared mounts expose a valid buffered read before their mmap view
    # becomes coherent. Only allocate the shard buffer after checking its
    # header; permanently truncated/corrupt files must still fail closed.
    LOGGER.warning("Retrying safetensors shard with buffered I/O: %s", path)
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            header_size = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 0 < header_size <= min(
                100_000_000, os.fstat(handle.fileno()).st_size - 8
            ):
                raise ValueError("invalid or truncated safetensors header length")
            header = json.loads(
                handle.read(header_size), object_pairs_hook=_unique_json_object
            )
            if not isinstance(header, dict):
                raise ValueError("safetensors header must be an object")
            handle.seek(0)
            return dict(load(handle.read()))
    except (OSError, ValueError, SafetensorError) as error:
        raise SafetensorError(f"Failed to load {path} with buffered I/O: {error}") from error








def _read_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique_json_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint index {path}: {error}") from error
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has an invalid weight_map: {path}")
    for key, filename in weight_map.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(filename, str)
            or PurePath(filename).name != filename
            or not filename.endswith(".safetensors")
            or filename == MODEL_FILENAME
        ):
            raise ValueError(f"checkpoint index has an unsafe entry: {key!r}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value
