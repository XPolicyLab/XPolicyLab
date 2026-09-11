from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

from huggingface_hub import split_torch_state_dict_into_shards
from torch import Tensor

MODEL_FILENAME = "model.safetensors"
MODEL_INDEX_FILENAME = "model.safetensors.index.json"
MODEL_FILENAME_PATTERN = "model{suffix}.safetensors"
DEFAULT_MAX_SHARD_SIZE = "5GB"


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


def save_checkpoint_state_dict(
    state_dict: Mapping[str, Tensor],
    directory: str | Path,
    *,
    max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
) -> list[Path]:
    """Atomically write a state dict using the Hugging Face shard convention."""

    from safetensors.torch import save_file

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors = _portable_tensors(state_dict)
    split = split_torch_state_dict_into_shards(
        tensors,
        filename_pattern=MODEL_FILENAME_PATTERN,
        max_shard_size=max_shard_size,
    )

    expected = set(split.filename_to_tensors)
    if split.is_sharded:
        expected.add(MODEL_INDEX_FILENAME)
    _remove_stale_weights(root, expected)

    written: list[Path] = []
    for filename, keys in split.filename_to_tensors.items():
        path = root / filename
        temporary = root / f".{filename}.{os.getpid()}.tmp"
        save_file({key: tensors[key] for key in keys}, str(temporary))
        os.replace(temporary, path)
        written.append(path)

    if split.is_sharded:
        index = {
            "metadata": dict(split.metadata),
            "weight_map": dict(sorted(split.tensor_to_filename.items())),
        }
        index_path = root / MODEL_INDEX_FILENAME
        _atomic_write_json(index_path, index)
        written.insert(0, index_path)
    return written


def load_checkpoint_state_dict(path: str | Path) -> dict[str, Tensor]:
    """Load old single-file or HF-sharded safetensors with strict index checks."""

    from safetensors.torch import load_file

    source = Path(path)
    files = checkpoint_weight_files(source)
    if len(files) == 1:
        return dict(load_file(str(files[0]), device="cpu"))

    index = _read_index(files[0])
    expected = index["weight_map"]
    merged: dict[str, Tensor] = {}
    tensor_files: dict[str, str] = {}
    for shard in files[1:]:
        values = dict(load_file(str(shard), device="cpu"))
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


def checkpoint_weights_sha256(path: str | Path) -> str:
    """Hash the ordered index and shards; preserve the legacy single-file hash."""

    files = checkpoint_weight_files(path)
    if len(files) == 1:
        return f"sha256:{_sha256_file(files[0])}"
    records = [
        {
            "path": file_path.relative_to(Path(path)).as_posix(),
            "size_bytes": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
        }
        for file_path in files
    ]
    return checkpoint_weights_sha256_from_records(path, records)


def checkpoint_weights_sha256_from_records(
    path: str | Path,
    records: list[Mapping[str, Any]],
) -> str:
    """Derive the model hash from already verified artifact checksums."""

    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ValueError("artifact checksum records must be a list of mappings")
    files = checkpoint_weight_files(path)
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record.get("path"), str)
    }
    root = Path(path)
    if len(files) == 1:
        relative = files[0].name
        record = by_path.get(relative)
        if record is None or not isinstance(record.get("sha256"), str):
            raise ValueError(f"artifact checksums are missing model weight {relative!r}")
        return f"sha256:{record['sha256']}"

    root = Path(path)
    aggregate = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        record = by_path.get(relative)
        if record is None:
            raise ValueError(f"artifact checksums are missing model weight {relative!r}")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if not isinstance(digest, str) or not isinstance(size, int):
            raise ValueError(f"artifact checksum record is invalid for {relative!r}")
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
    return f"sha256:{aggregate.hexdigest()}"


def _portable_tensors(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    storages: set[tuple[int, int]] = set()
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError("checkpoint state only accepts tensor values")
        tensor = value.detach().cpu().contiguous()
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes())
        if identity in storages:
            tensor = tensor.clone()
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
        storages.add(identity)
        tensors[str(key)] = tensor
    return tensors


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


def _remove_stale_weights(root: Path, expected: set[str]) -> None:
    candidates = {
        MODEL_FILENAME,
        MODEL_INDEX_FILENAME,
        *(path.name for path in root.glob("model-*-of-*.safetensors")),
    }
    for filename in candidates - expected:
        path = root / filename
        if path.is_file():
            path.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
