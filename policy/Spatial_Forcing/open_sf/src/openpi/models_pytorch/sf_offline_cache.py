from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SFCacheKey:
    dataset_uid: int
    episode_index: int
    step_index: int


def _safe_int(x) -> int:
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def make_cache_key(dataset_uid, episode_index, step_index) -> SFCacheKey:
    return SFCacheKey(
        dataset_uid=_safe_int(dataset_uid),
        episode_index=_safe_int(episode_index),
        step_index=_safe_int(step_index),
    )


def normalize_cache_dtype(cache_dtype: str) -> str:
    s = str(cache_dtype).lower()
    if s in ("fp16", "float16", "f16"):
        return "fp16"
    if s in ("bf16", "bfloat16"):
        return "bf16"
    if s in ("int8", "i8"):
        return "int8"
    raise ValueError(f"Unsupported SF cache dtype: {cache_dtype}. Use 'fp16', 'bf16', or 'int8'.")


def _chunk_base_path(cache_dir: str | os.PathLike, key: SFCacheKey, cache_dtype: str, chunk_size: int) -> tuple[Path, int]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    chunk_idx = key.step_index // chunk_size
    slot_idx = key.step_index % chunk_size
    base = (
        Path(cache_dir)
        / f"ds_{key.dataset_uid}"
        / f"ep_{key.episode_index}"
        / f"{cache_dtype}_chunk_{chunk_idx:08d}"
    )
    return base, slot_idx


def _ensure_file_size(path: Path, expected_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        current_size = os.fstat(fd).st_size
        if current_size == 0:
            os.ftruncate(fd, expected_size)
        elif current_size != expected_size:
            raise ValueError(f"Unexpected chunk file size for {path}: got={current_size}, expected={expected_size}")
    finally:
        os.close(fd)


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_mask_value(mask_path: Path, slot_idx: int) -> int:
    if not mask_path.exists():
        return 0
    with open(mask_path, "rb") as f:
        f.seek(slot_idx)
        b = f.read(1)
    if not b:
        return 0
    return int(b[0])


def _write_bytes_at(path: Path, offset: int, data: bytes) -> None:
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)


def _read_bytes_at(path: Path, offset: int, size: int) -> bytes | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        return None
    return data


def _decode_fp16_frame(raw: bytes, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    numel = expected_shape[0] * expected_shape[1]
    a = np.frombuffer(raw, dtype=np.float16, count=numel)
    if a.size != numel:
        return None
    return torch.from_numpy(a.copy()).reshape(expected_shape)


def _decode_bf16_frame(raw: bytes, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    numel = expected_shape[0] * expected_shape[1]
    a = np.frombuffer(raw, dtype=np.int16, count=numel)
    if a.size != numel:
        return None
    storage = torch.from_numpy(a.copy()).reshape(expected_shape)
    return storage.view(torch.bfloat16)


def _decode_int8_frame(q_raw: bytes, s_raw: bytes, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    p, d = expected_shape
    q_np = np.frombuffer(q_raw, dtype=np.int8, count=p * d)
    s_np = np.frombuffer(s_raw, dtype=np.float16, count=p)
    if q_np.size != p * d or s_np.size != p:
        return None
    q = torch.from_numpy(q_np.copy()).reshape(p, d).to(dtype=torch.float32)
    s = torch.from_numpy(s_np.copy()).reshape(p, 1).to(dtype=torch.float32)
    return (q * s).to(dtype=torch.float16)


def load_cached_tensor(
    cache_dir: str | os.PathLike,
    key: SFCacheKey,
    cache_dtype: str,
    chunk_size: int,
    expected_shape: Optional[Tuple[int, ...]] = None,
    strict_shape: bool = True,
) -> torch.Tensor | None:
    if expected_shape is None or len(expected_shape) != 2:
        raise ValueError(f"expected_shape must be 2D tuple, got {expected_shape}")
    expected_shape_2d = (int(expected_shape[0]), int(expected_shape[1]))
    cache_dtype = normalize_cache_dtype(cache_dtype)

    base, slot_idx = _chunk_base_path(cache_dir, key, cache_dtype, chunk_size)
    mask_path = base.with_suffix(".mask")
    if _read_mask_value(mask_path, slot_idx) != 1:
        return None

    p, d = expected_shape_2d
    try:
        if cache_dtype == "fp16":
            raw = _read_bytes_at(base.with_suffix(".f16bin"), slot_idx * (p * d * 2), p * d * 2)
            if raw is None:
                return None
            tensor = _decode_fp16_frame(raw, expected_shape_2d)
        elif cache_dtype == "bf16":
            raw = _read_bytes_at(base.with_suffix(".bf16bin"), slot_idx * (p * d * 2), p * d * 2)
            if raw is None:
                return None
            tensor = _decode_bf16_frame(raw, expected_shape_2d)
        elif cache_dtype == "int8":
            q_raw = _read_bytes_at(base.with_suffix(".i8bin"), slot_idx * (p * d), p * d)
            s_raw = _read_bytes_at(base.with_suffix(".s16bin"), slot_idx * (p * 2), p * 2)
            if q_raw is None or s_raw is None:
                return None
            tensor = _decode_int8_frame(q_raw, s_raw, expected_shape_2d)
        else:
            raise ValueError(f"Unsupported cache_dtype: {cache_dtype}")
        if tensor is None or tuple(tensor.shape) != expected_shape_2d:
            return None if strict_shape else tensor
        return tensor
    except Exception:
        return None


def save_cached_tensor(
    cache_dir: str | os.PathLike,
    key: SFCacheKey,
    tensor: torch.Tensor,
    cache_dtype: str,
    chunk_size: int,
    overwrite: bool = False,
) -> bool:
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor [P, D], got shape={tuple(tensor.shape)}")

    cache_dtype = normalize_cache_dtype(cache_dtype)
    frame = tensor.detach().to(device="cpu").contiguous()
    p, d = int(frame.shape[0]), int(frame.shape[1])
    base, slot_idx = _chunk_base_path(cache_dir, key, cache_dtype, chunk_size)
    lock_path = base.with_suffix(".lock")
    mask_path = base.with_suffix(".mask")

    with _exclusive_lock(lock_path):
        _ensure_file_size(mask_path, chunk_size)
        if not overwrite and _read_mask_value(mask_path, slot_idx) == 1:
            return False

        if cache_dtype == "fp16":
            data_path = base.with_suffix(".f16bin")
            slot_bytes = p * d * 2
            _ensure_file_size(data_path, chunk_size * slot_bytes)
            payload = frame.to(dtype=torch.float16).numpy().tobytes(order="C")
            _write_bytes_at(data_path, slot_idx * slot_bytes, payload)
        elif cache_dtype == "bf16":
            data_path = base.with_suffix(".bf16bin")
            slot_bytes = p * d * 2
            _ensure_file_size(data_path, chunk_size * slot_bytes)
            payload = frame.to(dtype=torch.bfloat16).view(torch.int16).numpy().tobytes(order="C")
            _write_bytes_at(data_path, slot_idx * slot_bytes, payload)
        elif cache_dtype == "int8":
            q_path = base.with_suffix(".i8bin")
            s_path = base.with_suffix(".s16bin")
            _ensure_file_size(q_path, chunk_size * p * d)
            _ensure_file_size(s_path, chunk_size * p * 2)
            frame_f32 = frame.to(dtype=torch.float32)
            scales = frame_f32.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            q = torch.round(frame_f32 / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            _write_bytes_at(q_path, slot_idx * p * d, q.numpy().tobytes(order="C"))
            _write_bytes_at(s_path, slot_idx * p * 2, scales.to(dtype=torch.float16).numpy().tobytes(order="C"))
        else:
            raise ValueError(f"Unsupported cache_dtype: {cache_dtype}")

        with open(mask_path, "r+b") as mask_f:
            mask_f.seek(slot_idx)
            mask_f.write(b"\x01")
    return True
