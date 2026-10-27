from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


TASK_INDEX_SHARDED_MMAP_V1 = "task_index_sharded_mmap_v1"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def tasks_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _OpenTextShard:
    start: int
    end: int
    offsets: np.ndarray
    lengths: np.ndarray
    prompt_hashes: np.ndarray
    embeddings: np.memmap


class TaskIndexShardedTextEmbeddingCache:
    """Lazy mmap reader for T5 embeddings keyed by LeRobot ``task_index``.

    Each shard stores only valid tokens. ``get`` reconstructs the padded
    ``[context_len, embedding_dim]`` tensor expected by FastWAM and verifies
    that the caller's prompt still matches the indexed task.
    """

    def __init__(self, root: str | Path, *, expected_context_len: int | None = None):
        self.root = Path(root).expanduser()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing completed text cache manifest: {manifest_path}. "
                "Run scripts/precompute_text_embeds_sharded.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fmt = str(self.manifest.get("format", "")).strip().lower()
        if fmt != TASK_INDEX_SHARDED_MMAP_V1:
            raise ValueError(
                f"Unsupported text cache format {fmt!r} in {manifest_path}; "
                f"expected {TASK_INDEX_SHARDED_MMAP_V1!r}."
            )

        self.total_tasks = int(self.manifest["total_tasks"])
        self.context_len = int(self.manifest["context_len"])
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.shard_size = int(self.manifest["shard_size"])
        if self.total_tasks < 1 or self.context_len < 1 or self.embedding_dim < 1 or self.shard_size < 1:
            raise ValueError(f"Invalid text cache dimensions in {manifest_path}: {self.manifest}")
        if expected_context_len is not None and self.context_len != int(expected_context_len):
            raise ValueError(
                f"Text cache context_len={self.context_len} does not match expected "
                f"{int(expected_context_len)} in {manifest_path}."
            )
        logical_dtype = str(self.manifest.get("logical_dtype", "bfloat16")).lower()
        storage_dtype = str(self.manifest.get("storage_dtype", "uint16")).lower()
        if logical_dtype not in {"bfloat16", "bf16"} or storage_dtype != "uint16":
            raise ValueError(
                "task_index_sharded_mmap_v1 currently requires logical_dtype=bfloat16 "
                f"and storage_dtype=uint16, got {logical_dtype}/{storage_dtype}."
            )
        self._open_shards: dict[int, _OpenTextShard] = {}

    @staticmethod
    def _shard_name(start: int, end: int) -> str:
        return f"shard_{start:07d}_{end:07d}"

    def _open_shard(self, shard_id: int) -> _OpenTextShard:
        cached = self._open_shards.get(shard_id)
        if cached is not None:
            return cached

        start = shard_id * self.shard_size
        end = min(start + self.shard_size, self.total_tasks)
        shard_dir = self.root / "shards" / self._shard_name(start, end)
        shard_manifest_path = shard_dir / "manifest.json"
        if not shard_manifest_path.is_file():
            raise FileNotFoundError(f"Missing text embedding shard manifest: {shard_manifest_path}")
        shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
        if int(shard_manifest.get("start", -1)) != start or int(shard_manifest.get("end", -1)) != end:
            raise ValueError(f"Text shard range mismatch in {shard_manifest_path}.")

        offsets = np.load(shard_dir / "offsets.npy", mmap_mode="r")
        lengths = np.load(shard_dir / "lengths.npy", mmap_mode="r")
        prompt_hashes = np.load(shard_dir / "prompt_sha256.npy", mmap_mode="r")
        count = end - start
        if offsets.shape != (count + 1,) or lengths.shape != (count,) or prompt_hashes.shape != (count,):
            raise ValueError(
                f"Invalid text shard index shapes in {shard_dir}: "
                f"offsets={offsets.shape} lengths={lengths.shape} hashes={prompt_hashes.shape}."
            )
        total_tokens = int(offsets[-1])
        if total_tokens < 0:
            raise ValueError(f"Negative token count in {shard_dir}.")
        embeddings_path = shard_dir / "embeddings.uint16.mmap"
        expected_bytes = total_tokens * self.embedding_dim * np.dtype(np.uint16).itemsize
        actual_bytes = embeddings_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Text shard payload size mismatch in {embeddings_path}: "
                f"got {actual_bytes}, expected {expected_bytes}."
            )
        embeddings = np.memmap(
            embeddings_path,
            mode="r",
            dtype=np.uint16,
            shape=(total_tokens, self.embedding_dim),
        )
        opened = _OpenTextShard(start, end, offsets, lengths, prompt_hashes, embeddings)
        self._open_shards[shard_id] = opened
        return opened

    def get(self, task_index: int, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        task_index = int(task_index)
        if task_index < 0 or task_index >= self.total_tasks:
            raise IndexError(
                f"task_index={task_index} is outside text cache range [0,{self.total_tasks})."
            )
        shard = self._open_shard(task_index // self.shard_size)
        local = task_index - shard.start
        length = int(shard.lengths[local])
        token_start = int(shard.offsets[local])
        token_end = int(shard.offsets[local + 1])
        if length < 1 or length > self.context_len or token_end - token_start != length:
            raise ValueError(
                f"Invalid cached token range for task_index={task_index}: "
                f"length={length}, offsets=({token_start},{token_end})."
            )

        cached_hash_raw: Any = shard.prompt_hashes[local]
        if isinstance(cached_hash_raw, bytes):
            cached_hash = cached_hash_raw.decode("ascii")
        else:
            cached_hash = str(cached_hash_raw)
        actual_hash = prompt_sha256(prompt)
        if cached_hash != actual_hash:
            raise ValueError(
                f"Prompt hash mismatch for task_index={task_index}: cache={cached_hash} "
                f"actual={actual_hash}. The dataset/tasks file or prompt template changed."
            )

        packed = np.asarray(shard.embeddings[token_start:token_end], dtype=np.uint16).copy()
        valid_context = torch.from_numpy(packed).view(torch.bfloat16)
        context = torch.zeros((self.context_len, self.embedding_dim), dtype=torch.bfloat16)
        context[:length].copy_(valid_context)
        mask = torch.zeros(self.context_len, dtype=torch.bool)
        mask[:length] = True
        return context, mask
