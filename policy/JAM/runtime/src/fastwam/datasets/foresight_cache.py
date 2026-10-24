"""Precomputed foresight-teacher feature cache (bf16 + per-sample zstd, bit-exact).

Layout (one or more roots, shards striped across them by free space at write time):
    <root>/meta.json                       # global: layers, seq_len, dim, num_samples, shard_size, codec
    <root>/shard_<start:08d>.bin           # concatenated per-sample zstd frames
    <root>/shard_<start:08d>.idx.npy       # int64 [shard_size, 2]: (offset, compressed_length)

Each record decompresses to bf16 bytes of shape [num_layers, seq_len, dim].
For historical encoder-hidden caches, ``seq_len`` contains every latent frame.
Compact VAE-direct caches may store only selected future offsets at raw patch
dimension (192); the reader expands the missing temporal slots with zeros, and
the model applies the same fixed isometric lift as the online path. bf16 is
stored via a uint16 view, so write->read round-trip is bit-exact; zstd is
lossless.

Consistency note: the base dataset's __getitem__ silently substitutes a RANDOM
sample when decoding fails. `ForesightCacheDataset` therefore bypasses that
fallback and implements its own, so the substituted sample and its cached
target always use the SAME index.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import zstandard

logger = logging.getLogger(__name__)

META_NAME = "meta.json"
ZSTD_LEVEL = 6  # good ratio, >500MB/s decompress; compression speed is not the bottleneck


def shard_name(start: int) -> str:
    return f"shard_{start:08d}"


class ShardWriter:
    """Writes one shard (a contiguous idx range) atomically: tmp files -> rename."""

    def __init__(self, root: Path, start: int, size: int):
        self.root = Path(root)
        self.start = int(start)
        self.size = int(size)
        self._cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._tmp_bin = tempfile.NamedTemporaryFile(
            dir=self.root, prefix=shard_name(start) + ".", suffix=".bin.tmp", delete=False
        )
        self._index = np.full((self.size, 2), -1, dtype=np.int64)
        self._offset = 0
        self._written = 0

    def add(self, idx: int, tensor: torch.Tensor) -> None:
        """tensor: [num_layers, seq_len, dim] bf16 on CPU."""
        rel = idx - self.start
        if not (0 <= rel < self.size):
            raise IndexError(f"idx {idx} outside shard [{self.start}, {self.start + self.size})")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"expected bf16, got {tensor.dtype}")
        raw = tensor.contiguous().view(torch.uint16).numpy().tobytes()
        comp = self._cctx.compress(raw)
        self._tmp_bin.write(comp)
        self._index[rel] = (self._offset, len(comp))
        self._offset += len(comp)
        self._written += 1

    def finalize(self) -> None:
        if self._written != self.size:
            self.abort()
            raise RuntimeError(
                f"shard {self.start}: wrote {self._written}/{self.size} records; aborted (tmp removed)"
            )
        self._tmp_bin.flush()
        os.fsync(self._tmp_bin.fileno())
        self._tmp_bin.close()
        idx_tmp = self.root / (shard_name(self.start) + ".idx.npy.tmp")
        with open(idx_tmp, "wb") as f:  # explicit handle: np.save must not rename/append
            np.save(f, self._index)
        # .idx first would let a reader see an index without data; rename .bin first.
        os.rename(self._tmp_bin.name, self.root / (shard_name(self.start) + ".bin"))
        os.rename(idx_tmp, self.root / (shard_name(self.start) + ".idx.npy"))

    def abort(self) -> None:
        try:
            self._tmp_bin.close()
            os.unlink(self._tmp_bin.name)
        except OSError:
            pass


def write_meta(root: Path, *, layers: Sequence[int], seq_len: int, dim: int,
               num_samples: int, shard_size: int, extra: Optional[dict] = None) -> None:
    meta = {
        "format_version": 1,
        "codec": "zstd",
        "dtype": "bfloat16",
        "layers": [int(x) for x in layers],
        "seq_len": int(seq_len),
        "dim": int(dim),
        "num_samples": int(num_samples),
        "shard_size": int(shard_size),
    }
    if extra:
        meta.update(extra)
    path = Path(root) / META_NAME
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: existing.get(k) for k in meta} != meta:
            raise RuntimeError(f"conflicting {path}: existing={existing} new={meta}")
        return
    # Every extraction rank discovers its first batch independently.  Use a
    # rank-unique temporary file so simultaneous, identical meta writes cannot
    # clobber a shared ``meta.json.tmp`` pathname.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def pick_root(roots: Sequence[Path], need_bytes: int) -> Path:
    """Route a new shard to the root with the most free space (with headroom)."""
    best, best_free = None, -1
    for r in roots:
        free = shutil.disk_usage(r).free
        if free > best_free:
            best, best_free = r, free
    if best is None or best_free < need_bytes * 2:
        raise RuntimeError(f"no cache root has {need_bytes * 2} bytes free: {list(roots)}")
    return best


class ForesightCacheReader:
    """Random-access reader across multiple roots. Bit-exact bf16 records."""

    def __init__(self, roots: Sequence[str | Path]):
        self.roots = [Path(r) for r in roots]
        metas = [json.loads((r / META_NAME).read_text()) for r in self.roots if (r / META_NAME).exists()]
        if not metas:
            raise FileNotFoundError(f"no {META_NAME} under any cache root: {self.roots}")
        for m in metas[1:]:
            if m != metas[0]:
                raise RuntimeError(f"cache roots have conflicting meta: {metas[0]} vs {m}")
        self.meta = metas[0]
        if self.meta["dtype"] != "bfloat16" or self.meta["codec"] != "zstd":
            raise RuntimeError(f"unsupported cache meta: {self.meta}")
        self.layers: list[int] = list(self.meta["layers"])
        self.target_type: str = str(self.meta.get("target_type", "encoder_hidden"))
        self.stored_offsets: Optional[list[int]] = (
            [int(value) for value in self.meta["stored_offsets"]]
            if self.meta.get("stored_offsets") is not None
            else None
        )
        self.tokens_per_frame: Optional[int] = (
            int(self.meta["tokens_per_frame"])
            if self.meta.get("tokens_per_frame") is not None
            else None
        )
        self.num_target_frames: Optional[int] = (
            int(self.meta["num_target_frames"])
            if self.meta.get("num_target_frames") is not None
            else None
        )
        self.stored_seq_len: int = int(self.meta["seq_len"])
        self.seq_len: int = self.stored_seq_len
        self.dim: int = self.meta["dim"]
        self.num_samples: int = self.meta["num_samples"]
        self.shard_size: int = self.meta["shard_size"]
        if self.stored_offsets is not None:
            if self.tokens_per_frame is None or self.num_target_frames is None:
                raise RuntimeError(
                    "compact cache metadata requires tokens_per_frame and num_target_frames: "
                    f"{self.meta}"
                )
            if len(set(self.stored_offsets)) != len(self.stored_offsets):
                raise RuntimeError(f"duplicate compact-cache offsets: {self.stored_offsets}")
            if any(offset < 0 or offset >= self.num_target_frames for offset in self.stored_offsets):
                raise RuntimeError(
                    f"compact-cache offsets {self.stored_offsets} outside "
                    f"[0, {self.num_target_frames})."
                )
            expected_stored_seq = len(self.stored_offsets) * self.tokens_per_frame
            if self.stored_seq_len != expected_stored_seq:
                raise RuntimeError(
                    f"compact-cache seq_len={self.stored_seq_len}, expected "
                    f"len(stored_offsets)*tokens_per_frame={expected_stored_seq}."
                )
            self.seq_len = self.num_target_frames * self.tokens_per_frame
        self._stored_record_shape = (
            len(self.layers),
            self.stored_seq_len,
            self.dim,
        )
        self._dctx = None  # per-process (created lazily; not fork/pickle-safe)
        self._shard_paths: dict[int, Path] = {}
        for r in self.roots:
            for p in r.glob("shard_*.bin"):
                self._shard_paths.setdefault(int(p.stem.split("_")[1]), p)
        self._indices: dict[int, np.ndarray] = {}
        self._files: dict[int, object] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_dctx"] = None
        state["_files"] = {}
        state["_indices"] = {}
        return state

    def missing_shards(self) -> list[int]:
        starts = set(range(0, self.num_samples, self.shard_size))
        return sorted(starts - set(self._shard_paths))

    def get(self, idx: int) -> torch.Tensor:
        """-> bf16 tensor [num_layers, seq_len, dim], bit-identical to what was written."""
        if not (0 <= idx < self.num_samples):
            raise IndexError(idx)
        start = (idx // self.shard_size) * self.shard_size
        path = self._shard_paths.get(start)
        if path is None:
            raise FileNotFoundError(f"cache shard for idx {idx} (start {start}) not found in {self.roots}")
        if start not in self._indices:
            self._indices[start] = np.load(path.with_suffix("").with_suffix(".idx.npy"))
            self._files[start] = open(path, "rb", buffering=0)
        offset, length = self._indices[start][idx - start]
        if offset < 0:
            raise RuntimeError(f"cache record missing for idx {idx}")
        f = self._files[start]
        f.seek(int(offset))
        comp = f.read(int(length))
        # Cache streams ~350GB+/epoch through a 96GB cgroup where PAGE CACHE counts:
        # keeping pages is futile and rides the memsw cap (observed cause of the
        # cached-era crash-rate jump). Drop them immediately after each read.
        try:
            os.posix_fadvise(f.fileno(), int(offset), int(length), os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        if self._dctx is None:
            self._dctx = zstandard.ZstdDecompressor()
        raw = self._dctx.decompress(comp)
        arr = np.frombuffer(raw, dtype=np.uint16).copy().reshape(self._stored_record_shape)
        stored = torch.from_numpy(arr).view(torch.bfloat16)
        if self.stored_offsets is None:
            return stored

        # Compact VAE-direct records are laid out as [stored_offset, spatial_token].
        # Restore the full temporal grid so the existing padding mask and per-frame
        # loss retain their exact offset semantics. Unstored slots are never selected
        # by a validated config and therefore remain zero without affecting loss.
        stored = stored.view(
            len(self.layers),
            len(self.stored_offsets),
            self.tokens_per_frame,
            self.dim,
        )
        expanded = torch.zeros(
            (
                len(self.layers),
                self.num_target_frames,
                self.tokens_per_frame,
                self.dim,
            ),
            dtype=torch.bfloat16,
        )
        expanded[:, self.stored_offsets] = stored
        return expanded.reshape(len(self.layers), self.seq_len, self.dim)


class ForesightCacheDataset(torch.utils.data.Dataset):
    """Wraps the base dataset, attaching `foresight_enc_cache` [L, seq_len, dim] per sample.

    Bypasses the base class's random-substitution error fallback and re-implements
    it consistently: whatever index ends up being used, data AND cached target come
    from that same index.
    """

    SAMPLE_KEY = "foresight_enc_cache"
    INPUT_LATENT_SAMPLE_KEY = "input_latents_cache"

    def __init__(self, base: torch.utils.data.Dataset, roots: Sequence[str | Path],
                 layers: Sequence[int], target_type: str = "encoder_hidden",
                 offsets: Optional[Sequence[int]] = None,
                 input_latent_roots: Optional[Sequence[str | Path]] = None,
                 max_retries: int = 3):
        self.base = base
        self.reader = ForesightCacheReader(roots)
        want = [int(x) for x in layers]
        if not set(want).issubset(self.reader.layers):
            raise RuntimeError(f"cache has layers {self.reader.layers}, model wants {want}")
        self._layer_rows = [self.reader.layers.index(L) for L in want]
        self.layers = want
        self.target_type = str(target_type)
        self.offsets = None if offsets is None else [int(value) for value in offsets]
        self.max_retries = max_retries
        self.input_reader: Optional[ForesightCacheReader] = None
        self.input_latent_shape: Optional[tuple[int, int, int, int]] = None
        if self.reader.target_type != self.target_type:
            raise RuntimeError(
                f"cache target_type={self.reader.target_type!r}, "
                f"model wants {self.target_type!r}."
            )
        if self.reader.stored_offsets is not None:
            if self.offsets is None:
                raise RuntimeError(
                    "compact cache has selected offsets but model offsets are unspecified."
                )
            missing_offsets = sorted(set(self.offsets) - set(self.reader.stored_offsets))
            if missing_offsets:
                raise RuntimeError(
                    f"compact cache stores offsets {self.reader.stored_offsets}, "
                    f"model additionally wants {missing_offsets}."
                )
        if len(base) != self.reader.num_samples:
            raise RuntimeError(
                f"dataset length {len(base)} != cache num_samples {self.reader.num_samples}; "
                "cache was built for a different dataset config"
            )
        missing = self.reader.missing_shards()
        if missing:
            raise RuntimeError(f"cache incomplete: {len(missing)} shards missing, first={missing[:5]}")

        if input_latent_roots:
            self.input_reader = ForesightCacheReader(input_latent_roots)
            if self.input_reader.target_type != "input_vae_latent":
                raise RuntimeError(
                    "input-latent cache must have target_type='input_vae_latent', "
                    f"got {self.input_reader.target_type!r}."
                )
            latent_shape = self.input_reader.meta.get("latent_shape")
            if not isinstance(latent_shape, list) or len(latent_shape) != 4:
                raise RuntimeError(
                    "input-latent cache metadata requires latent_shape=[C,T,H,W], "
                    f"got {latent_shape!r}."
                )
            self.input_latent_shape = tuple(int(value) for value in latent_shape)
            if min(self.input_latent_shape) <= 0:
                raise RuntimeError(
                    f"invalid input-latent shape {self.input_latent_shape}."
                )
            record_numel = (
                len(self.input_reader.layers)
                * self.input_reader.seq_len
                * self.input_reader.dim
            )
            expected_numel = int(np.prod(self.input_latent_shape))
            if self.input_reader.layers != [0] or record_numel != expected_numel:
                raise RuntimeError(
                    "input-latent cache record does not match latent_shape: "
                    f"layers={self.input_reader.layers} record_numel={record_numel} "
                    f"latent_shape={self.input_latent_shape}."
                )
            if len(base) != self.input_reader.num_samples:
                raise RuntimeError(
                    f"dataset length {len(base)} != input-latent cache "
                    f"num_samples {self.input_reader.num_samples}."
                )
            input_missing = self.input_reader.missing_shards()
            if input_missing:
                raise RuntimeError(
                    "input-latent cache incomplete: "
                    f"{len(input_missing)} shards missing, first={input_missing[:5]}."
                )

    def __len__(self):
        return len(self.base)

    def _get_consistent(self, idx: int):
        data = self.base._get(idx) if hasattr(self.base, "_get") else self.base[idx]
        data[self.SAMPLE_KEY] = self.reader.get(idx)[self._layer_rows]
        if self.input_reader is not None:
            input_record = self.input_reader.get(idx)
            data[self.INPUT_LATENT_SAMPLE_KEY] = input_record.reshape(
                self.input_latent_shape
            )
        return data

    def __getitem__(self, idx: int):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return self._get_consistent(idx)
            except Exception as e:  # mirror base fallback, but keep data/cache index-consistent
                last_err = e
                logger.warning("cache-wrapped sample %d failed (%s); substituting deterministically", idx, e)
                idx = (idx * 1103515245 + 12345 + attempt) % len(self)
        raise RuntimeError(f"ForesightCacheDataset: {self.max_retries} consecutive failures") from last_err
