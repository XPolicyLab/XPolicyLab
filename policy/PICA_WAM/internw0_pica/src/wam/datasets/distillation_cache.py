"""Portable row-aligned windows and offline 4D teacher features."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

FEATURE_DIM = 1430
CONTRACT = {
    "feature_api_version": "track4world_da3_v1",
    "blocks": [
        ["geometry", 1024, "mean+L2"],
        ["motion_2d", 128, "final_refinement_mean+L2"],
        ["motion_3d", 256, "final_refinement_mean+L2"],
        ["camera", 18, "relative_pose_stats+L2"],
        ["visibility", 4, "valid_visible_confidence_stats+L2"],
    ],
}
POOLING_SHA256 = hashlib.sha256(
    json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
EXTRACTION = {
    "teacher_model": "TencentARC/Track4World",
    "pooling_sha256": POOLING_SHA256,
    "feature_dim": FEATURE_DIM,
    "horizon": 32,
    "num_frames": 33,
    "canvas_size": [384, 256],
    "image_size": 256,
    "teacher_model_seqlen": 32,
    "motion_chunk_size": 32,
    "iters": 4,
    "inference_batch_size": 1,
    "pixel_dtype": "uint8",
    "teacher_seed": 42,
    "compile_motion": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def dataset_id(spec) -> str:
    """Use logical source/name, independently of data and artifact roots."""
    value = f"{spec.source_name}/{spec.name}"
    parts = value.split("/")
    if not spec.source_name or any(p in {"", ".", ".."} for p in parts) or ":" in value or "\\" in value:
        raise ValueError(f"Invalid dataset identity: {value!r}")
    return value


class WindowPlan:
    """Memory-map [dataset_id ordinal, local episode, local start] uint32 rows."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "windows.json").read_text())
        if self.manifest.get("format") != "internw0_4d_windows_v1":
            raise ValueError("Unsupported 4D window plan.")
        if self.manifest.get("extraction") != EXTRACTION:
            raise ValueError("4D window extraction contract mismatch.")
        self.datasets = self.manifest["datasets"]
        if not self.datasets or len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Window dataset identities must be nonempty and unique.")
        for value in self.datasets:
            if value.startswith("/") or ":" in value or "\\" in value or any(p in {"", ".", ".."} for p in value.split("/")):
                raise ValueError("Window identities must be relative source/name identifiers.")
        self._rows = None
        self._pid = None
        rows = self.rows
        if rows.dtype != np.dtype("<u4") or rows.shape != (len(self), 3) or len(self) <= 0:
            raise ValueError("Invalid windows.npy shape or dtype.")

    def __len__(self):
        return int(self.manifest["sample_count"])

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def rows(self):
        if self._rows is None or self._pid != os.getpid():
            self._rows = np.load(self.root / "windows.npy", mmap_mode="r", allow_pickle=False)
            self._pid = os.getpid()
        return self._rows

    def identity(self, row):
        if not 0 <= int(row) < len(self):
            raise IndexError(row)
        source, episode, start = map(int, self.rows[int(row)])
        return self.datasets[source], episode, start

    def __getstate__(self):
        return {**self.__dict__, "_rows": None, "_pid": None}


class TeacherCache(WindowPlan):
    """Read complete, contiguous fp16 feature shards without SQLite lookups."""

    def __init__(self, root):
        super().__init__(root)
        manifest = json.loads((self.root / "features.json").read_text())
        if manifest.get("format") != "internw0_4d_features_v1" or manifest.get("windows_sha256") != self.fingerprint:
            raise ValueError("Teacher cache does not match the window plan.")
        self.shards = manifest["shards"]
        cursor = 0
        for shard in self.shards:
            start, end = int(shard["start"]), int(shard["end"])
            path = Path(shard["file"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Feature paths must stay inside the cache directory.")
            if start != cursor or end <= start or end > len(self):
                raise ValueError("Teacher shards must cover the plan once, in row order.")
            if (self.root / path).stat().st_size != (end - start) * FEATURE_DIM * 2:
                raise ValueError(f"Teacher feature size mismatch: {path}")
            cursor = end
        if cursor != len(self):
            raise ValueError("Teacher feature cache is incomplete.")
        self._ends = [int(s["end"]) for s in self.shards]
        self._features = {}
        self._features_pid = None

    def feature(self, row):
        if not 0 <= int(row) < len(self):
            raise IndexError(row)
        if self._features_pid != os.getpid():
            self._features = {}
            self._features_pid = os.getpid()
        pos = bisect.bisect_right(self._ends, int(row))
        shard = self.shards[pos]
        if pos not in self._features:
            # Bound open mappings in long runs with many small shards.
            if len(self._features) >= 8:
                self._features.pop(next(iter(self._features)))
            self._features[pos] = np.memmap(self.root / shard["file"], mode="r", dtype="<f2", shape=(shard["end"] - shard["start"], FEATURE_DIM))
        value = np.array(self._features[pos][int(row) - shard["start"]], copy=True)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite teacher feature at row {row}.")
        return torch.from_numpy(value).unsqueeze(0)

    def __getstate__(self):
        return {**super().__getstate__(), "_features": {}, "_features_pid": None}
