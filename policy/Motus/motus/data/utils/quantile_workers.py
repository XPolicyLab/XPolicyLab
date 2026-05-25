import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


# Ensure single-threaded kernels inside workers
torch.set_num_threads(1)


def _load_latent_tensor(pt_path: Path, key: str) -> Optional[torch.Tensor]:
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None
    if isinstance(data, dict):
        t = data.get(key)
        if isinstance(t, torch.Tensor):
            return t.float()
        return None
    if isinstance(data, torch.Tensor):
        return data.float()
    return None


def _to_2d(t: torch.Tensor) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() >= 2:
        return t.view(-1, t.shape[-1])
    return None


def minmax_worker(args: Tuple[List[str], str]) -> Tuple[np.ndarray, np.ndarray, int]:
    files, key = args
    cur_min: Optional[np.ndarray] = None
    cur_max: Optional[np.ndarray] = None
    used = 0
    for f in files:
        t = _load_latent_tensor(Path(f), key)
        if t is None:
            continue
        t2 = _to_2d(t)
        if t2 is None or t2.numel() == 0:
            continue
        x = t2.cpu().numpy()
        mn = x.min(axis=0)
        mx = x.max(axis=0)
        if cur_min is None:
            cur_min = mn
            cur_max = mx
        else:
            cur_min = np.minimum(cur_min, mn)
            cur_max = np.maximum(cur_max, mx)
        used += 1
    if cur_min is None:
        return np.array([]), np.array([]), 0
    return cur_min.astype(np.float64), cur_max.astype(np.float64), used


def hist_worker(args: Tuple[List[str], str, np.ndarray, np.ndarray, int]) -> Tuple[str, int]:
    files, key, gmin, gmax, num_bins = args
    D = int(gmin.shape[0])
    hist = np.zeros((D, num_bins), dtype=np.int64)
    ranges = np.maximum(gmax - gmin, 1e-12)
    scale = (num_bins - 1) / ranges
    total_rows = 0
    for f in files:
        t = _load_latent_tensor(Path(f), key)
        if t is None:
            continue
        t2 = _to_2d(t)
        if t2 is None or t2.numel() == 0:
            continue
        x = t2.cpu().numpy()
        idx = np.floor((x - gmin) * scale).astype(np.int64)
        np.clip(idx, 0, num_bins - 1, out=idx)
        block = 64
        for start in range(0, D, block):
            end = min(start + block, D)
            for j in range(start, end):
                counts = np.bincount(idx[:, j], minlength=num_bins)
                hist[j] += counts
        total_rows += x.shape[0]

    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"latent_hist_{os.getpid()}_{np.random.randint(1_000_000_000)}.npy"
    np.save(out_path, hist, allow_pickle=False)
    return str(out_path), total_rows


