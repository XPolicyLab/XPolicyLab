import argparse
import json
import math
import os
import multiprocessing as mp
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
try:
    # When run as a package module
    from .quantile_workers import minmax_worker, hist_worker  # type: ignore
except Exception:
    # When run as a script: add current dir to sys.path and import
    import sys
    sys.path.append(str(Path(__file__).parent))
    from quantile_workers import minmax_worker, hist_worker  # type: ignore

# Reduce intra-op threading to avoid oversubscription when using many processes
import torch
torch.set_num_threads(1)


def _iter_pt_files(root: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _collect_files_from_cfg(cfg_path: Path, latent_dir_name: str, pattern: str) -> List[str]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --cfg mode. Please install pyyaml.") from e

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_dirs: List[str] = cfg["dataset"]["dataset_dir"]
    files: List[str] = []
    for root in dataset_dirs:
        root_path = Path(root)
        # find all subdirectories named latent_dir_name
        for d in root_path.rglob(latent_dir_name):
            if d.is_dir():
                for p in d.rglob(pattern):
                    if p.is_file():
                        files.append(str(p))
    files.sort()
    return files


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


"""
Worker functions moved to a separate module (quantile_workers.py) to be pickleable
under spawn start method. This avoids AttributeError: Can't get attribute ... on __mp_main__.
"""


def _chunk_list(items: List[str], n_chunks: int) -> List[List[str]]:
    n = len(items)
    if n_chunks <= 1:
        return [items]
    size = math.ceil(n / n_chunks)
    return [items[i : i + size] for i in range(0, n, size)]


def compute_quantiles_mp(
    input_dir: Path,
    *,
    key: str,
    pattern: str,
    recursive: bool,
    num_workers: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    # list files
    files = [str(p) for p in _iter_pt_files(input_dir, recursive, pattern)]
    files.sort()
    if not files:
        raise RuntimeError(f"No files matched under {input_dir} with pattern {pattern}")
    return _compute_from_files(files, key=key, num_workers=num_workers, num_bins=num_bins)


def _compute_from_files(
    files: List[str],
    *,
    key: str,
    num_workers: int,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    if not files:
        raise RuntimeError("No files to process")

    # pass 1: global min/max (parallel)
    # Use more chunks than workers to improve task scheduling
    chunks = _chunk_list(files, max(num_workers * 8, 1))
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    used_files = 0
    ctx = get_context('spawn')
    with ctx.Pool(processes=num_workers, maxtasksperchild=32) as pool:
        for mn, mx, used in tqdm(
            pool.imap_unordered(minmax_worker, [(ch, key) for ch in chunks], chunksize=1),
            total=len(chunks),
            desc="Pass1 min/max",
        ):
            if used == 0:
                continue
            mins.append(mn)
            maxs.append(mx)
            used_files += used

    if not mins:
        raise RuntimeError("No usable tensors for min/max")
    gmin = np.minimum.reduce(mins)
    gmax = np.maximum.reduce(maxs)
    D = int(gmin.shape[0])

    # pass 2: hist accumulation (parallel -> temp files)
    hist_paths: List[str] = []
    total_rows = 0
    with ctx.Pool(processes=num_workers, maxtasksperchild=32) as pool:
        for path, rows in tqdm(
            pool.imap_unordered(hist_worker, [(ch, key, gmin, gmax, num_bins) for ch in chunks], chunksize=1),
            total=len(chunks),
            desc="Pass2 hist",
        ):
            hist_paths.append(path)
            total_rows += rows

    # reduce temp hists
    final_hist = np.zeros((D, num_bins), dtype=np.int64)
    for hp in tqdm(hist_paths, desc="Reduce"):
        h = np.load(hp, allow_pickle=False)
        if h.shape != final_hist.shape:
            continue
        final_hist += h
        try:
            Path(hp).unlink(missing_ok=True)
        except Exception:
            pass

    # compute quantiles from hist
    # edges per dim
    eps = 1e-12
    edges = [np.linspace(gmin[i] - eps, gmax[i] + eps, num_bins + 1) for i in range(D)]
    q01 = np.empty(D, dtype=np.float32)
    q99 = np.empty(D, dtype=np.float32)
    tgt01 = 0.01 * total_rows
    tgt99 = 0.99 * total_rows
    for i in range(D):
        cumsum = np.cumsum(final_hist[i])
        idx01 = int(np.searchsorted(cumsum, tgt01))
        idx99 = int(np.searchsorted(cumsum, tgt99))
        idx01 = max(0, min(idx01, num_bins - 1))
        idx99 = max(0, min(idx99, num_bins - 1))
        q01[i] = edges[i][idx01]
        q99[i] = edges[i][idx99]

    return q01, q99, used_files, total_rows, D


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast multi-process quantile stats (q01/q99) for latent_action")
    parser.add_argument("--input-dir", type=str, required=False, help="Root directory to scan")
    parser.add_argument("--cfg", type=str, required=False, help="latent_action.yaml; process only latent_action subdirs")
    parser.add_argument("--latent-dir-name", type=str, default="latent_action", help="Subdir name to include in --cfg mode")
    parser.add_argument("--output", type=str, default="latent_action_q01_q99.json", help="Output JSON path")
    parser.add_argument("--key", type=str, default="latent_action", help="Key for dict .pt files")
    parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive scan")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Parallel workers (default: CPU count)")
    parser.add_argument("--num-bins", type=int, default=4096, help="Histogram bins per dim (default 4096)")

    args = parser.parse_args()
    recursive = not args.no_recursive

    # Determine file list
    files: Optional[List[str]] = None
    input_dir_str: Optional[str] = None
    if args.cfg:
        files = _collect_files_from_cfg(Path(args.cfg), args.latent_dir_name, args.pattern)
        if not files:
            raise RuntimeError("No .pt files found under any latent_action subdir from cfg")
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        files = [str(p) for p in _iter_pt_files(input_dir, recursive, args.pattern)]
        input_dir_str = str(input_dir.resolve())
    else:
        raise RuntimeError("Either --cfg or --input-dir must be provided")

    q01, q99, used_files, total_rows, D = _compute_from_files(
        files,
        key=args.key,
        num_workers=max(1, args.num_workers),
        num_bins=max(64, args.num_bins),
    )

    out = {
        "latent_action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "num_files_used": used_files,
            "num_samples": total_rows,
            "latent_dim": int(D),
            "input_dir": input_dir_str,
            "cfg": args.cfg,
            "pattern": args.pattern,
            "recursive": recursive,
            "key": args.key,
            "num_bins": int(max(64, args.num_bins)),
            "num_workers": int(max(1, args.num_workers)),
            "latent_dir_name": args.latent_dir_name,
        }
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"Saved: {out_path} | files={used_files} rows={total_rows} dim={D}")


if __name__ == "__main__":
    main()


