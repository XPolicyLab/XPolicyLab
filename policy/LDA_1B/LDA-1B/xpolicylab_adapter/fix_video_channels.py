"""Fix the R/B channel swap in produced LeRobot mp4s — parallel, with mtime resume.

The old process_data.py applied an extra cv2.COLOR_BGR2RGB to frames that were
already RGB, so every produced video has its red/blue channels swapped (wooden
table renders blue). This rewrites each mp4 in place with R/B swapped, making the
colors match data/<dataset>/<task>/<env>/preview_video/*.mp4.

The swap is its own inverse, and a swapped file is indistinguishable from an
unswapped one — so resuming an interrupted run can't rely on a marker. Instead we
use mtime: the interrupted run rewrote the files it already fixed, giving them a
NEWER mtime than the not-yet-fixed files (which keep their original generation
time). We therefore process only files OLDER than a cutoff and skip the newer
(already-fixed) ones.

  - --cutoff omitted: cutoff auto-detected as the midpoint of the largest gap in
    the sorted mtimes (the generation -> interrupted-run boundary).
  - --cutoff "<ISO time>" or "<epoch>": files with mtime >= cutoff are treated as
    already done and skipped.

Always run --dry-run first: it prints how many videos WOULD be changed and the
files right at the cutoff boundary, so you can spot-check "nearby" videos.

preview_video/ directories are skipped automatically (they are the color ground
truth and must never be swapped).

Usage:
    python fix_video_channels.py <DIR> --dry-run
    python fix_video_channels.py <DIR> --workers 16
    python fix_video_channels.py <DIR> --cutoff "2026-05-21 14:00:00"
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm

DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "data"
VIDEO_CODEC = "mp4v"  # matches process_data.py / meta/info.json


def _collect_mp4s(targets: List[Path]) -> List[Path]:
    found: List[Path] = []
    for t in targets:
        if t.is_file() and t.suffix == ".mp4":
            found.append(t)
        elif t.is_dir():
            found.extend(sorted(t.rglob("*.mp4")))
        else:
            raise FileNotFoundError(f"Target not found: {t}")
    # Never touch ground-truth preview videos; skip leftover mkstemp temp files
    # (real LeRobot videos are episode_*.mp4, so a "tmp" prefix is safe to exclude).
    return [p for p in found
            if "preview_video" not in p.parts and not p.name.startswith("tmp")]


def _parse_cutoff(text: str) -> float:
    try:
        return float(text)  # epoch seconds
    except ValueError:
        return dt.datetime.fromisoformat(text).timestamp()


def _auto_cutoff(mtimes: List[float]) -> Optional[float]:
    """Midpoint of the largest gap in sorted mtimes, or None if <2 distinct values."""
    uniq = sorted(set(mtimes))
    if len(uniq) < 2:
        return None
    gaps = [(uniq[i + 1] - uniq[i], uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]
    _, lo, hi = max(gaps, key=lambda g: g[0])
    return (lo + hi) / 2.0


def _swap_in_place(path_str: str) -> Tuple[str, int, Optional[str]]:
    """Rewrite `path` with R/B swapped on every frame. Returns (path, frames, error)."""
    path = Path(path_str)
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return (path_str, 0, "cannot open")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (w, h))
        if not writer.isOpened():
            tmp.unlink(missing_ok=True)
            return (path_str, 0, "VideoWriter failed to open")

        n = 0
        try:
            while True:
                ok, frame = cap.read()  # BGR
                if not ok:
                    break
                writer.write(frame[:, :, ::-1])  # swap R<->B, then write
                n += 1
        finally:
            cap.release()
            writer.release()

        if n == 0:
            tmp.unlink(missing_ok=True)
            return (path_str, 0, "no frames decoded")
        os.replace(tmp, path)  # atomic within the same directory
        return (path_str, n, None)
    except Exception as e:  # keep the pool alive; report per-file
        return (path_str, 0, repr(e))


def _fmt(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", default=[str(DEFAULT_TARGET)],
                        help="mp4 files or dirs (recursive). Default: LDA_1B/data")
    parser.add_argument("--cutoff", default=None,
                        help="mtime cutoff (ISO time or epoch); files >= cutoff are "
                             "treated as already fixed and skipped. Default: auto-detect.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="parallel worker processes (default: CPU count)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the count to change and the boundary, do not modify")
    args = parser.parse_args()

    mp4s = _collect_mp4s([Path(t).resolve() for t in args.targets])
    if not mp4s:
        print("No .mp4 files found under the given targets.")
        return

    mtimes = {p: p.stat().st_mtime for p in mp4s}
    cutoff = _parse_cutoff(args.cutoff) if args.cutoff else _auto_cutoff(list(mtimes.values()))

    if cutoff is None:
        to_process = sorted(mp4s, key=lambda p: mtimes[p])
        skipped: List[Path] = []
        print("[warn] could not determine a cutoff (mtimes not bimodal); "
              "treating ALL files as not-yet-fixed.")
    else:
        to_process = sorted((p for p in mp4s if mtimes[p] < cutoff), key=lambda p: mtimes[p])
        skipped = sorted((p for p in mp4s if mtimes[p] >= cutoff), key=lambda p: mtimes[p])

    print(f"found {len(mp4s)} mp4(s); cutoff = "
          f"{_fmt(cutoff) if cutoff else 'n/a'}")
    print(f"  already-fixed (skip, mtime >= cutoff): {len(skipped)}")
    print(f"  TO CHANGE     (process, mtime < cutoff): {len(to_process)}")

    # Show the boundary so the user can spot-check "nearby" videos.
    if cutoff is not None and to_process and skipped:
        print("  --- boundary (around cutoff) ---")
        for p in to_process[-3:]:
            print(f"    process | {_fmt(mtimes[p])} | {p}")
        for p in skipped[:3]:
            print(f"    skip    | {_fmt(mtimes[p])} | {p}")

    if args.dry_run:
        print("[dry-run] nothing modified.")
        return
    if not to_process:
        print("nothing to do.")
        return

    workers = max(1, min(args.workers, len(to_process)))
    total_frames = 0
    errors: List[Tuple[str, str]] = []
    with Pool(processes=workers) as pool:
        for path_str, n, err in tqdm(
            pool.imap_unordered(_swap_in_place, [str(p) for p in to_process]),
            total=len(to_process), desc=f"swap R/B (x{workers})", unit="vid",
            dynamic_ncols=True,
        ):
            if err:
                errors.append((path_str, err))
            else:
                total_frames += n

    print(f"[fix] swapped R/B in {len(to_process) - len(errors)} videos / {total_frames} frames "
          f"using {workers} workers.")
    if errors:
        print(f"[fix] {len(errors)} file(s) failed:")
        for path_str, err in errors[:20]:
            print(f"    {err}  <-  {path_str}")


if __name__ == "__main__":
    main()
