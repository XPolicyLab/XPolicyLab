#!/usr/bin/env python
"""Migrate old-schema FastWAM LeRobot datasets to the simplified task_index-only
schema.

Old schema (produced by the previous process_data.py revision):
  - meta/info.json `features` had: coarse_task_index, coarse_quality_index, quality_index
  - meta/info.json `total_tasks` = unique_instructions + 2
  - meta/info.json `chunks_size` = 1000 (hard-coded), but every episode was actually
        written into chunk-000/, so for datasets with >1000 episodes the LeRobot
        loader's `ep_idx // chunks_size` chunk routing pointed at non-existent
        chunk-001/, chunk-002/, ... directories and fell back to
        snapshot_download(repo_id=<local path>) -> HFValidationError.
  - meta/tasks.jsonl had 2 trailing dead rows: "xpolicylab_quality", "success"
  - data/chunk-000/episode_*.parquet had columns:
        coarse_task_index, coarse_quality_index, quality_index

New schema (after fix):
  - Only `task_index` is kept (in features, tasks.jsonl, parquet).
  - tasks.jsonl contains only unique instructions.
  - chunks_size = max(total_episodes, 1) so every episode lives in chunk-000/.

The fix is purely a schema cleanup; the actual training labels never change
because the legacy slots were dead in RoboTwin's default drop_high_level_prob=1.0
training path. The chunks_size correction IS required for correctness when the
dataset has more than 1000 episodes (otherwise training crashes at dataset
load time with HFValidationError).

This script is *idempotent*: it detects whether a dataset is already on the new
schema and skips it. It writes parquet/jsonl atomically via tmp + rename.

Usage:
  # default: scan <policy>/data/*/lerobot
  python migrate_legacy_task_slots.py

  # explicit roots (each must be a `<dataset_id>/lerobot` dir or its parent)
  python migrate_legacy_task_slots.py path/to/lerobot path/to/dataset_id ...

  # dry run (no writes), default off
  DRY_RUN=1 python migrate_legacy_task_slots.py

Run inside the FastWAM policy conda env (`conda activate fastwam`).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd

LEGACY_FEATURE_KEYS = ("coarse_task_index", "coarse_quality_index", "quality_index")
LEGACY_TRAILING_TASKS = ("xpolicylab_quality", "success")

DRY_RUN = os.environ.get("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "y"}
POLICY_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = POLICY_DIR / "data"


def _log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if DRY_RUN:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    if DRY_RUN:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def discover_lerobot_dirs(args: List[str]) -> List[Path]:
    """Resolve user-given paths to a list of `<dataset_id>/lerobot` directories."""
    candidates: List[Path] = []
    if not args:
        if not DEFAULT_DATA_ROOT.exists():
            return []
        for sub in sorted(DEFAULT_DATA_ROOT.iterdir()):
            cand = sub / "lerobot"
            if cand.is_dir():
                candidates.append(cand)
        return candidates

    for raw in args:
        p = Path(raw).expanduser().resolve()
        if (p / "meta" / "info.json").is_file():
            candidates.append(p)
            continue
        cand = p / "lerobot"
        if (cand / "meta" / "info.json").is_file():
            candidates.append(cand)
            continue
        _log("WARN", f"skip {p}: no meta/info.json found (or {cand}/meta/info.json)")
    return candidates


def migrate_info_json(lerobot_dir: Path) -> Tuple[bool, int]:
    """Returns (changed, new_total_tasks_or_-1)."""
    info_path = lerobot_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    legacy_present = [k for k in LEGACY_FEATURE_KEYS if k in features]
    new_total_tasks = -1
    changed = False
    chunks_fix = None

    if legacy_present:
        for k in legacy_present:
            features.pop(k, None)
        info["features"] = features
        changed = True

    # Recompute total_tasks from tasks.jsonl (more reliable than trusting the old value).
    tasks_path = lerobot_dir / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        with tasks_path.open("r", encoding="utf-8") as f:
            n_lines = sum(1 for line in f if line.strip())
        # tasks.jsonl will be trimmed by migrate_tasks_jsonl too; recompute the
        # expected post-trim count here so that info.json matches tasks.jsonl after
        # both migrations finish.
        trailing_to_trim = 0
        lines = [l for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for legacy_text in reversed(LEGACY_TRAILING_TASKS):
            if not lines:
                break
            try:
                rec = json.loads(lines[-1])
            except json.JSONDecodeError:
                break
            if rec.get("task") == legacy_text:
                lines.pop()
                trailing_to_trim += 1
            else:
                break
        new_total_tasks = n_lines - trailing_to_trim
        if info.get("total_tasks") != new_total_tasks:
            info["total_tasks"] = new_total_tasks
            changed = True

    # Fix chunks_size / total_chunks if the old writer left chunks_size=1000 with
    # total_episodes >1000 (all episodes are physically in chunk-000/, but the
    # loader uses ep_idx // chunks_size to derive the chunk dir and breaks).
    total_episodes = int(info.get("total_episodes", 0))
    actual_chunks_dirs = sorted(p.name for p in (lerobot_dir / "data").iterdir() if p.is_dir()) if (lerobot_dir / "data").is_dir() else []
    only_chunk_000 = actual_chunks_dirs == ["chunk-000"]
    chunks_size = int(info.get("chunks_size", 0))
    if only_chunk_000 and total_episodes > chunks_size and chunks_size > 0:
        new_chunks_size = max(chunks_size, total_episodes)
        new_total_chunks = 1
        info["chunks_size"] = new_chunks_size
        info["total_chunks"] = new_total_chunks
        chunks_fix = (chunks_size, new_chunks_size)
        changed = True

    if changed:
        _atomic_write_bytes(info_path, json.dumps(info, indent=2, ensure_ascii=False).encode("utf-8"))
        msg = f"updated {info_path}: dropped {legacy_present}, total_tasks={new_total_tasks}"
        if chunks_fix is not None:
            msg += f", chunks_size={chunks_fix[0]}->{chunks_fix[1]} (all episodes in chunk-000)"
        _log("info.json", msg)
    return changed, new_total_tasks


def migrate_tasks_jsonl(lerobot_dir: Path) -> bool:
    tasks_path = lerobot_dir / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return False
    raw_lines = [l for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = []
    for ln in raw_lines:
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError as exc:
            _log("WARN", f"tasks.jsonl line not valid JSON, kept as-is: {exc}")
            records.append({"_raw": ln})

    trimmed = 0
    for legacy_text in reversed(LEGACY_TRAILING_TASKS):
        if not records:
            break
        last = records[-1]
        if isinstance(last, dict) and last.get("task") == legacy_text:
            records.pop()
            trimmed += 1
        else:
            break

    if not trimmed:
        return False

    # Re-emit jsonl, preserving original "task_index" values (they're stable for
    # the surviving real instruction entries).
    out = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    _atomic_write_bytes(tasks_path, out.encode("utf-8"))
    _log("tasks.jsonl", f"trimmed {trimmed} legacy entries from {tasks_path}")
    return True


def migrate_parquet_files(lerobot_dir: Path) -> int:
    data_dir = lerobot_dir / "data"
    if not data_dir.is_dir():
        return 0
    updated = 0
    for parquet_path in sorted(data_dir.rglob("episode_*.parquet")):
        df = pd.read_parquet(parquet_path)
        legacy = [c for c in LEGACY_FEATURE_KEYS if c in df.columns]
        if not legacy:
            continue
        df = df.drop(columns=legacy)
        _atomic_write_parquet(parquet_path, df)
        updated += 1
        _log("parquet", f"dropped {legacy} from {parquet_path.relative_to(lerobot_dir)}")
    return updated


def migrate_lerobot_dir(lerobot_dir: Path) -> None:
    _log("scan", f"{lerobot_dir}")
    changed_info, _ = migrate_info_json(lerobot_dir)
    changed_tasks = migrate_tasks_jsonl(lerobot_dir)
    n_parquet = migrate_parquet_files(lerobot_dir)
    if not (changed_info or changed_tasks or n_parquet):
        _log("ok", f"already on new schema: {lerobot_dir}")
        return
    _log(
        "done",
        f"{lerobot_dir}: info_changed={changed_info} tasks_trimmed={changed_tasks} parquet_updated={n_parquet}",
    )


def main(argv: List[str]) -> int:
    if shutil.which("python") is None:
        # nonsense check kept to fail fast in weird PATHs
        _log("error", "python missing on PATH")
        return 2

    targets = discover_lerobot_dirs(argv[1:])
    if not targets:
        _log("error", "no <dataset_id>/lerobot/meta/info.json found; pass paths explicitly")
        return 1

    if DRY_RUN:
        _log("dry-run", "DRY_RUN=1 set; no files will be modified")
    for lerobot_dir in targets:
        try:
            migrate_lerobot_dir(lerobot_dir)
        except Exception as exc:  # noqa: BLE001
            _log("error", f"failed on {lerobot_dir}: {exc}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
