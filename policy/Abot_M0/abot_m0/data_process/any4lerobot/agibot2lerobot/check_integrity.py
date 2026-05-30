import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


# ----------------------------
# Raw-side: expected episodes (same as converter)
# ----------------------------
def load_task_info_map(task_info_json: Path) -> Dict[int, dict]:
    data = json.loads(task_info_json.read_text(encoding="utf-8"))
    return {int(ep["episode_id"]): ep for ep in data}


def expected_eids_from_raw(src_path: Path, task_key: str) -> List[int]:
    """
    expected_eids = episodes that exist in observations/<task_id> AND have annotation in task_info/<task_key>.json
    """
    task_id = task_key.split("_")[-1]
    task_info_json = src_path / "task_info" / f"{task_key}.json"
    if not task_info_json.exists():
        return []

    task_info = load_task_info_map(task_info_json)

    obs_dir = src_path / "observations" / task_id
    if not obs_dir.exists():
        return []

    all_eids = []
    for d in obs_dir.iterdir():
        if d.is_dir():
            try:
                all_eids.append(int(d.name))
            except ValueError:
                pass
    all_eids.sort()

    return [eid for eid in all_eids if eid in task_info]


def get_all_task_keys_from_raw(src_path: Path) -> List[str]:
    files = sorted((src_path / "task_info").glob("*.json"))
    return [f.stem for f in files]


# ----------------------------
# Cache mechanism for src_path task info
# ----------------------------
def get_cache_file_path(src_path: Path) -> Path:
    """Generate cache file path based on src_path."""
    # Use absolute path and hash it to create a unique cache filename
    abs_src = src_path.resolve()
    path_str = str(abs_src)
    path_hash = hashlib.md5(path_str.encode("utf-8")).hexdigest()[:12]
    cache_name = f".check_integrity_cache_{path_hash}.json"
    return Path.cwd() / cache_name


def load_cache(cache_file: Path, src_path: Path) -> Optional[Dict]:
    """Load cached task info if it exists and matches src_path."""
    if not cache_file.exists():
        return None
    
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # Verify that cached src_path matches current src_path
        cached_src = Path(data.get("src_path", ""))
        if cached_src.resolve() != src_path.resolve():
            return None
        return data
    except Exception:
        return None


def save_cache(cache_file: Path, src_path: Path, task_keys: List[str], expected_eids_map: Dict[str, List[int]]):
    """Save task info to cache file."""
    data = {
        "src_path": str(src_path.resolve()),
        "task_keys": task_keys,
        "expected_eids_map": expected_eids_map,
    }
    cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_expected_eids_map(src_path: Path, task_keys: List[str]) -> Dict[str, List[int]]:
    """Build expected_eids map for all task_keys."""
    expected_eids_map = {}
    bar = tqdm(task_keys, desc="Building cache", dynamic_ncols=True, leave=False)
    for task_key in bar:
        expected_eids_map[task_key] = expected_eids_from_raw(src_path, task_key)
        bar.set_postfix_str(f"{task_key}: {len(expected_eids_map[task_key])} episodes")
    return expected_eids_map


# ----------------------------
# Output-side: resolve task directory
# ----------------------------
def resolve_task_dir(output_roots: List[Path], task_key: str) -> Optional[Path]:
    """
    Search for task directory in multiple output roots.
    For each root, you may pass:
      A) .../agibot_convert              -> expects agibotworld/task_xxx
      B) .../agibot_convert/agibotworld  -> expects task_xxx directly
    We'll support both.
    """
    for output_root in output_roots:
        # case B: output_root/task_xxx
        cand_b = output_root / task_key
        if cand_b.exists() and cand_b.is_dir():
            return cand_b

        # case A: output_root/agibotworld/task_xxx
        cand_a = output_root / "agibotworld" / task_key
        if cand_a.exists() and cand_a.is_dir():
            return cand_a

    return None


# ----------------------------
# Output-side: read actual total_episodes (prefer meta)
# ----------------------------
def read_total_episodes_from_meta(task_dir: Path) -> Optional[int]:
    """
    Your converted layout:
      task_dir/
        meta/
        data/
        videos/
    We try common meta files under meta/.
    """
    meta_dir = task_dir / "meta"
    if not meta_dir.exists():
        return None

    # Try common jsons first
    candidates = [
        meta_dir / "meta.json",
        meta_dir / "metadata.json",
        meta_dir / "info.json",
        meta_dir / "dataset_info.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                for k in ["total_episodes", "num_episodes"]:
                    if k in obj and isinstance(obj[k], int):
                        return int(obj[k])
            except Exception:
                pass

    # Try jsonl episode index (very common)
    jsonl_candidates = [
        meta_dir / "episodes.jsonl",
        task_dir / "episodes.jsonl",
    ]
    for p in jsonl_candidates:
        if p.exists():
            try:
                n = 0
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for _ in f:
                        n += 1
                return n
            except Exception:
                pass

    return None


def read_total_episodes_with_lerobot(task_dir: Path, repo_id: str) -> Optional[int]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        return None
    try:
        ds = LeRobotDataset(repo_id=repo_id, root=task_dir)  # type: ignore
        meta = getattr(ds, "meta", None)
        if meta is not None and hasattr(meta, "total_episodes"):
            return int(meta.total_episodes)
    except Exception:
        return None
    return None


def read_actual_total_episodes(task_dir: Path, repo_id: str) -> Optional[int]:
    # Prefer reading from meta folder (no dependency)
    v = read_total_episodes_from_meta(task_dir)
    if v is not None:
        return v
    # Fallback to lerobot if meta json not found
    return read_total_episodes_with_lerobot(task_dir, repo_id)


# ----------------------------
# Check + report
# ----------------------------
@dataclass
class TaskCheck:
    task_key: str
    expected: int
    actual: Optional[int]
    ok: bool
    reason: str
    task_dir: Optional[str]


def check_one(output_roots: List[Path], task_key: str, expected: int, strict: bool) -> TaskCheck:

    task_dir = resolve_task_dir(output_roots, task_key)
    if task_dir is None:
        return TaskCheck(task_key, expected, None, False, "converted dir not found (path root mismatch?)", None)

    actual = read_actual_total_episodes(task_dir, repo_id=task_key)
    if actual is None:
        return TaskCheck(task_key, expected, None, False, "cannot read total_episodes from meta/ or lerobot", str(task_dir))

    if strict:
        ok = (actual == expected)
        reason = "ok" if ok else "strict mismatch"
    else:
        ok = (0 <= actual <= expected)
        reason = "ok" if ok else "non-strict mismatch"

    return TaskCheck(task_key, expected, actual, ok, reason, str(task_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True, help="raw root: contains task_info/ and observations/")
    parser.add_argument("--output-paths", type=Path, nargs="+", required=True, help="converted root(s) or converted/agibotworld (supports multiple paths)")
    parser.add_argument("--task-ids", type=str, nargs="+", default=[], help="optional: task_327 task_366 ...")
    parser.add_argument("--non-strict", action="store_true", help="allow actual <= expected")
    parser.add_argument("--save-failed", type=Path, default=None, help="optional: write failed task keys to a file")
    parser.add_argument("--no-cache", action="store_true", help="disable cache and rebuild from scratch")
    args = parser.parse_args()

    strict = not args.non_strict
    output_roots = args.output_paths

    # Try to load cache
    cache_file = get_cache_file_path(args.src_path)
    expected_eids_map = None
    task_keys = None
    
    if not args.no_cache:
        cached_data = load_cache(cache_file, args.src_path)
        if cached_data is not None:
            print(f"[Cache loaded] {cache_file}")
            task_keys = cached_data["task_keys"]
            expected_eids_map = {k: v for k, v in cached_data["expected_eids_map"].items()}
    
    # If cache not available, build from scratch
    if task_keys is None or expected_eids_map is None:
        print(f"[Building cache] Scanning {args.src_path}...")
        task_keys = get_all_task_keys_from_raw(args.src_path)
        expected_eids_map = build_expected_eids_map(args.src_path, task_keys)
        save_cache(cache_file, args.src_path, task_keys, expected_eids_map)
        print(f"[Cache saved] {cache_file}")
    
    # Filter task_keys if --task-ids specified
    if args.task_ids:
        wanted = set(args.task_ids)
        task_keys = [k for k in task_keys if k in wanted]

    results: List[TaskCheck] = []
    failed: List[TaskCheck] = []

    bar = tqdm(task_keys, desc="Integrity check", dynamic_ncols=True)
    for task_key in bar:
        expected = len(expected_eids_map.get(task_key, []))
        r = check_one(output_roots, task_key, expected, strict=strict)
        results.append(r)
        if not r.ok:
            failed.append(r)

        if r.ok:
            bar.set_postfix_str(f"{task_key}: ok (exp={r.expected}, act={r.actual})")
        else:
            bar.set_postfix_str(f"{task_key}: FAIL ({r.reason})")

    # report
    print("\n" + "-" * 90)
    print(f"{'task':<12} {'expected':>9} {'actual':>9} {'PASS':>6}  reason")
    print("-" * 90)
    for r in results:
        a = "None" if r.actual is None else str(r.actual)
        p = "PASS" if r.ok else "FAIL"
        print(f"{r.task_key:<12} {r.expected:>9} {a:>9} {p:>6}  {r.reason}")
    print("-" * 90)
    print(f"Total tasks: {len(results)} | Failed: {len(failed)}")

    if failed:
        failed_keys = [r.task_key for r in failed]
        print("\n[INCOMPLETE TASKS]")
        for r in failed:
            print(f" - {r.task_key} | exp={r.expected} act={r.actual} | dir={r.task_dir} | {r.reason}")

        if args.save_failed is not None:
            args.save_failed.write_text("\n".join(failed_keys) + "\n", encoding="utf-8")
            print(f"\n[saved failed list] {args.save_failed}")

        raise RuntimeError(f"Incomplete tasks ({len(failed_keys)}): {failed_keys}")


if __name__ == "__main__":
    main()