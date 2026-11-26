"""Results store: one JSONL line per (run id, manifest amendment); readers see the newest amendment of each run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from knowin_agent.experiments.manifest import MANIFEST, SCHEMA, manifest_paths, read_manifest

STORE = "results.jsonl"

ManifestLike = Union[Mapping[str, Any], str, Path]


def run_id(record: Mapping[str, Any]) -> str:
    return record["identity"]["run_id"]


def amendment(record: Mapping[str, Any]) -> int:
    return int((record.get("amendment") or {}).get("number", 0))


def _key(record: Mapping[str, Any]) -> tuple[str, int]:
    return run_id(record), amendment(record)


def _as_manifest(item: ManifestLike) -> dict:
    if isinstance(item, Mapping):
        record = dict(item)
    else:
        path = Path(item)
        record = read_manifest(path) if path.is_dir() else json.loads(path.read_text(encoding="utf-8"))
        record.setdefault("_source", str(path))
    if record.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} manifest: {item!r:.120}")
    return record


def _read_lines(store: Path) -> list[dict]:
    store = Path(store)
    if not store.is_file():
        return []
    return [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(store: Union[str, Path], manifests: Union[ManifestLike, Iterable[ManifestLike]]) -> int:
    """Append manifests (dicts, manifest files or run dirs); a (run id, amendment) already stored is skipped."""
    if isinstance(manifests, (Mapping, str, Path)):
        manifests = [manifests]
    store = Path(store)
    seen = {_key(r) for r in _read_lines(store)}
    added = []
    for item in manifests:
        record = _as_manifest(item)
        if _key(record) in seen:
            continue
        seen.add(_key(record))
        added.append(record)
    if added:
        store.parent.mkdir(parents=True, exist_ok=True)
        with open(store, "a", encoding="utf-8") as handle:
            for record in added:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(added)


def ingest(store: Union[str, Path], runs_root: Union[str, Path]) -> int:
    """Append every run directory under ``runs_root`` that has a manifest (newest amendment of each)."""
    runs = sorted({p.parent for p in Path(runs_root).rglob("manifest*.json")
                   if p.name == MANIFEST or p.name.startswith("manifest-")})
    return append(store, [run for run in runs if manifest_paths(run)])


def load(store: Union[str, Path], *, latest: bool = True) -> list[dict]:
    """Stored records in first-seen order; with ``latest`` one record per run id (its newest amendment)."""
    records = _read_lines(Path(store))
    if not latest:
        return records
    newest: dict[str, dict] = {}
    for record in records:
        current = newest.get(run_id(record))
        if current is None or amendment(record) > amendment(current):
            newest[run_id(record)] = record
    order = list(dict.fromkeys(run_id(r) for r in records))
    return [newest[r] for r in order]


def _matches(value: Any, wanted: Any) -> bool:
    if wanted is None:
        return True
    if isinstance(wanted, (str, int)) or wanted is True or wanted is False:
        return value == wanted
    return value in set(wanted)


def query(records: Iterable[Mapping[str, Any]], *, profile=None, task=None, phase=None, split=None,
          pipeline_id=None, round=None, compliance=None, include_excluded: bool = True) -> list[dict]:
    """Filter records; each criterion is a value or a collection of accepted values (None = any)."""
    out = []
    for record in records:
        identity = record.get("identity", {})
        scene = record.get("scene") or {}
        if not (_matches(record.get("profile"), profile) and _matches(identity.get("task"), task)
                and _matches(identity.get("phase"), phase) and _matches(scene.get("split"), split)
                and _matches(identity.get("pipeline_id"), pipeline_id) and _matches(identity.get("round"), round)
                and _matches((record.get("compliance") or {}).get("level"), compliance)):
            continue
        if not include_excluded and (record.get("results") or {}).get("excluded"):
            continue
        out.append(dict(record))
    return out


def profiles(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r.get("profile")) for r in records})


def tasks(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({r["identity"]["task"] for r in records})


def score_of(record: Mapping[str, Any]) -> Optional[float]:
    score = (record.get("results") or {}).get("score")
    return None if score is None else float(score)
