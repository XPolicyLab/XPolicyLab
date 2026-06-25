"""Persist dispatch and per-trial publish state under artifact_root."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from robodojo.schemas import DispatchPayload

PUBLISH_STATUS_PENDING = "pending"
PUBLISH_STATUS_DONE = "done"
PUBLISH_STATUS_FAILED = "failed"
INCOMPLETE_STATUSES = frozenset({PUBLISH_STATUS_PENDING, PUBLISH_STATUS_FAILED})


@dataclass(frozen=True)
class PublishRecord:
    evaluation_id: str
    trial_index: int
    hdf5_path: str | None
    run_status: str
    publish_status: str


def _evaluation_dir(artifact_root: Path, evaluation_id: str) -> Path:
    return Path(artifact_root) / quote(evaluation_id, safe="")


def _dispatch_path(artifact_root: Path, evaluation_id: str) -> Path:
    return _evaluation_dir(artifact_root, evaluation_id) / "dispatch.json"


def _publish_path(artifact_root: Path, evaluation_id: str, trial_index: int) -> Path:
    return (
        _evaluation_dir(artifact_root, evaluation_id)
        / "trials"
        / str(trial_index)
        / "publish.json"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def record_dispatch(
    artifact_root: Path,
    evaluation_id: str,
    dispatch: DispatchPayload,
) -> None:
    _atomic_write_json(
        _dispatch_path(artifact_root, evaluation_id),
        dispatch.model_dump(mode="json"),
    )


def load_dispatches(artifact_root: Path) -> dict[str, DispatchPayload]:
    root = Path(artifact_root)
    if not root.is_dir():
        return {}

    dispatches: dict[str, DispatchPayload] = {}
    for eval_dir in root.iterdir():
        if not eval_dir.is_dir():
            continue
        dispatch_file = eval_dir / "dispatch.json"
        if not dispatch_file.is_file():
            continue
        try:
            raw = json.loads(dispatch_file.read_text(encoding="utf-8"))
            evaluation_id = unquote(eval_dir.name)
            dispatches[evaluation_id] = DispatchPayload.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return dispatches


def record_pending(
    artifact_root: Path,
    evaluation_id: str,
    trial_index: int,
    hdf5_path: str | None,
    run_status: str,
) -> None:
    path = _publish_path(artifact_root, evaluation_id, trial_index)
    payload: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "trial_index": trial_index,
        "hdf5_path": hdf5_path,
        "run_status": run_status,
        "publish_status": PUBLISH_STATUS_PENDING,
    }
    if path.is_file():
        payload = {**json.loads(path.read_text(encoding="utf-8")), **payload}
    _atomic_write_json(path, payload)


def record_outcome(
    artifact_root: Path,
    evaluation_id: str,
    trial_index: int,
    publish_status: str,
) -> None:
    path = _publish_path(artifact_root, evaluation_id, trial_index)
    payload: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "trial_index": trial_index,
        "publish_status": publish_status,
    }
    if path.is_file():
        payload = {**json.loads(path.read_text(encoding="utf-8")), **payload}
    _atomic_write_json(path, payload)


def load_publish_record(
    artifact_root: Path,
    evaluation_id: str,
    trial_index: int,
) -> PublishRecord | None:
    path = _publish_path(artifact_root, evaluation_id, trial_index)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return PublishRecord(
        evaluation_id=str(raw.get("evaluation_id", evaluation_id)),
        trial_index=int(raw.get("trial_index", trial_index)),
        hdf5_path=str(raw["hdf5_path"]) if raw.get("hdf5_path") else None,
        run_status=str(raw.get("run_status", "done")),
        publish_status=str(raw.get("publish_status", PUBLISH_STATUS_PENDING)),
    )


def load_incomplete(artifact_root: Path) -> list[PublishRecord]:
    root = Path(artifact_root)
    if not root.is_dir():
        return []

    records: list[PublishRecord] = []
    for eval_dir in root.iterdir():
        if not eval_dir.is_dir():
            continue
        evaluation_id = unquote(eval_dir.name)
        trials_dir = eval_dir / "trials"
        if not trials_dir.is_dir():
            continue
        for trial_dir in trials_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            publish_file = trial_dir / "publish.json"
            if not publish_file.is_file():
                continue
            try:
                trial_index = int(trial_dir.name)
                raw = json.loads(publish_file.read_text(encoding="utf-8"))
                publish_status = str(raw.get("publish_status", PUBLISH_STATUS_PENDING))
                if publish_status not in INCOMPLETE_STATUSES:
                    continue
                records.append(
                    PublishRecord(
                        evaluation_id=str(raw.get("evaluation_id", evaluation_id)),
                        trial_index=int(raw.get("trial_index", trial_index)),
                        hdf5_path=str(raw["hdf5_path"])
                        if raw.get("hdf5_path")
                        else None,
                        run_status=str(raw.get("run_status", "done")),
                        publish_status=publish_status,
                    )
                )
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    records.sort(key=lambda item: (item.evaluation_id, item.trial_index))
    return records
