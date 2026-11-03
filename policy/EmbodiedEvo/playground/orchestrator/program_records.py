"""Condition C: export public Playground history without workspace snapshots."""

import hashlib
import json
from pathlib import Path
import shutil

from orchestrator.files import hashes, write_json
from orchestrator.frozen_source import digest, read, snapshot


PUBLIC_FILES = {
    "code.py", "sources.json", "stdout.txt", "stderr.txt", "result.json",
    "observation.json", "arrays.npz", "images.json",
    *{f"current_cam_{camera}.png" for camera in ("head", "left_wrist", "right_wrist")},
    *{f"execution_cam_{camera}.mp4" for camera in ("head", "left_wrist", "right_wrist")},
}


def reject_private_fields(value):
    if isinstance(value, dict):
        if set(value) & {"score", "reward", "shaped_reward", "ledger", "skills", "lessons"}:
            raise ValueError("Private score or knowledge in public Playground feedback")
        for child in value.values():
            reject_private_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_private_fields(child)


def export_bundle(source, destination, instruction):
    """Copy the full accepted execution sequence; never select by success."""
    source_snapshot = snapshot(source)
    if source_snapshot is None:
        raise ValueError("C requires a valid frozen Playground source")
    destination, instruction = Path(destination).resolve(), Path(instruction).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "records"
    observations = Path(source["run"]) / "playground/workspace/observations"
    entries = sorted(p for p in observations.iterdir() if p.is_dir() and p.name.isdecimal())
    expected = read(Path(source["run"]) / "playground/session-result.json")["executions_used"]
    if [p.name for p in entries] != [f"{i:06d}" for i in range(expected + 1)]:
        raise ValueError("Playground execution history is incomplete")
    index = []
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("Linked Playground execution directory")
        if int(entry.name) and not all((entry / n).is_file() for n in ("code.py", "result.json", "stdout.txt", "stderr.txt")):
            raise ValueError(f"Missing public execution evidence: {entry}")
        files = []
        for name in sorted(PUBLIC_FILES):
            original = entry / name
            if not original.exists():
                continue
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"Invalid public record: {original}")
            if original.suffix == ".json":
                reject_private_fields(read(original))
            target = root / "observations" / entry.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            files.append(str(target.relative_to(root)))
        index.append({"execution": int(entry.name), "files": files})
    write_json(root / "index.json", {"executions": index, "execution_count": expected})
    (root / "README.md").write_text(
        "# Playground program records\n\n"
        "Read index.json for the complete ordered history. observations/000000 is the initial "
        "observation. Later directories contain the exact executed code and original public feedback. "
        "Both successful and unsuccessful executions are retained. Paths inside archived JSON are "
        "relative to program_records/, not to the current Test workspace.\n\n"
        "This is a different episode. Its state, success flags, action counts and remaining budgets "
        "describe that past episode. Use scripts/env.py status and current observations for this Test. "
        "Reset was allowed in Playground but is forbidden in Test. Copy/adapt useful code into "
        "submission/; do not execute a past trajectory blindly. These records stay read-only.\n",
        encoding="utf-8",
    )
    if snapshot(source) != source_snapshot:
        raise ValueError("Playground source changed during export")
    manifest = {"source_task": source["task_id"], "source_run": source["run"],
                "source_snapshot": source_snapshot, "execution_count": expected,
                "files": hashes(root)}
    write_json(destination / "bundle.json", manifest)
    return {"manifest": str(destination / "bundle.json"),
            "manifest_sha256": digest(destination / "bundle.json"),
            "instruction": str(instruction), "instruction_sha256": digest(instruction)}


def validate_bundle(reference, *, source_snapshot=None, full=False):
    manifest = Path(reference["manifest"])
    if manifest.is_symlink() or digest(manifest) != reference["manifest_sha256"]:
        raise ValueError("C record manifest changed")
    data = read(manifest)
    if source_snapshot is not None and data["source_snapshot"] != source_snapshot:
        raise ValueError("C records refer to a different Playground snapshot")
    root = manifest.parent / "records"
    if full and hashes(root) != data["files"]:
        raise ValueError("C program records changed")
    return root
