"""Import evidenced offline asset revisions into a new common-Agent release.

This helper does not run experiments, train models, or choose per-task executors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from .release import ROOT, contained, digest, freeze


def import_decisions(decisions_path: Path, candidate: Path) -> Path:
    decisions_path = decisions_path.resolve()
    record = json.loads(decisions_path.read_text())
    if (record.get("schema") != "r2si.asset-revisions.v2" or record.get("author_type") != "outer_agent"
            or not record.get("agent_run_id") or not record.get("updates")):
        raise ValueError("Expected evidenced offline asset revisions")
    revisions = []
    for update in record["updates"]:
        target = update["target"]
        if (not target.startswith("vendor/policy/RoboRSI_RoboProbe/skills/")
                or Path(target).is_absolute() or ".." in Path(target).parts):
            raise ValueError("Only existing Skill assets may be revised by this importer")
        contained(ROOT, target)
        if not update.get("reason") or not update.get("evidence"):
            raise ValueError("Every revision needs a reason and experiment evidence")
        for reference in update["evidence"]:
            source = Path(reference["path"])
            source = source if source.is_absolute() else decisions_path.parent / source
            if not source.is_file() or digest(source) != reference.get("sha256"):
                raise ValueError("Experiment evidence is missing or changed")
        source = Path(update["source"])
        source = source if source.is_absolute() else decisions_path.parent / source
        if not source.is_file() or digest(source) != update.get("sha256"):
            raise ValueError("Revision asset is missing or changed")
        revisions.append((target, source))
    candidate = candidate.resolve()
    if candidate.exists() or candidate.is_relative_to(ROOT):
        raise FileExistsError("Use a new directory outside the active release")
    shutil.copytree(ROOT, candidate, ignore=shutil.ignore_patterns("release.json", "__pycache__"))
    for target, source in revisions:
        shutil.copyfile(source, candidate / target)
    shutil.copyfile(decisions_path, candidate / "configs/outer_asset_revisions.json")
    return freeze(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(import_decisions(args.decisions, args.candidate))
