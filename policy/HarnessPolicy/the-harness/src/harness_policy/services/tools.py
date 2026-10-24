"""`harness doctor` and `harness fetch`: is this machine ready for a deploy, and get
the weights it lacks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import deploy as deploys
from .supervisor import status


def gpus() -> list[dict] | None:
    """GPUs nvidia-smi reports (index, name, memory in GB), or None without it."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = []
    for line in out.strip().splitlines():
        index, name, used, total = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "name": name, "used_gb": round(float(used) / 1024, 1),
                     "total_gb": round(float(total) / 1024, 1)})
    return rows


def doctor(deploy, *, names=None, keys=()) -> dict:
    """What stands between this machine and running `deploy`: missing files, ports
    held by something else, GPUs short of memory, missing interpreters and API keys.
    `keys` are environment variables the selected system needs (API keys)."""
    problems, notes = [], []
    if sys.version_info < (3, 11):
        problems.append(f"Python {sys.version.split()[0]}: 3.11 or newer is needed")
    if deploy.path is None:
        notes.append("No deploy file (--deploy or $HARNESS_DEPLOY): no service runs here; hosted models only.")
    devices = gpus()
    by_index = {gpu["index"]: gpu for gpu in devices or []}
    names = list(names) if names else sorted(n for n, p in deploy.placements.items() if p.mode != "off")
    planned: dict[int, float] = {}
    groups: dict[tuple, float] = {}
    rows = {row["service"]: row for row in status(deploy, names)}
    for name in names:
        placement = deploy.placement(name)
        spec = deploy.services[name]
        if placement.mode == "off":
            continue
        if placement.mode == "external":
            if not rows[name].get("up"):
                problems.append(f"{name}: external URL {placement.url} does not answer")
            continue
        missing = deploys.missing_files(deploy, name)
        if missing:
            problems.append(f"{name}: missing {', '.join(missing)} (harness fetch {name} --deploy {deploy.name})")
        argv, _environment, cwd = deploys.launch(deploy, name)
        if not (Path(argv[0]).exists() or shutil.which(argv[0])):
            problems.append(f"{name}: interpreter {argv[0]} not found (set python for it in the deploy file)")
        if not Path(cwd).is_dir():
            problems.append(f"{name}: working directory {cwd} not found")
        if not rows[name].get("up") and isinstance(placement.gpu, int):
            if placement.mode == "on_demand" and placement.swap:
                # One service of a swap group is loaded at a time: count the largest.
                key = (placement.gpu, placement.swap)
                groups[key] = max(groups.get(key, 0.0), spec.vram_gb)
            else:
                planned[placement.gpu] = planned.get(placement.gpu, 0.0) + spec.vram_gb
        if isinstance(placement.gpu, int) and devices is not None and placement.gpu not in by_index:
            problems.append(f"{name}: GPU {placement.gpu} does not exist here")
        port = deploys.port(deploy, name)
        if rows[name].get("up") and not rows[name].get("started_by_harness"):
            notes.append(f"{name}: port {port} already answers (started outside harness up); it will be used as is")
    for (index, _group), need in groups.items():
        planned[index] = planned.get(index, 0.0) + need
    for index, need in sorted(planned.items()):
        gpu = by_index.get(index)
        if gpu and gpu["total_gb"] - gpu["used_gb"] < need:
            problems.append(f"GPU {index}: {need:.1f} GB planned for services still to start, "
                            f"{gpu['total_gb'] - gpu['used_gb']:.1f} GB free")
    for name in (deploy.sim.get("gpus") or []):
        if devices is not None and name not in by_index:
            problems.append(f"sim: GPU {name} does not exist here")
    for key in keys:
        if not os.environ.get(key):
            problems.append(f"{key} is not set (source ~/.harness_env or export it)")
    if not deploy.robodojo.is_dir():
        notes.append(f"RoboDojo checkout not found at {deploy.robodojo} (needed for RoboDojo workers and XPolicyLab services)")
    return {"deploy": deploy.name, "ok": not problems, "problems": problems, "notes": notes,
            "gpus": devices, "services": [rows[name] for name in names if name in rows]}


def fetch(deploy, name: str, *, log=print) -> int:
    """Download the weights of one service into the deploy's weights root. Returns
    the number of entries that failed."""
    spec = deploy.services[name]
    scripts = deploys.REPO / "scripts"
    failures = 0
    for entry in spec.weights:
        dest = Path(deploy.weights) / entry["dir"]
        if dest.is_symlink() or (dest.exists() and entry.get("local")):
            log(f"{name}: {dest} is provided locally; skipped")
            continue
        if "modelscope_model" in entry:
            command = [sys.executable, str(scripts / "modelscope_download.py"), "model", entry["modelscope_model"], str(dest)]
        elif "modelscope_dataset" in entry:
            command = [sys.executable, str(scripts / "modelscope_download.py"), "dataset",
                       entry["modelscope_dataset"], str(dest), "--root", entry.get("root", "")]
        elif "hf_model" in entry or "hf_dataset" in entry:
            repo = entry.get("hf_model") or entry["hf_dataset"]
            command = [sys.executable, str(scripts / "hf_mirror_download.py"), repo, str(dest)]
            if "hf_dataset" in entry:
                command += ["--repo-type", "dataset"]
            for pattern in entry.get("include", []):
                command += ["--include", pattern]
        else:
            log(f"{name}: {entry['dir']} must be put in place by hand ({entry.get('local')})")
            continue
        for pattern in entry.get("exclude", []) if "modelscope_model" in entry or "modelscope_dataset" in entry else []:
            command += ["--exclude", pattern]
        log(f"{name}: {' '.join(command)}")
        if subprocess.run(command).returncode != 0:
            failures += 1
    return failures
