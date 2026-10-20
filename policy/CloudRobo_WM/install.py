"""Verify bundled source and install into a stage-specific editable environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def run(*args: str, **kwargs) -> str:
    result = subprocess.run(args, check=True, text=True, **kwargs)
    return result.stdout.strip() if result.stdout else ""


def install(stage: str, interpreter: str, offline: bool) -> None:
    if not (ROOT.parents[1] / "setup_policy_server.py").is_file():
        raise RuntimeError("Copy the adapter to XPolicyLab/policy/CloudRobo_WM before running install.sh")
    lock = json.loads((ROOT / "source_lock.json").read_text())
    spec = lock[stage]
    checkout = ROOT / spec["directory"]
    if not checkout.exists():
        raise RuntimeError(f"Bundled source directory is missing: {checkout}")
    if (checkout / ".git").exists():
        # Preserve the existing checks for an already working source checkout.
        head = run("git", "-C", str(checkout), "rev-parse", "HEAD", capture_output=True)
        dirty = run("git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no", capture_output=True)
        if head != spec["commit"] or dirty:
            raise RuntimeError(f"Existing {checkout.name} is not the clean pinned commit.")
    else:
        manifest = json.loads((ROOT / "vendored_sources.json").read_text())
        if manifest["sources"][stage] != spec:
            raise RuntimeError("Vendored source revision differs from source_lock.json")
        rows = [row for row in manifest["files"] if row["stage"] == stage]
        if not rows:
            raise RuntimeError(f"No source hashes for {stage}")
        expected = set()
        for row in rows:
            path = ROOT / row["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(checkout.resolve()):
                raise RuntimeError("Invalid vendored source path")
            expected.add(path.resolve())
            if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Vendored source checksum mismatch: {row['path']}")
        actual = {path.resolve() for path in checkout.rglob("*") if path.is_file()
                  and not any(part in {".venv", "__pycache__"} or part.endswith(".egg-info")
                              for part in path.relative_to(checkout).parts)
                  and path.suffix not in {".pyc", ".pyo"}}
        if actual != expected:
            raise RuntimeError("Unexpected or missing files in vendored source snapshot")
        head = spec["commit"] + " (verified vendored snapshot)"

    target_python = Path(shutil.which(interpreter) or interpreter).absolute()
    # Avoid resolving a venv's Python symlink to the base interpreter.
    version_guard = ('assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Inference requires Python 3.10-3.12"'
                     if stage == "inference" else
                     'assert sys.version_info[:2] == (3, 12), "Post-training reference requires Python 3.12"')
    check = ('import sys; ' + version_guard + '; from importlib import metadata; '
             'import torch, torchvision; print(sys.version_info[:2]); '
             'print(metadata.version("torch").partition("+")[0], metadata.version("torchvision").partition("+")[0])')
    run(str(target_python), "-c", check)
    venv = checkout / ".venv"
    if not venv.exists():
        run(str(target_python), "-m", "venv", "--system-site-packages", str(venv))
    python = venv / "bin" / "python"
    if stage == "posttrain":
        run(str(python), str(checkout / "tools" / "install_dependencies.py"))
    else:
        # Exact runtime constraints prevent pip from replacing the selected framework build.
        versions = json.loads(run(str(python), "-c", 'import json; from importlib import metadata; print(json.dumps({n:metadata.version(n) for n in ("torch", "torchvision")}))', capture_output=True))
        with tempfile.TemporaryDirectory(prefix="crwm-constraints-") as tmp:
            constraints = Path(tmp) / "constraints.txt"
            constraints.write_text("\n".join([*(f"{n}==={v}" for n, v in versions.items()), "numpy==1.26.4", "opencv-python-headless==4.10.0.84"]) + "\n")
            run(str(python), "-m", "pip", "install", "--constraint", str(constraints), "--editable", str(checkout), "--editable", str(ROOT.parents[1]))
        after = json.loads(run(str(python), "-c", 'import json; from importlib import metadata; print(json.dumps({n:metadata.version(n) for n in ("torch", "torchvision")}))', capture_output=True))
        if versions != after:
            raise RuntimeError("Framework versions changed during installation")
    origin = run(str(python), "-c", 'import crwm; print(crwm.__file__)', capture_output=True)
    if not Path(origin).resolve().is_relative_to(checkout.resolve()):
        raise RuntimeError(f"crwm resolves outside the selected source checkout: {origin}")
    print(f"Installed {stage} at {head}; interpreter: {python}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inference", "posttrain"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    install(args.stage, args.python, args.offline)


if __name__ == "__main__":
    main()
