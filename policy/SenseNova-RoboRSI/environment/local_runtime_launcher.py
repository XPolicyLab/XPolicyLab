#!/usr/bin/env python3
"""Place current policy code in its downloaded native workspace and run eval.sh.

This wrapper does not modify agent code, install packages, load a model, or start
anything during --prepare-only / --check. The native workspace is a real directory:
symlinking all of XPolicyLab would make __file__.resolve() select the source-only
checkout and lose the native env_cfg/task/assets root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import uuid

SKIP = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
        ".ruff_cache", ".codex", ".agents", ".claude", ".cursor", ".aws", ".ssh", "node_modules", "wandb"}
POLICY_DATA_SKIP = {".venv", "runtime", "checkpoints", "outputs", "environment", ".validation"}
SECRET_NAMES = {"auth.json", "api_keys.yaml", "api_keys.yml", "credentials", ".env"}
LINK_ROOTS = (".venv", "runtime", "checkpoints")


class LayoutError(ValueError):
    pass


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex)
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def source_files(source: Path, policy_name: str):
    """Select framework plus exactly one policy without following source links."""
    for directory, names, files in os.walk(source, followlinks=False):
        here = Path(directory)
        relative = here.relative_to(source)
        names[:] = sorted(name for name in names if name not in SKIP)
        if relative == Path("policy"):
            names[:] = [name for name in names if name == policy_name]
        policy_top = relative == Path("policy") / policy_name
        if policy_top:
            names[:] = [name for name in names if name not in POLICY_DATA_SKIP]
        # Source symlinks can conceal dependencies or escape the copied tree;
        # installed runtime links are handled explicitly after the source copy.
        for name in names:
            if (here / name).is_symlink():
                raise LayoutError("source directory symlink needs explicit mapping")
        for name in sorted(files):
            path = here / name
            if (name in SKIP or policy_top and name in POLICY_DATA_SKIP
                    or name in SECRET_NAMES or name.endswith((".pyc", ".pyo"))):
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise LayoutError("source file must be regular")
            yield path, path.relative_to(source)


def validate_native(workspace: Path):
    if workspace.is_symlink() or not workspace.is_dir():
        raise LayoutError("official workspace must be a real downloaded directory")
    for name in ("scripts/eval_policy.sh", "src/eval_client/main.py", "env_cfg/arx_x5.yml",
                 "env/seed_manager/seed_manager.py", "utils/load_file.py"):
        if not (workspace / name).is_file():
            raise LayoutError("downloaded native workspace is missing a required entry")


def prepare(policy: Path, workspace: Path):
    policy = policy.resolve(strict=True)
    validate_native(workspace)
    framework = policy.parent.parent
    if framework.name != "XPolicyLab" or policy.parent.name != "policy":
        raise LayoutError("expected XPolicyLab/policy/POLICY source layout")
    if workspace.resolve() == framework.parent:
        raise LayoutError("native workspace must be distinct from the source checkout")
    target = workspace / "XPolicyLab"
    stage = workspace / (".XPolicyLab.local-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    files = []
    try:
        for source, relative in source_files(framework, policy.name):
            destination = stage / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            expected = sha256(source)
            if sha256(destination) != expected:
                raise LayoutError("copied source SHA-256 mismatch")
            files.append({"path": relative.as_posix(), "bytes": source.stat().st_size,
                          "sha256": expected})
        copied_policy = stage / "policy" / policy.name
        for name in LINK_ROOTS:
            runtime = policy / name
            if not runtime.exists() and not runtime.is_symlink():
                raise LayoutError("canonical policy environment or weights are missing")
            destination = copied_policy / name
            if destination.exists() or destination.is_symlink():
                raise LayoutError("code filter included a runtime root")
            os.symlink(os.path.relpath(runtime, (target / "policy" / policy.name)), destination)
        if target.is_symlink():
            raise LayoutError("refusing to replace symlinked XPolicyLab")
        backup = None
        if target.exists():
            backup = workspace / ("XPolicyLab.before-local-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                                  + "-" + uuid.uuid4().hex[:8])
            target.rename(backup)
        stage.rename(target)
        receipt = {"schema": 1, "operation": "copy_current_code_into_downloaded_native_workspace",
                   "source_policy": str(policy), "native_workspace": str(workspace.resolve()),
                   "source_files": files, "runtime_roots": list(LINK_ROOTS),
                   "previous_framework_backup": str(backup) if backup else None,
                   "models_or_simulator_started": False,
                   "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(policy / "environment" / "local-launcher-code.json", receipt)
        return {"status": "prepared", "source_files": len(files),
                "policy": str(target / "policy" / policy.name)}
    except BaseException:
        # Keep a failed private stage for diagnosis; never delete source/runtime.
        raise


def check(policy: Path, workspace: Path, simulator: Path, conda: Path):
    validate_native(workspace)
    entries = {"controller": policy / ".venv/bin/python", "simulator": simulator / "bin/python",
               "conda": conda / "bin/conda", "conda_shell": conda / "etc/profile.d/conda.sh"}
    for name, path in entries.items():
        if not path.is_file() or (name != "conda_shell" and not os.access(path, os.X_OK)):
            raise LayoutError("local runtime entry is missing or not executable: " + name)
    deployed = workspace / "XPolicyLab/policy" / policy.name
    if deployed.is_symlink() or not (deployed / "eval.sh").is_file():
        raise LayoutError("prepare a real current-policy directory before running")
    for name in LINK_ROOTS:
        path = deployed / name
        if not path.is_symlink() or path.resolve(strict=True) != (policy / name).resolve(strict=True):
            raise LayoutError("deployed runtime does not point to canonical local policy")
    return {"status": "layout_checked", "models_or_simulator_started": False,
            "native_workspace": str(workspace.resolve()), "policy": str(deployed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sim-prefix", type=Path, required=True)
    parser.add_argument("--conda-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER,
                        help="after --: the first 9 original eval.sh arguments; simulator prefix is bound locally")
    args = parser.parse_args(argv)
    policy = args.policy_dir.absolute()
    workspace = args.workspace.absolute()
    if args.prepare_only:
        result = prepare(policy, workspace)
    else:
        result = check(policy, workspace, args.sim_prefix.absolute(), args.conda_root.absolute())
    if not args.run:
        print(json.dumps(result, sort_keys=True))
        return 0
    evaluation = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    if len(evaluation) != 9:
        parser.error("--run requires 9 eval.sh arguments after --; the 10th is the local simulator prefix")
    if not os.environ.get("CODEX_API_KEY"):
        parser.error("set CODEX_API_KEY before evaluation")
    shell = '''set -euo pipefail
export PATH="$1/bin:$1/condabin:$PATH"
set +u
source "$1/etc/profile.d/conda.sh"
set -u
shift
exec bash "$@"
'''
    deployed = workspace / "XPolicyLab/policy" / policy.name
    environment = os.environ.copy()
    environment.setdefault("ROBORSI_TRAJECTORY_DIR", str(policy / "outputs/trajectories"))
    environment.setdefault("PYTHONNOUSERSITE", "1")
    # Do not print this environment or full argv: it may include operator inputs.
    return subprocess.call(["bash", "-c", shell, "local-roborsi", str(args.conda_root.absolute()),
                            str(deployed / "eval.sh"), *evaluation, str(args.sim_prefix.absolute())],
                           cwd=deployed, env=environment)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LayoutError, OSError) as error:
        print(json.dumps({"status": "failed", "exception": type(error).__name__,
                          "reason": str(error)}), file=sys.stderr)
        raise SystemExit(2)
