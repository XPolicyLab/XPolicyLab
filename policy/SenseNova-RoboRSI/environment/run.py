#!/usr/bin/env python3
"""Run downloaded environments at their original prefixes in a mount namespace.

No installed file, shebang, editable metadata, venv configuration or original
installer receipt is modified. Payloads live locally; the namespace maps those
directories to the original absolute names only for this child process. It
inherits the host network and GPU devices. --probe runs standard-library Python
identity checks only and does not import a model or simulator.

Plan schema (all paths absolute):
 {"policy_dir": ".../XPolicyLab/policy/POLICY", "native_workspace": "...",
  "source_workspace": "/original/workspace/RoboDojo",
  "source_policy": "/original/workspace/RoboDojo/XPolicyLab/policy/POLICY",
  "simulator_prefix": "/original/conda/envs/RoboDojo",
  "conda_root": "/original/conda", "controller_python": "/original/.../bin/python",
  "bwrap": {"path": "/local/bwrap", "sha256": "64 hex"},
  "local_access_roots": ["/owned/local/source-and-snapshots"],
  "mounts": [{"local": "/local/payload", "source": "/original/prefix"}],
  "native_kit_extension_parent": "/logical/preserved/extensions/v2",  # optional
  "native_kit_portable_per_run": true,  # optional, defaults false
  "probes": [{"name": "controller", "python": "/original/.../bin/python"}]}

The native workspace must be a real directory. The current strict XPolicyLab
directory is mounted over its original namespace path (not symlinked), so
__file__.resolve() retains the correct official workspace parent. Mounts are read-only by default.
The current run's state, temporary files, trajectories and native eval_result
are separate writable directories under the canonical policy's outputs/.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

spec = importlib.util.spec_from_file_location("roborsi_local_runtime_launcher",
                                             Path(__file__).with_name("local_runtime_launcher.py"))
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class NamespaceError(ValueError):
    pass


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or any(x in str(path) for x in "\0\r\n"):
        raise NamespaceError("namespace paths must be clean and absolute")
    return path


NATIVE_KIT_EXTENSION_ENV = "ROBORSI_NATIVE_KIT_EXTENSION_PARENT"
NATIVE_KIT_PORTABLE_ENV = "ROBORSI_NATIVE_KIT_PORTABLE_ROOT"
NATIVE_KIT_EXTENSION_CONFIGS = {
    "isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311":
        "1984a4ecf3134745215638420e449c197083e97a3932901656479a7677f9a3fb",
    "omni.kit.pip_archive-d38fa9ecd1fb6df4":
        "824447f8269824374116ec9ee8f12f8e83a5bca49f85b2a564b1f9678a6a541d",
}


def _check_native_extension_config(root, expected_sha256, error):
    """Accept only the preserved public extension metadata at an owned real root."""
    for directory in (root, root / "config"):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise error("native extension metadata requires owned real directories")
    config = root / "config/extension.toml"
    info = config.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise error("native extension config must be an owned regular file")
    if hashlib.sha256(config.read_bytes()).hexdigest() != expected_sha256:
        raise error("native extension metadata differs from the preserved source")


def native_kit_extension_parent(plan):
    """Validate virtual discovery parent through its two explicit owned mounts."""
    if "native_kit_extension_parent" not in plan:
        return None
    parent = absolute(plan["native_kit_extension_parent"])
    if any(character.isspace() for character in str(parent)):
        raise NamespaceError("native extension parent cannot contain whitespace")
    # The virtual parent is created by bubblewrap from these child mounts; it
    # need not exist on the host. Validate real local payloads, never host /root.
    for name, expected in NATIVE_KIT_EXTENSION_CONFIGS.items():
        source = parent / name
        rows = [row for row in plan["mounts"] if Path(row["source"]) == source]
        if len(rows) != 1:
            raise NamespaceError("native extension parent requires both explicit preserved SDK mounts")
        local = Path(rows[0]["local"])
        roots = [Path(value).resolve(strict=True) for value in plan["local_access_roots"]]
        if not any(local.resolve(strict=True).is_relative_to(root) for root in roots):
            raise NamespaceError("native extension metadata escapes owned local access roots")
        _check_native_extension_config(local, expected, NamespaceError)
    return str(parent)


def native_kit_portable_root(plan, run_root=None):
    """Create only this run's owned writable Kit cache/data/logs root."""
    enabled = plan.get("native_kit_portable_per_run", False)
    if not isinstance(enabled, bool):
        raise NamespaceError("native_kit_portable_per_run must be a boolean")
    if not enabled or run_root is None:
        return None
    root = absolute(run_root) / "cache/native-kit"
    if any(character.isspace() for character in str(root)):
        raise NamespaceError("native Kit storage cannot contain whitespace")
    owned_path(plan, root, must_exist=False)
    for directory in (root.parent.parent, root.parent):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise NamespaceError("native Kit storage requires an existing owned run/cache root")
    root.mkdir(mode=0o700, exist_ok=True)
    owned_path(plan, root, must_exist=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise NamespaceError("native Kit storage must be an owned real directory")
    return str(root)


def load_plan(path):
    plan = json.loads(path.read_text())
    for name in ("policy_dir", "native_workspace", "source_workspace", "source_policy",
                 "simulator_prefix", "conda_root", "controller_python"):
        plan[name] = str(absolute(plan[name]))
    policy = Path(plan["policy_dir"])
    if policy.parent.name != "policy" or policy.parent.parent.name != "XPolicyLab":
        raise NamespaceError("unexpected canonical policy layout")
    expected = Path(plan["source_workspace"]) / "XPolicyLab/policy" / policy.name
    if Path(plan["source_policy"]) != expected:
        raise NamespaceError("source policy must be in the selected native workspace")
    layout.validate_native(Path(plan["native_workspace"]))
    access = [absolute(value) for value in plan["local_access_roots"]]
    if not access or any(not root.is_dir() or root.is_symlink() for root in access):
        raise NamespaceError("local access roots must be existing real directories")
    for root in access:
        if root.stat().st_uid != os.geteuid():
            raise NamespaceError("local access root is not owned by the current user")
    def allowed(local):
        return any(local.resolve(strict=True).is_relative_to(root.resolve(strict=True)) for root in access)
    if not allowed(policy) or not allowed(Path(plan["native_workspace"])):
        raise NamespaceError("canonical code/native workspace escapes local access roots")
    seen = set()
    mounts = []
    for row in plan["mounts"]:
        local, source = absolute(row["local"]), absolute(row["source"])
        if local.is_symlink() or not local.exists() or not allowed(local):
            raise NamespaceError("namespace payload must be a real existing owned local object")
        if str(source) in seen or source == Path("/"):
            raise NamespaceError("duplicate or root source mount")
        if source.parts[:2] in [("/", "etc"), ("/", "usr"), ("/", "dev"), ("/", "proc")]:
            raise NamespaceError("payload mount must not replace a host system root")
        seen.add(str(source))
        mounts.append({"local": str(local), "source": str(source)})
    # The current code copy always supplies the whole official workspace mount.
    source_workspace = plan["source_workspace"]
    existing = [row for row in mounts if row["source"] == source_workspace]
    if existing and existing[0]["local"] != plan["native_workspace"]:
        raise NamespaceError("native source workspace maps to the wrong local version")
    if not existing:
        mounts.append({"local": plan["native_workspace"], "source": source_workspace})
    source_framework = str(Path(source_workspace) / "XPolicyLab")
    local_framework = str(policy.parent.parent)
    existing = [row for row in mounts if row["source"] == source_framework]
    if existing and existing[0]["local"] != local_framework:
        raise NamespaceError("namespace framework must use the current strict source directory")
    if not existing:
        mounts.append({"local": local_framework, "source": source_framework})
    mounts.sort(key=lambda row: (len(PurePosixPath(row["source"]).parts), row["source"]))
    plan["mounts"] = mounts
    native_kit_extension_parent(plan)
    native_kit_portable_root(plan)
    binary = absolute(plan["bwrap"]["path"])
    expected = plan["bwrap"]["sha256"]
    if (not re.fullmatch("[0-9a-f]{64}", expected) or not binary.is_file()
            or not os.access(binary, os.X_OK) or layout.sha256(binary) != expected):
        raise NamespaceError("bubblewrap does not match the pinned executable SHA-256")
    return plan


def namespace_local_path(plan, value):
    """Resolve a namespace path through explicit mounts, without following links."""
    path = absolute(value)
    mappings = [(Path(row["source"]), Path(row["local"])) for row in plan["mounts"]]
    mappings += [(Path(root), Path(root)) for root in plan["local_access_roots"]]
    matches = [(source, local) for source, local in mappings if path.is_relative_to(source)]
    if not matches:
        raise NamespaceError("resume metadata path is outside the explicit local mappings")
    source, local = max(matches, key=lambda row: len(row[0].parts))
    return local / path.relative_to(source)


def owned_path(plan, path, *, must_exist):
    """Require a literal owned local path, including all existing ancestors."""
    path = absolute(path)
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise NamespaceError("resume metadata path must exist and be accessible") from error
    if resolved != path:
        raise NamespaceError("resume metadata paths must not traverse symlinks")
    roots = [Path(root) for root in plan["local_access_roots"]]
    if not any(path.is_relative_to(root) for root in roots):
        raise NamespaceError("resume metadata must stay in an explicit owned local root")
    parent = path
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_uid != os.geteuid():
        raise NamespaceError("resume metadata local path is not owned by the current user")
    return path


def resume_metadata_mount(plan, source, evaluation):
    """Grant only the selected episode's recovery metadata directory write access.

    The unchanged resume preparer updates its original adapter-recovery state;
    action records and the old agent workspace remain read-only. In a later
    attempt the owner may name an earlier episode's state, so preserve that
    path and resolve it through the same namespace plan instead of relocating it.
    """
    module = "XPolicyLab.policy." + Path(plan["policy_dir"]).name + ".hybrid_resume"
    if evaluation[:2] != ["-m", module]:
        raise NamespaceError("--resume-state-source is only valid for this policy's hybrid_resume module")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True)
    parsed, _ = parser.parse_known_args(evaluation[2:])
    source = absolute(source)
    if absolute(parsed.source) != source:
        raise NamespaceError("resume metadata source must equal hybrid_resume --source")
    local_source = owned_path(plan, namespace_local_path(plan, source), must_exist=True)
    if not local_source.is_dir():
        raise NamespaceError("resume metadata source must be an existing runtime directory")
    owner_path = local_source / "episode-owner.json"
    state = source / "adapter-recovery/state.json"
    if owner_path.exists() or owner_path.is_symlink():
        owned_path(plan, owner_path, must_exist=True)
        if not owner_path.is_file() or owner_path.stat().st_size > 65536:
            raise NamespaceError("resume source owner must be a small regular metadata file")
        owner = json.loads(owner_path.read_text())
        if not isinstance(owner, dict):
            raise NamespaceError("resume source owner metadata must be an object")
        state = absolute(owner.get("recovery_state", str(state)))
    if state.name != "state.json" or state.parent.name != "adapter-recovery":
        raise NamespaceError("resume write access is limited to adapter-recovery/state.json metadata")
    # An owned prefix alone is insufficient: it also contains executable
    # environments and model weights. The destination must remain the recovery
    # subdirectory of a retained runtime, including an earlier own attempt.
    local_runtime = owned_path(plan, namespace_local_path(plan, state.parent.parent), must_exist=True)
    controller = owned_path(plan, local_runtime / "agent/controller", must_exist=True)
    if (not local_runtime.name.startswith("runtime-") or not local_runtime.is_dir()
            or not controller.is_dir()):
        raise NamespaceError("resume state must belong to a retained runtime's adapter-recovery directory")
    local_state = owned_path(plan, namespace_local_path(plan, state.parent), must_exist=False)
    local_state.mkdir(mode=0o700, parents=True, exist_ok=True)
    owned_path(plan, local_state, must_exist=True)
    if not local_state.is_dir():
        raise NamespaceError("resume recovery metadata must be a directory")
    for name in ("state.json", "state.lock"):
        entry = local_state / name
        if entry.exists() or entry.is_symlink():
            owned_path(plan, entry, must_exist=True)
            if not entry.is_file() or entry.stat().st_uid != os.geteuid():
                raise NamespaceError("resume recovery state must be an owned regular file")
    return {"source_runtime": str(source), "source": str(state.parent), "local": str(local_state)}


def build_command(plan, command, run_root=None, *, probe=False, resume_metadata=None):
    # Build a fresh small root rather than mounting / read-only then attempting
    # to create missing mount-point parents underneath a read-only host
    # mount. Every payload/local-source mount below remains read-only.
    result = [plan["bwrap"]["path"], "--unshare-user", "--die-with-parent", "--new-session"]
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt", "/run", "/sys"):
        if Path(name).exists():
            result += ["--ro-bind", name, name]
    # The controller must let the unchanged inner bubblewrap write its own
    # uid_map. A fresh procfs preserves the current PID namespace; only the
    # outer controller gains that kernel interface, while the original agent
    # sandbox still creates its original isolated proc layout.
    result += ["--proc", "/proc"]
    if Path("/dev").is_dir():
        result += ["--dev-bind", "/dev", "/dev"]
    result += ["--tmpfs", "/tmp", "--dir", "/root", "--dir", "/home", "--dir", "/mnt", "--dir", "/heart"]
    for root in sorted(plan["local_access_roots"], key=lambda value: len(PurePosixPath(value).parts)):
        result += ["--ro-bind", root, root]
    for row in plan["mounts"]:
        result += ["--ro-bind", row["local"], row["source"]]
    if run_root is not None:
        result += ["--bind", str(run_root), str(run_root)]
        # Model servers use the policy's original outputs/ spelling. Give
        # that path a private per-run output tree rather than write access
        # to the canonical source, old runs, environments or checkpoints.
        result += ["--bind", str(run_root / "policy-outputs"),
                   str(Path(plan["source_policy"]) / "outputs")]
        # The official writer retains its original directory spelling, while
        # data persists under the canonical local policy's per-run output.
        result += ["--bind", str(run_root / "eval_result"),
                   str(Path(plan["source_workspace"]) / "eval_result")]
        for name, value in (("ROBORSI_RUNTIME_STATE_DIR", run_root / "state"),
                            ("ROBORSI_TRAJECTORY_DIR", run_root / "trajectories"),
                            ("TMPDIR", run_root / "tmp"), ("XDG_CACHE_HOME", run_root / "cache")):
            result += ["--setenv", name, str(value)]
    if resume_metadata is not None:
        result += ["--bind", resume_metadata["local"], resume_metadata["source"]]
    result += ["--unsetenv", NATIVE_KIT_EXTENSION_ENV, "--unsetenv", NATIVE_KIT_PORTABLE_ENV]
    if run_root is not None:
        extension_parent = native_kit_extension_parent(plan)
        if extension_parent is not None:
            result += ["--setenv", NATIVE_KIT_EXTENSION_ENV, extension_parent]
        portable_root = native_kit_portable_root(plan, run_root)
        if portable_root is not None:
            result += ["--setenv", NATIVE_KIT_PORTABLE_ENV, portable_root]
    result += ["--setenv", "PYTHONDONTWRITEBYTECODE", "1",
               "--setenv", "PYTHONNOUSERSITE", "1",
               "--setenv", "PATH", plan["conda_root"] + "/bin:" + plan["conda_root"] + "/condabin:"
               + os.environ.get("PATH", "/usr/bin:/bin"),
               "--setenv", "CONDA_EXE", plan["conda_root"] + "/bin/conda",
               "--setenv", "_CONDA_EXE", plan["conda_root"] + "/bin/conda",
               "--setenv", "CONDA_PYTHON_EXE", plan["conda_root"] + "/bin/python",
               "--unsetenv", "CONDA_PREFIX", "--unsetenv", "CONDA_SHLVL",
               "--unsetenv", "_CE_CONDA", "--unsetenv", "_CE_M",
               "--setenv", "PYTHONPATH", plan["source_workspace"] + os.pathsep
               + str(Path(plan["source_workspace"]) / "XPolicyLab"),
               "--chdir", plan["source_workspace"], "--", *command]
    return result


def run_bounded(command, timeout):
    # No command line or inherited environment is printed. Kill only this
    # newly created process group on timeout, leaving all remote tests alone.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise NamespaceError("local namespace identity probe timed out") from None


def probe(plan):
    rows = []
    program = ("import json,platform,sys; print(json.dumps({'python':platform.python_version(),"
               "'architecture':platform.machine(),'prefix':sys.prefix,'base_prefix':sys.base_prefix}))")
    probes = plan.get("probes") or [{"name": "controller", "python": plan["controller_python"]}]
    for row in probes:
        python = str(absolute(row["python"]))
        code, stdout, stderr = run_bounded(build_command(plan, [python, "-I", "-S", "-c", program]), 30)
        if code:
            # Runtime loader errors contain software paths, not model/key body.
            rows.append({"name": row["name"], "status": "failed", "exit_code": code,
                         "loader_diagnostic": stderr[-1000:]})
            continue
        try:
            result = json.loads(stdout.strip())
        except ValueError:
            raise NamespaceError("identity probe did not return the expected standard-library metadata") from None
        passed = result.get("architecture") == "x86_64" and (
            not row.get("expected_version") or result.get("python") == row["expected_version"])
        rows.append({"name": row["name"], "status": "passed" if passed else "failed", **result,
                     "expected_version": row.get("expected_version")})
    return {"status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
            "probes": rows, "installed_bytes_modified": False,
            "models_or_simulator_started": False, "gpu_inference_verified": False}


def check_links(plan):
    roots = plan.get("closure_roots") or [row["source"] for row in plan["mounts"]]
    for root in roots:
        absolute(root)
        if not any(Path(root).is_relative_to(Path(row["source"])) for row in plan["mounts"]):
            raise NamespaceError("link closure root is not covered by an explicit payload mount")
    program = '''import json,os,sys
counts={'roots':0,'entries':0,'symlinks':0,'broken_links':0};broken=[]
for root in json.loads(sys.argv[1]):
 if not os.path.isdir(root):
  broken.append({'path':root,'reason':'root_missing'});counts['broken_links']+=1;continue
 counts['roots']+=1
 for directory,names,files in os.walk(root,followlinks=False):
  names[:]=[n for n in names if n not in {'__pycache__','.git','outputs','.validation','original-by-sha256'}]
  for name in names+files:
   path=os.path.join(directory,name);counts['entries']+=1
   if os.path.islink(path):
    counts['symlinks']+=1
    if not os.path.exists(path):
     counts['broken_links']+=1
     if len(broken)<30:broken.append({'path':path,'target':os.readlink(path),'reason':'unresolved_or_cyclic'})
print(json.dumps(dict(counts,broken=broken)))
'''
    code, stdout, stderr = run_bounded(build_command(plan, [plan["controller_python"], "-I", "-S", "-c", program,
                                                          json.dumps(roots)]), 600)
    if code:
        raise NamespaceError("namespace link closure scan failed: " + stderr[-1000:])
    result = json.loads(stdout.strip())
    return {"status": "passed" if result["broken_links"] == 0 else "failed", **result,
            "installed_bytes_modified": False, "models_or_simulator_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("namespace-plan.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--check-links", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true", help="run the unchanged resume_eval.py with the local plan's simulator prefix")
    mode.add_argument("--python", action="store_true", help="run controller Python; relative .py entry resolves inside this policy")
    mode.add_argument("--exec", action="store_true", help="run an explicit command inside the same prefix namespace")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume-request", type=Path)
    parser.add_argument("--resume-check-only", action="store_true")
    parser.add_argument("--resume-state-source", type=Path,
                        help="only with --python -m this-policy.hybrid_resume: make the selected owner's adapter-recovery metadata writable")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    if args.prepare_only:
        print(json.dumps({"status": "layout_checked", "source_policy": plan["source_policy"],
                          "canonical_policy": plan["policy_dir"], "mounts": len(plan["mounts"]),
                          "code_overlay": "bind current XPolicyLab onto the original native workspace path",
                          "installed_bytes_modified": False, "models_or_simulator_started": False}, sort_keys=True))
        return 0
    if args.probe:
        result = probe(plan)
        print(json.dumps(result, sort_keys=True))
        layout.atomic_json(args.plan.parent / "local-identity-probe.json", result)
        return 0 if result["status"] == "passed" else 1
    if args.check_links:
        result = check_links(plan)
        print(json.dumps(result, sort_keys=True))
        layout.atomic_json(args.plan.parent / "local-link-closure.json", result)
        return 0 if result["status"] == "passed" else 1
    evaluation = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    if (args.run or args.resume) and len(evaluation) != 9:
        parser.error("--run/--resume requires the first 9 eval.sh arguments after --; simulator prefix is bound by this plan")
    if (args.run or args.resume and not args.resume_check_only) and not os.environ.get("CODEX_API_KEY"):
        parser.error("set CODEX_API_KEY before evaluation")
    if args.resume and args.resume_request is None:
        parser.error("--resume requires --resume-request")
    if args.resume_state_source is not None and not args.python:
        parser.error("--resume-state-source is only valid with --python")
    if not args.run and not evaluation:
        parser.error("--python/--exec requires an explicit command or Python argument after --")
    policy = Path(plan["policy_dir"])
    run_root = args.run_root.absolute() if args.run_root else policy / "outputs" / (
        "local-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    if not run_root.is_relative_to(policy / "outputs") or run_root.is_symlink():
        raise NamespaceError("run root must be under the canonical local policy's outputs")
    for name in ("state", "trajectories", "tmp", "cache", "eval_result", "policy-outputs"):
        (run_root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    resume_metadata = (resume_metadata_mount(plan, args.resume_state_source, evaluation)
                       if args.resume_state_source is not None else None)
    shell = '''set -euo pipefail
export PATH="$1/bin:$1/condabin:$PATH"
set +u
source "$1/etc/profile.d/conda.sh"
set -u
shift
exec bash "$@"
'''
    if args.run:
        command = ["/bin/bash", "-c", shell, "local-roborsi", plan["conda_root"],
                   str(Path(plan["source_policy"]) / "eval.sh"), *evaluation, plan["simulator_prefix"]]
    elif args.resume:
        request = args.resume_request.absolute()
        if not request.is_file() or not request.resolve().is_relative_to(run_root.resolve()):
            raise NamespaceError("resume request must be an existing file under --run-root so its claim remains writable")
        command = [plan["controller_python"], str(Path(plan["source_policy"]) / "resume_eval.py"),
                   "--request", str(request)]
        if args.resume_check_only:
            command.append("--check-only")
        command += ["--", *evaluation, plan["simulator_prefix"]]
    elif args.python:
        if evaluation[0].endswith(".py") and not Path(evaluation[0]).is_absolute():
            entry = Path(evaluation[0])
            if ".." in entry.parts or not (policy / entry).is_file():
                raise NamespaceError("relative Python entry must be an existing policy file")
            evaluation[0] = str(Path(plan["source_policy"]) / entry)
        command = [plan["controller_python"], *evaluation]
    else:
        command = evaluation
    local_receipt = {"schema": 1, "plan_sha256": layout.sha256(args.plan), "run_root": str(run_root),
                     "source_workspace": plan["source_workspace"], "policy_dir": str(policy),
                     "installed_bytes_modified": False, "native_scoring_unchanged": True,
                     "operation": "eval" if args.run else "resume_eval" if args.resume
                     else "controller_python" if args.python else "explicit_command"}
    if plan.get("native_kit_extension_parent") is not None:
        local_receipt["native_kit_extension_parent"] = plan["native_kit_extension_parent"]
    if plan.get("native_kit_portable_per_run", False):
        local_receipt["native_kit_portable_root"] = str(run_root / "cache/native-kit")
    if resume_metadata is not None:
        local_receipt["writable_resume_metadata"] = resume_metadata
    layout.atomic_json(run_root / "local-launch.json", local_receipt)
    return subprocess.call(build_command(plan, command, run_root,
                                         resume_metadata=resume_metadata), cwd=policy)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (NamespaceError, layout.LayoutError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "exception": type(error).__name__,
                          "reason": str(error)}), file=sys.stderr)
        raise SystemExit(2)
