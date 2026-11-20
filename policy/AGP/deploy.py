"""AGP deploy hook: RoboDojo's eval client calls eval_one_episode(TASK_ENV, model_client) once per layout.

The episode runs in the packaged runtime of this folder (deploy.yml agp.runtime_root,
agp_runtime/agp_core/official_deploy.py); its interpreters and everything it writes are in AGP_HOME (default
~/.agp-robodojo). A failure to load it raises AgentLoadFailure, a BaseException, so that RoboDojo's main.py stops
instead of counting or skipping the layout; the process then exits with AGP_ABORT_EXIT_CODE (default 3), which
RoboDojo's restart loop does not retry.

With EVAL_ENV_TYPE=debug (XPolicyLab's debug closed loop) the runtime is not loaded: the episode is the plain loop of
policy/demo_policy/deploy.py against the stub server.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

POLICY_DIR = Path(__file__).resolve().parent
# deploy.yml agp.runtime_root when it is missing: the packaged runtime in this folder
DEFAULT_RUNTIME_ROOT = "agp_runtime"


class AgentLoadFailure(BaseException):
    """The AGP implementation could not be loaded (not an Exception: main.py must not skip the layout)."""


def _register(name: str, directory: Path):
    init = directory / "__init__.py"
    loaded = sys.modules.get(name)
    if loaded is not None:
        where = Path(getattr(loaded, "__file__", "") or "").resolve()
        if where != init.resolve():
            raise AgentLoadFailure(f"package {name!r} is already imported from {where}, expected {init}")
        return loaded
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise AgentLoadFailure(f"no package {name!r} at {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _pinned_root():
    """deploy.yml agp.runtime_root, else DEFAULT_RUNTIME_ROOT, as an absolute path: a leading ~ is the user's home,
    a relative value is relative to this folder (the rule of agp_core/paths.py deploy_pins)."""
    try:
        import yaml
        agp = (yaml.safe_load((POLICY_DIR / "deploy.yml").read_text()) or {}).get("agp") or {}
    except Exception as exc:
        raise AgentLoadFailure(f"cannot read {POLICY_DIR / 'deploy.yml'}: {type(exc).__name__}: {exc}") from exc
    value = os.path.expanduser(str(agp.get("runtime_root") or DEFAULT_RUNTIME_ROOT))
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(POLICY_DIR, value))


def _refuse_frozen(reason):
    """A frozen measurement that cannot run the frozen system: report it where the launcher looks, then abort."""
    import json
    print(f"[AGP] frozen system refused: {reason}", flush=True)
    results = os.environ.get("AGP_RESULTS_DIR")
    if results and os.path.isdir(results):
        try:
            Path(results, "freeze_refusal.json").write_text(json.dumps(
                {"kind": "freeze_mismatch", "error": reason, "by": "XPolicyLab/policy/AGP/deploy.py"}))
        except OSError:
            pass
    raise AgentLoadFailure(f"frozen system mismatch: {reason}")


ARM_CONTROLS = ("AGP_ARM_MANIFEST", "AGP_ARM_SEAL")     # an A/B arm's configuration (never frozen)


def _implementation():
    pinned = _pinned_root()
    caller = os.environ.get("OURS_ROOT")
    if (caller and os.environ.get("AGP_FROZEN_MEASUREMENT") == "1"
            and os.path.realpath(caller) != os.path.realpath(pinned)):
        _refuse_frozen(f"OURS_ROOT={caller} differs from deploy.yml agp.runtime_root {pinned}")
    arm = [name for name in ARM_CONTROLS if os.environ.get(name)]
    if arm and os.environ.get("AGP_FROZEN_MEASUREMENT") == "1":
        _refuse_frozen(f"{', '.join(arm)} set: an A/B arm's configuration is refused in a frozen measurement")
    root = Path(caller or pinned).resolve()
    try:
        for name in ("agp_core", "migration"):
            _register(name, root / name)
        return importlib.import_module("agp_core.official_deploy")
    except AgentLoadFailure:
        raise
    except Exception as exc:
        raise AgentLoadFailure(f"cannot load the AGP implementation from {root}: {type(exc).__name__}: {exc}") from exc


_WATCHDOG = """
import os, sys, time
pid, start, delay = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
def alive():
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19] == start
    except (OSError, IndexError):
        return False
deadline = time.monotonic() + delay
while alive() and time.monotonic() < deadline:
    time.sleep(1)
if alive():
    print(f"[AGP watchdog] pid {pid} still running {delay:.0f} s after the infrastructure abort "
          "(simulator shutdown hangs): SIGTERM, then SIGKILL", file=sys.stderr, flush=True)
    os.kill(pid, 15)
    time.sleep(10)
    if alive():
        os.kill(pid, 9)
"""


def _arm_abort_watchdog():
    """Start the separate watchdog process (see the module docstring); never raises."""
    try:
        pid = os.getpid()
        with open(f"/proc/{pid}/stat") as f:
            start = f.read().rsplit(")", 1)[1].split()[19]
        delay = float(os.environ.get("AGP_ABORT_EXIT_S") or 60)
        subprocess.Popen(["/usr/bin/python3", "-B", "-c", _WATCHDOG, str(pid), start, str(delay)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, start_new_session=True)
        print(f"[AGP] infrastructure abort: watchdog armed ({delay:.0f} s)", flush=True)
    except Exception as exc:  # the abort itself must not be masked
        print(f"[AGP] abort watchdog unavailable: {type(exc).__name__}: {exc}", flush=True)


def _is_infrastructure_abort(exc):
    return isinstance(exc, AgentLoadFailure) or type(exc).__name__ == "AgentInfrastructureFailure"


def _exit_after_report():
    """Wrap sys.excepthook: after the abort's traceback is printed, exit without the simulator's shutdown."""
    previous = sys.excepthook
    if getattr(previous, "_agp_abort_hook", False):
        return
    code = int(os.environ.get("AGP_ABORT_EXIT_CODE") or 3)

    def hook(exc_type, exc, tb):
        try:
            previous(exc_type, exc, tb)
        finally:
            if _is_infrastructure_abort(exc):
                print(f"[AGP] run aborted by {exc_type.__name__}; exiting with code {code} without the simulator's "
                      "shutdown", file=sys.stderr, flush=True)
                sys.stdout.flush()
                os._exit(code)
    hook._agp_abort_hook = True
    sys.excepthook = hook


def _prepare_abort():
    _arm_abort_watchdog()
    _exit_after_report()


def _debug_episode(TASK_ENV, model_client):
    """EVAL_ENV_TYPE=debug: the loop of policy/demo_policy/deploy.py (observation in, joint action chunk out)."""
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode(TASK_ENV, model_client):
    if os.environ.get("EVAL_ENV_TYPE") == "debug":
        _debug_episode(TASK_ENV, model_client)
        return
    try:
        _implementation().eval_one_episode(TASK_ENV, model_client, policy_dir=POLICY_DIR)
    except BaseException as exc:
        if _is_infrastructure_abort(exc):
            _prepare_abort()
        raise


def eval_one_episode_batch(TASK_ENV, model_client):
    _prepare_abort()
    raise AgentLoadFailure("AGP runs one environment per episode; keep eval_batch: false in deploy.yml")
