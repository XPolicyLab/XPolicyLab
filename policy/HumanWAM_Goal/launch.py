"""Resolve one HumanWAM-Goal run profile and start the policy server or the native RoboTwin client.

``server`` starts XPolicyLab's standard ``setup_policy_server.py`` with this
policy's ``deploy.yml`` plus the resolved release paths, preloading the pinned
cuBLASLt library (``model.py`` verifies the mapped copy). ``client`` runs the
pinned native RoboTwin evaluator through ``official_native``: by default
(``HWGOAL_SEED_MODE=fixed``) it replays the task's 100 released cases;
``HWGOAL_SEED_MODE=native`` keeps the native expert-check seed selection.
``HWGOAL_CHECK_ONLY=1`` prints the resolved profile without starting either.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml


POLICY_DIR = Path(__file__).resolve().parent
# The XPolicyLab checkout (used for its scripts and utils), not a sys.path root.
XPL_ROOT = POLICY_DIR.parents[1]
TASK_CONFIGS = ("demo_clean", "demo_randomized")
# Installed by install.sh; the profile ships with the released code.
DEFAULT_REPO = POLICY_DIR / "OpenHumanWAM"
PROFILE = "humanwam_goal/configs/robotwin.yaml"


def env(name, default=None):
    value = os.environ.get(name, "").strip()
    return value or default


def local(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else (POLICY_DIR / path).resolve())


def eval_env_type():
    value = env("EVAL_ENV_TYPE", "sim")
    if value not in ("sim", "debug"):
        raise ValueError(f"HumanWAM-Goal supports EVAL_ENV_TYPE=sim or debug, not {value!r}")
    return value


def resolve(args):
    """One profile for server and client; environment variables override deploy.yml."""
    if args.bench != "RoboTwin" or args.action != "joint" or args.seed != 0:
        raise ValueError("HumanWAM-Goal's RoboTwin release supports bench RoboTwin, action_type joint and seed 0")
    if args.env_cfg not in ("aloha_agilex", "arx_x5"):
        raise ValueError("Expected the Aloha-AgileX wire configuration (aloha_agilex; arx_x5 is equivalent)")
    deploy = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
    release = json.loads((POLICY_DIR / "release.json").read_text())
    task_config = env("HWGOAL_TASK_CONFIG", "demo_clean")
    if task_config not in TASK_CONFIGS:
        raise ValueError(f"HWGOAL_TASK_CONFIG must be one of {TASK_CONFIGS}")
    entry = release["checkpoints"].get(args.checkpoint, {})
    checkpoint = env("HWGOAL_CHECKPOINT", deploy.get("checkpoint_path") or entry.get("path"))
    repo = Path(env("HWGOAL_REPO", str(DEFAULT_REPO)))
    profile = env("HWGOAL_PROFILE", deploy.get("robotwin_profile") or str(repo / PROFILE))
    cache_sha = env(
        "HWGOAL_GOAL_CACHE_SHA256",
        deploy.get("official_goal_cache_sha256") or release["goal_cache"]["manifest_sha256"],
    )
    cublaslt = release["runtime"]["cublaslt"]
    output = env("HWGOAL_OUTPUT", deploy.get("output")) or str(
        POLICY_DIR / "eval_result" / args.task / task_config / time.strftime("%Y%m%d-%H%M%S")
    )
    mode = eval_env_type()
    server = dict(
        policy_name="HumanWAM_Goal",
        protocol="ws",
        host=args.host,
        port=args.port,
        bench_name=args.bench,
        task_name=args.task,
        ckpt_name=args.checkpoint,
        env_cfg_type=args.env_cfg,
        action_type=args.action,
        seed=args.seed,
        eval_batch=False,
        eval_env_type=mode,
        checkpoint_path=local(checkpoint),
        official_goal_cache=local(env("HWGOAL_GOAL_CACHE", deploy["official_goal_cache"])),
        official_goal_cache_sha256=cache_sha,
        robotwin_profile=local(profile),
        cublaslt_library=local(env("HWGOAL_CUBLASLT", deploy.get("cublaslt_library") or cublaslt["path"])),
        cublaslt_sha256=cublaslt["sha256"],
        output=str(Path(output).resolve() / "server"),
    )
    seed_mode = env("HWGOAL_SEED_MODE", "fixed")
    if seed_mode not in ("fixed", "native"):
        raise ValueError("HWGOAL_SEED_MODE must be fixed or native")
    case_range = env("HWGOAL_FIXED_CASES")
    if case_range is not None:
        from official_native import parse_case_range

        if seed_mode != "fixed":
            raise ValueError("HWGOAL_FIXED_CASES applies to HWGOAL_SEED_MODE=fixed")
        parse_case_range(case_range)
    robotwin_root = Path(env("ROBOTWIN_ROOT", str(XPL_ROOT.parent))).resolve()
    native = dict(
        task_name=args.task,
        task_config=task_config,
        policy_name="HumanWAM_Goal",
        ckpt_setting=args.checkpoint,
        policy_server_url=f"ws://{args.host}:{args.port}",
        xpolicylab_root=str(XPL_ROOT),
    )
    client = dict(
        robotwin_root=str(robotwin_root),
        native=native,
        cache_manifest_sha256=cache_sha,
        seed_mode=seed_mode,
        # Fixed mode reads the released cases from the goal cache's pinned manifest.
        goal_cache=server["official_goal_cache"],
        case_range=case_range,
        output=str(Path(output).resolve() / "native-client"),
    )
    return server, client


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_cublaslt(server):
    path = Path(server["cublaslt_library"])
    if not path.is_file():
        raise FileNotFoundError(f"Pinned cuBLASLt library is missing: {path} (run install.sh or set HWGOAL_CUBLASLT)")
    if file_sha256(path) != server["cublaslt_sha256"]:
        raise ValueError(f"{path} differs from the cuBLASLt pinned in release.json")


def preload_cublaslt(server):
    """The dynamic loader reads LD_PRELOAD at process start, so set it before exec."""
    path = Path(server["cublaslt_library"])
    if not path.is_file():
        raise FileNotFoundError(f"Pinned cuBLASLt library is missing: {path} (run install.sh or set HWGOAL_CUBLASLT)")
    existing = os.environ.get("LD_PRELOAD")
    os.environ["LD_PRELOAD"] = str(path) + (":" + existing if existing else "")


def check(server, client, policy_side=True):
    """CPU-only checks; no weights, simulator, cache entry or network."""
    from metadata_layout import validate_runtime_lookup
    from official_native import build_arguments, checked_source, fixed_cases, require_sha256

    if policy_side:
        check_cublaslt(server)
    if server["eval_env_type"] == "sim":
        require_sha256(client["cache_manifest_sha256"])
        validate_runtime_lookup(XPL_ROOT, client["robotwin_root"])
        checked_source(client["robotwin_root"])
        build_arguments(client["native"], client["robotwin_root"], client["seed_mode"])
        for key in ("official_goal_cache", "robotwin_profile"):
            if not Path(server[key]).exists():
                raise FileNotFoundError(f"{key}: {server[key]} (run download_checkpoint.py)")
        if client["seed_mode"] == "fixed":
            native = client["native"]
            cases = fixed_cases(
                client["goal_cache"], client["cache_manifest_sha256"],
                native["task_config"], native["task_name"], client["case_range"],
            )
            client = dict(client, fixed_cases=[case["case_id"] for case in cases])
    return dict(
        check_only=True,
        server=server,
        client=client,
        limits="CPU source/config checks only; no simulator or weights",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["server", "client", "check"])
    for name in ("bench", "task", "checkpoint", "env_cfg", "action"):
        parser.add_argument(name)
    parser.add_argument("seed", type=int)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1,65535]")
    server, client = resolve(args)
    if args.mode == "check" or env("HWGOAL_CHECK_ONLY") == "1":
        # The client machine of a split deployment does not need the policy runtime.
        print(json.dumps(check(server, client, policy_side=args.mode != "client"), indent=2))
    elif args.mode == "server":
        preload_cublaslt(server)
        argv = [
            sys.executable,
            str(XPL_ROOT / "setup_policy_server.py"),
            "--config_path",
            str(POLICY_DIR / "deploy.yml"),
            "--overrides",
        ]
        argv.extend(f"{key}={value!r}" for key, value in server.items())
        os.execv(sys.executable, argv)
    else:
        from official_native import run_native

        check(server, client, policy_side=False)
        run_native(
            client["robotwin_root"],
            client["native"],
            client["cache_manifest_sha256"],
            Path(client["output"]),
            client["seed_mode"],
            client["goal_cache"],
            client["case_range"],
        )


if __name__ == "__main__":
    main()
