"""Serve flash or memory checkpoints with isolated RoboDojo WebSocket sessions."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
POLICY = ROOT.parent
PROFILES = ("flash", "memory")


def default_config():
    import yaml
    return yaml.safe_load((POLICY / "wam_deploy_robodojo.yml").read_text())


def digest_config(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def online_config(baseline, profile=None, max_envs_per_session=1, *, host="0.0.0.0", port=18880):
    if type(max_envs_per_session) is not int or max_envs_per_session not in range(1, 11):
        raise ValueError("Supported batch capacities are integers 1-10")
    if profile is not None and profile not in PROFILES:
        raise ValueError("Expected flash or memory")
    return {**copy.deepcopy(baseline), "host": host, "port": port}


def assert_same_policy(baseline, actual, profile=None, max_envs_per_session=1):
    if actual != online_config(baseline, profile, max_envs_per_session):
        raise ValueError("Unexpected policy configuration changes")


def gpu_ids(value):
    ids = value.split(",")
    if not ids or any(not item.isdecimal() for item in ids):
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated non-negative integers")
    ids = [str(int(item)) for item in ids]
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("GPU IDs must be distinct")
    return ids


def check_gpus(ids, host, port):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=30).strip().splitlines()
    inventory = {row.split(",")[0].strip(): int(row.split(",")[1]) for row in rows}
    if any(gpu not in inventory for gpu in ids):
        raise RuntimeError("A selected GPU is unavailable")
    busy = subprocess.check_output([
        "nvidia-smi", "-i", ",".join(ids),
        "--query-compute-apps=pid,process_name", "--format=csv,noheader",
    ], text=True, timeout=30).strip()
    if busy or any(inventory[gpu] > 100 for gpu in ids):
        raise RuntimeError("Selected GPUs are occupied; no existing process has been stopped")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def checkpoint_config(config, profile):
    """Resolve shared deployment metadata without constructing GPU weights."""
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
    from ..xpolicylab_adapter.runtime_config import resolve_checkpoint_deployment_config
    checkpoint = resolve_checkpoint_root(config, POLICY / "checkpoints", policy_dir=POLICY)
    for name in ("config.json", "checkpoint_manifest.json", "transform_bundle.json", "provenance.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Checkpoint is missing {name}: {checkpoint}")
    native = json.loads((checkpoint / "config.json").read_text())["worldscape_policy_model_config"]["model"]
    core = native["wam"]["core"]["parameters"]
    expected = 1 if profile == "flash" else 24
    if core["max_chunk_size"] != expected or core["num_action_per_block"] != 24:
        raise ValueError(f"{profile} requires an H24/C{expected} checkpoint")
    if native["mode"] != "interactive":
        raise ValueError("This server supports interactive checkpoints")
    config = resolve_checkpoint_deployment_config(config, checkpoint)
    return {**config, "checkpoint_path": str(checkpoint)}


def setup_environment(run, config, ids):
    from ..xpolicylab_adapter.runtime_config import prepare_source
    native = prepare_source(config)
    config["worldscape_root"] = str(native)
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(ids), PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    # Spawned workers inherit the current adapter and native source selection.
    paths = [str(POLICY.parents[2]), str(POLICY.parents[1]), str(native), str(native / "src")]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    for key, subdir in {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "CUDA_CACHE_PATH": "cache/cuda",
        "TRITON_CACHE_DIR": "cache/triton", "MPLCONFIGDIR": "cache/matplotlib",
    }.items():
        path = run / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["WSP_SERVER_TRITON_ROOT"] = os.environ["TRITON_CACHE_DIR"]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--config", type=Path, default=POLICY / "wam_deploy_robodojo.yml")
    parser.add_argument("--checkpoint", help="Checkpoint directory; overrides checkpoint_path in YAML")
    parser.add_argument("--gpus", type=gpu_ids, default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18880)
    parser.add_argument("--envs-per-session", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--max-sessions", type=int, help="Default: one session per selected GPU")
    parser.add_argument("--session-idle-seconds", type=float, default=1800)
    parser.add_argument("--run-dir", type=Path, help="Parent for unique logs/config/cache run directories")
    parser.add_argument("--client-mode", choices=("official", "explicit"), default="official")
    parser.add_argument("--session-mode", choices=("isolated", "legacy"), default="isolated")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="CPU imports and checkpoint metadata check")
    mode.add_argument("--verify", metavar="WS_URL", help="Read-only running service identity check")
    from .replay import add_replay_arguments, replay_options, replay_settings
    add_replay_arguments(parser)
    args = parser.parse_args()
    try:
        recovery = replay_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1-65535")
    sessions = args.max_sessions if args.max_sessions is not None else len(args.gpus)
    if not 1 <= sessions <= len(args.gpus):
        parser.error("max-sessions must be between 1 and the number of selected GPUs")
    if not math.isfinite(args.session_idle_seconds) or args.session_idle_seconds <= 0:
        parser.error("session-idle-seconds must be finite and positive")
    if args.session_mode == "isolated" and args.client_mode != "official":
        parser.error("Isolated sessions require official client mode")
    if args.session_mode == "legacy" and args.envs_per_session != 1:
        parser.error("Legacy mode requires --envs-per-session 1")
    import yaml
    from ..xpolicylab_adapter.runtime_config import validate_config
    config = yaml.safe_load(args.config.read_text())
    if config.get("policy_name") != POLICY.name or config.get("runtime_profile") != "robodojo":
        parser.error("Use the WorldScape_RSI RoboDojo deployment config")
    if config.get("action_type") != "joint" or config.get("env_cfg_type") != "arx_x5":
        parser.error("This server supports arx_x5 / joint")
    if args.checkpoint:
        config["checkpoint_path"] = str(Path(args.checkpoint).expanduser().resolve())
    config["multi_gpu"] = {**(config.get("multi_gpu") or {}), "enabled": True,
                           "max_envs_per_gpu": args.envs_per_session}
    config["eval_batch"] = True
    config = checkpoint_config(validate_config(config), args.profile)
    config = online_config(config, args.profile, args.envs_per_session, host=args.host, port=args.port)
    runtime_root = args.run_dir or Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "worldscape-policy/server"
    runtime_root = runtime_root.expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{args.profile}-{socket.gethostname()}-", dir=runtime_root))
    setup_environment(run, config, args.gpus)
    print(f"ONLINE_RUN={run}", flush=True)
    from ..wam_model import Model
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from .official_client import OfficialClientMixin
    from .session_model import SessionPool
    from .session_server import SessionPolicyServer
    from .replay import authenticate_inspection, configure_logging
    identity = dict(profile=args.profile, config=config, config_sha256=digest_config(config),
                    client_mode=args.client_mode, session_mode=args.session_mode, gpu_ids=args.gpus,
                    max_sessions=sessions, max_envs_per_session=args.envs_per_session,
                    max_environments=sessions * args.envs_per_session,
                    replay=replay_settings(recovery) if args.session_mode == "isolated" else None)
    if args.verify:
        from client_server.ws.model_client import WsModelClient
        with WsModelClient(url=args.verify, evaluation_id="identity-check", trial_id="identity",
                           request_timeout_s=30, max_connect_attempts=1) as client:
            authenticate_inspection(client)
            info = client.call(func_name="online_info")
        for key in ("profile", "config_sha256", "client_mode", "session_mode", "gpu_ids", "max_sessions"):
            if info.get(key) != identity[key]:
                raise RuntimeError(f"Endpoint differs in {key}")
        print(json.dumps(info, indent=2))
        return
    (run / "server.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    save(run / "preflight.json", {**identity, "python": sys.executable, "gpu_model_loaded": False})
    if args.check:
        print("PASS: CPU preflight; no model loaded and no port bound", flush=True)
        return
    locks = []
    model = None
    try:
        # Serialize cooperative starts even when two services select overlapping GPUs.
        for gpu in sorted(args.gpus, key=int):
            lock = (runtime_root / f"{socket.gethostname()}-gpu-{gpu}.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_gpus(args.gpus, args.host, args.port)
        configure_logging()
        service_model = SessionPool if args.session_mode == "isolated" else (
            type("OfficialModel", (OfficialClientMixin, Model), {}) if args.client_mode == "official" else Model)

        class OnlineModel(service_model):
            def online_info(self):
                backend = getattr(self, "pool", getattr(self, "backend", self))
                backend._check()
                if any(not worker.process.is_alive() for worker in backend._workers):
                    raise RuntimeError("Policy worker is unavailable")
                return {**identity, "worker_count": len(backend._workers)}

        def stop(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        save(run / "status.json", {**identity, "status": "loading"})
        options = {"max_envs_per_session": args.envs_per_session} if args.session_mode == "isolated" else {}
        model = OnlineModel(config, **options)
        model.online_info()
        save(run / "effective_model_config.json", model.model_cfg)

        async def serve():
            cls = SessionPolicyServer if args.session_mode == "isolated" else PolicyServer
            opts = dict(max_sessions=sessions, idle_timeout=args.session_idle_seconds, **recovery) if args.session_mode == "isolated" else {}
            server = cls(model, PolicyServerConfig(host=args.host, port=args.port,
                         ws_ping_interval_s=config.get("ws_ping_interval_s", 20),
                         ws_ping_timeout_s=config.get("ws_ping_timeout_s", 1200)), **opts)
            await server.start()
            save(run / "status.json", {**identity, "status": "ready"})
            print(f"READY: ws://{args.host}:{args.port}; {len(args.gpus)} workers loaded", flush=True)
            try:
                await server.serve_forever()
            finally:
                await server.stop()
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
    finally:
        if model is not None:
            model.close()
        save(run / "status.json", {**identity, "status": "stopped"})
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()
