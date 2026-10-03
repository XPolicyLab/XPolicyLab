"""Owned same-machine XPolicyLab evaluation; no alternate RSI/selection loop."""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from PhysicalRSI_core.infra.processes import ManagedProcess
from PhysicalRSI_core.infra.storage import atomic_json, read_json

from .policy_service import PolicyService, verify
from .expert_model import websocket_client


def load_release(policy_dir, arguments):
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    fields = (
        "bench_name",
        "task_name",
        "ckpt_name",
        "env_cfg_type",
        "action_type",
        "seed",
    )
    config = dict(zip(fields, arguments[:6]))
    root = resolve_checkpoint_root(
        config, policy_dir / "checkpoints", policy_dir=policy_dir
    )
    spec = read_json(root / "service.json")
    verify(spec)
    if spec.get("model_adapter") != "xpolicylab":
        raise ValueError("Release must select the xpolicylab model adapter")
    for key in ("bench_name", "task_name", "env_cfg_type", "action_type", "seed"):
        if str(spec["config"].get(key)) != config[key]:
            raise ValueError(f"Release configuration differs from eval argument: {key}")
    return spec


def client_command(policy_dir, arguments, endpoint, *, mode):
    xpl = policy_dir.parent.parent
    address = urlparse(endpoint)
    if address.scheme != "ws" or address.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Same-machine evaluation requires a loopback WS endpoint")
    if mode == "debug":
        env = arguments[9]
        executable = Path(env).expanduser() / "bin/python"
        if executable.is_file():
            # Preserve virtualenv interpreter symlinks rather than resolving them.
            prefix = [str(executable.absolute())]
        elif env == "base":
            prefix = [sys.executable]
        else:
            prefix = ["conda", "run", "--no-capture-output", "-n", env, "python"]
        return prefix + [
            str(xpl / "utils/debug_env_client.py"),
            "--bench_name",
            arguments[0],
            "--task_name",
            arguments[1],
            "--env_cfg_type",
            arguments[3],
            "--policy_name",
            "physicalRSI",
            "--host",
            address.hostname,
            "--port",
            str(address.port),
            "--eval_batch",
            "true",
            "--eval_episode_num",
            os.environ.get("PHYSICALRSI_DEBUG_EPISODES", "1"),
        ]
    if mode != "sim":
        raise ValueError("EVAL_ENV_TYPE must be debug or sim")
    return [
        "bash",
        str(xpl / "utils/setup_env_client.sh"),
        str(xpl / "utils"),
        str(policy_dir / "deploy.yml"),
        arguments[9],
        str(address.port),
        arguments[0],
        arguments[1],
        arguments[3],
        "physicalRSI",
        "physicalrsi_committed_evaluation",
        str(xpl.parent),
        arguments[5],
        arguments[7],
        address.hostname,
    ]


def run_client(command, *, root, environment, cancelled, timeout_s=3600):
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Positive finite client deadline required")
    started = time.monotonic()
    client = ManagedProcess(
        "xpolicylab-client",
        command,
        env_overrides=environment,
        log_path=root / "client.log",
    )
    try:
        if cancelled():
            raise InterruptedError("Evaluation cancelled")
        client.start()
        atomic_json(root / "client-process.json", dict(pid=client.pid))
        while client.poll() is None:
            if cancelled():
                raise InterruptedError("Evaluation cancelled")
            if time.monotonic() - started >= timeout_s:
                raise TimeoutError("Evaluation deadline exceeded")
            time.sleep(0.05)
        if client.poll() != 0:
            raise RuntimeError("Environment client failed; inspect client.log")
    finally:
        client.stop()
        atomic_json(
            root / "client-cleanup.json",
            dict(state="stopped", returncode=client.poll()),
        )


def attach_client(spec, *, endpoint, output, command, environment, cancelled):
    """Check the external owned server before starting this client's environment.

    This checks accidental endpoint/release mismatches on the same machine; it
    is not authentication or ownership of the separately managed server.
    """
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    client = None
    try:
        expected = verify(spec)
        if cancelled():
            raise InterruptedError("Evaluation cancelled")
        client = websocket_client(endpoint=endpoint, timeout_s=10)
        identity = client.call(func_name="physicalrsi_service_identity")
        if identity != {"service_identity": expected}:
            raise ValueError("Policy endpoint differs from requested release")
        atomic_json(root / "attachment.json", dict(endpoint=endpoint, **identity))
        client.close()
        client = None
        run_client(command, root=root, environment=environment, cancelled=cancelled)
        atomic_json(
            root / "result.json",
            dict(
                state="client_completed",
                server_cleanup="external_owner",
                physical_qualification=False,
            ),
        )
    except BaseException as error:
        atomic_json(
            root / "failed.json", dict(error=str(error), physical_qualification=False)
        )
        raise
    finally:
        if client is not None:
            client.close()


def evaluate(
    spec,
    *,
    output,
    command,
    environment,
    cancelled,
    timeout_s=3600,
    client_environment=None,
):
    """`command` constructs client argv from the actual bound service endpoint."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        with PolicyService(
            spec,
            python=sys.executable,
            environment=environment,
            output=root / "server",
            cancelled=cancelled,
        ) as service:
            run_client(
                command(service.ready["endpoint"]),
                root=root,
                environment=environment
                if client_environment is None
                else client_environment,
                cancelled=cancelled,
                timeout_s=timeout_s,
            )
        atomic_json(
            root / "result.json", dict(state="completed", physical_qualification=False)
        )
    except BaseException as error:
        atomic_json(
            root / "failed.json", dict(error=str(error), physical_qualification=False)
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    standalone = parser.add_mutually_exclusive_group()
    standalone.add_argument("--server-only", action="store_true")
    standalone.add_argument("--client-endpoint")
    parser.add_argument("arguments", nargs=10)
    args = parser.parse_args()
    directory = args.policy_dir.resolve()
    spec = load_release(directory, args.arguments)
    mode = os.environ.get("EVAL_ENV_TYPE", "sim")
    if mode not in {"debug", "sim"}:
        raise ValueError("EVAL_ENV_TYPE must be debug or sim")
    if mode == "sim" and spec["config"]["physicalrsi"]["mode"] != "committed":
        raise ValueError(
            "Candidate exploration must use the fresh-layout native runtime"
        )
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    xpl = directory.parent.parent
    environment = dict(
        PYTHONPATH=os.pathsep.join(
            [str(xpl.parent), str(xpl), os.environ.get("PYTHONPATH", "")]
        ),
        CUDA_VISIBLE_DEVICES=args.arguments[6],
    )
    # The simulator launcher consumes env_gpu_id itself; the policy CUDA mask
    # must not hide the independently requested environment GPU.
    client_environment = dict(PYTHONPATH=environment["PYTHONPATH"], EVAL_ENV_TYPE=mode)
    if args.server_only:
        with PolicyService(
            spec,
            python=sys.executable,
            environment=environment,
            output=args.output,
            cancelled=stopping.is_set,
        ) as service:
            print(json.dumps(service.ready), flush=True)
            while not stopping.wait(0.05):
                if service.process.poll() is not None:
                    raise RuntimeError("Policy service exited")
        return
    if args.client_endpoint:
        attach_client(
            spec,
            endpoint=args.client_endpoint,
            output=args.output,
            command=client_command(
                directory, args.arguments, args.client_endpoint, mode=mode
            ),
            environment=client_environment,
            cancelled=stopping.is_set,
        )
        return
    evaluate(
        spec,
        output=args.output,
        environment=environment,
        client_environment=client_environment,
        command=lambda endpoint: client_command(
            directory, args.arguments, endpoint, mode=mode
        ),
        cancelled=stopping.is_set,
    )


if __name__ == "__main__":
    main()
