"""Opt-in real WebSocket + Docker startup load, with zero model API calls.

From the checkout: policy/embodiedevo/.venv/bin/python -m \
    XPolicyLab.policy.embodiedevo.tests.run_server_load --clients 256 --run-dir /fresh/path
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import yaml

from client_server.ws.model_server import PolicyServerConfig
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.messages import MessageType
from XPolicyLab.policy.embodiedevo.runtime.files import write_json
from XPolicyLab.policy.embodiedevo.runtime.serving import TaskRoutingServer, close_models, prepare_models
from XPolicyLab.policy.embodiedevo.runtime.workspace import POLICY_DIR, tree_hashes
from XPolicyLab.policy.embodiedevo.tests.container_probe import run_container_probe
from XPolicyLab.utils.process_data import encode_image_bit


async def run(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
    cfg.update(server_tasks=["stack_bowls"], diagnostic=False)
    with patch.dict(os.environ, EMBODIEDEVO_RUN_DIR=str(root / "server"),
                    EMBODIEDEVO_CODEX_HOME=str(root / "unused-provider")):
        models, directory = prepare_models(cfg)
    server = TaskRoutingServer(models, PolicyServerConfig(host="127.0.0.1", port=0,
        max_message_bytes=64 * 1024 * 1024, max_receive_queue=2, max_pending_responses=4),
        run_dir=directory, max_active_agents=max(256, args.clients),
        disconnect_grace_s=2, startup_parallelism=16, startup_timeout_s=120)
    clients, heartbeats, lag, workers, names = [], [], [], [], []
    done = asyncio.Event()
    report = {"clients": args.clients, "full_episodes": 0, "model_api_calls": 0,
              "probe_network": "none", "probe_auth_mounted": False,
              "camera_shape": [480, 640, 3], "camera_count": 3}
    tasks, monitors = [], []

    async def client(identity):
        value = PolicyEvalClient(PolicyEvalClientConfig(url=server.url, evaluation_id=identity,
            request_timeout_s=120, max_connect_attempts=1, proxy=None))
        clients.append(value)
        await value.connect()
        return value

    async def heartbeat(observer):
        while not done.is_set():
            start = time.monotonic()
            await observer.request(MessageType.HEARTBEAT, {})
            heartbeats.append(time.monotonic() - start)
            await asyncio.sleep(0.2)

    async def event_loop():
        while not done.is_set():
            start = time.monotonic()
            await asyncio.sleep(0.05)
            lag.append(max(0, time.monotonic() - start - 0.05))

    rgb = np.full((480, 640, 3), [7, 20, 30], dtype=np.uint8)
    cameras = {name: {"color": encode_image_bit(rgb)}
               for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}

    async def launch(index):
        c = await client(f"load-{index}")
        case = "stack_bowls_case"
        await c.reset(trial_id="startup-only", action_case_id=case)
        await c.prepare_case(case, case_meta={"native_step_limit": 10,
            "environments": [{"env_idx": 0, "layout_id": index}]})
        state = {}
        dims = models["stack_bowls"].robot_dims
        prefixes = [""] if len(dims["arm_dim"]) == 1 else ["left_", "right_"]
        for prefix, arm, gripper in zip(prefixes, dims["arm_dim"], dims["ee_dim"], strict=True):
            state[prefix + "arm_joint_state"] = np.full(arm, float(index))
            state[prefix + "ee_joint_state"] = np.zeros(gripper)
            state[prefix + "ee_pose"] = np.array([0., 0., 0., 1., 0., 0., 0.])
        await c.call("update_obs", {"instruction": f"startup marker {index}",
                                    "state": state, "vision": cameras}, action_case_id=case)
        async with asyncio.timeout(args.timeout):
            while True:
                response = await c.call("get_action", action_case_id=case)
                if response.payload["result"]:
                    action = response.payload["result"][0]
                    assert all(np.array_equal(action[k], v) for k, v in state.items() if k.endswith("joint_state"))
                    break
        slot = server.sessions[(c.config.evaluation_id, case)]
        context = slot.model.contexts[0]
        workers.append(context.worker)
        marker = slot.model.run / "episodes/episode_0000/runtime.json"
        names.append(json.loads(marker.read_text())["container"])
        count = len(names)
        if count % 16 == 0 or count == args.clients:
            print(json.dumps({"first_actions": count, **server.status()}), flush=True)
        return c, marker.parent / "workspace"

    try:
        await server.start()
        observer = await client("observer")
        monitors = [asyncio.create_task(heartbeat(observer)), asyncio.create_task(event_loop())]
        started = time.monotonic()
        with patch("XPolicyLab.policy.embodiedevo.model.run_worker", run_container_probe):
            tasks = [asyncio.create_task(launch(i)) for i in range(args.clients)]
            launched = await asyncio.gather(*tasks)
        report.update(first_actions=len(launched), startup_seconds=time.monotonic() - started,
                      service_at_peak=server.status())
        inspected = await asyncio.to_thread(subprocess.run, ["docker", "inspect", *names],
                                            capture_output=True, text=True, check=True, timeout=60)
        containers = json.loads(inspected.stdout)
        report["concurrently_running_containers"] = sum(c["State"]["Running"] for c in containers)
        assert report["concurrently_running_containers"] == args.clients
        assert len({c["Id"] for c in containers}) == args.clients
        assert all(c["HostConfig"]["NetworkMode"] == "none" for c in containers)
        assert all(c["HostConfig"]["ReadonlyRootfs"] for c in containers)
        assert all(len(c["Mounts"]) == 9 for c in containers)  # workspace, socket, probe, six read-only components
        def knowledge(workspace):
            files = tree_hashes(workspace, components=("skills", "lessons", "instruction.md", "primitives", "scripts"))
            return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        hashes = await asyncio.gather(*(asyncio.to_thread(knowledge, ws) for _, ws in launched))
        assert len(set(hashes)) == 1
        report["identical_knowledge_workspaces"] = len(hashes)
        report["knowledge_sha256"] = hashes[0]
        cleanup_start = time.monotonic()
        # Exercise both explicit CLOSE and abrupt WebSocket loss at fleet scale.
        await asyncio.gather(*(c.close() if i % 2 == 0 else c._reset_connection_state()
                               for i, (c, _) in enumerate(launched)))
        async with asyncio.timeout(args.timeout):
            while server.sessions:
                await asyncio.sleep(0.1)
        report.update(cleanup_seconds=time.monotonic() - cleanup_start,
                      cleanup_reserved_agents=server.capacity.total,
                      surviving_worker_processes=sum(w.is_alive() for w in workers))
        checked = await asyncio.to_thread(subprocess.run, ["docker", "ps", "-a", "--format", "{{.Names}}"],
                                          capture_output=True, text=True, check=True, timeout=30)
        report["surviving_test_containers"] = len(set(names) & set(checked.stdout.splitlines()))
        assert report["cleanup_reserved_agents"] == report["surviving_worker_processes"] == report["surviving_test_containers"] == 0
        report["passed"] = True
    except BaseException as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        done.set()
        monitor_results = await asyncio.gather(*monitors, return_exceptions=True)
        report["monitor_errors"] = [str(r) for r in monitor_results if isinstance(r, BaseException)]
        if report["monitor_errors"]:
            report["passed"] = False
        report["heartbeat_samples"] = len(heartbeats)
        for label, values in (("heartbeat", heartbeats), ("event_loop_lag", lag)):
            report[label + "_p95_ms"] = float(np.percentile(values, 95) * 1000) if values else None
            report[label + "_max_ms"] = max(values, default=0) * 1000
        await asyncio.gather(*(c.close() for c in clients), return_exceptions=True)
        try:
            await server.stop()
        finally:
            close_models(models)
            write_json(root / "report.json", report)
            print(json.dumps(report, indent=2), flush=True)
        if report["monitor_errors"]:
            raise RuntimeError("Heartbeat monitor failed: " + "; ".join(report["monitor_errors"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=256)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.clients <= 0:
        parser.error("--clients must be positive")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
