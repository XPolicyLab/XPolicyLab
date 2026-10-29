#!/usr/bin/env python3
"""Evaluate a LIBERO model: a model server and a simulator client in two environments.

PrismWAM environment:  python scripts/libero/eval.py serve --model libero --port 8001
LIBERO environment:    python scripts/libero/eval.py run --port 8001 --output outputs/libero_eval

`run` evaluates libero_spatial, libero_object, libero_goal and libero_10 (50 episodes
per task), appends one line per episode to <output>/episodes.jsonl, skips episodes
already recorded there, and writes per-suite success rates to <output>/summary.json.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_10": 700}  # other suites: 400
SETTLE_STEPS = 30
RESOLUTION = 256
DEPTH_KEYS = ("agentview_depth", "robot0_eye_in_hand_depth")
RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def serve(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from prismwam import load_pretrained
    from prismwam.integrations.libero import LiberoPolicy

    policy = LiberoPolicy(load_pretrained(args.model, device=args.device))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/reset":
                policy.reset(json.loads(body)["instruction"])
                return self.respond({"replan_steps": policy.replan_steps})
            if self.path == "/observe":
                data = np.load(io.BytesIO(body))
                policy.observe(
                    data["rgb"], data["depth"], data["state"], int(data["step"])
                )
                return self.respond({})
            if self.path == "/act":
                return self.respond({"action": policy.act(seed=args.seed).tolist()})
            self.send_error(404)

    print(f"LIBERO policy ready on http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


class Client:
    def __init__(self, port):
        import urllib.request

        self.url = f"http://127.0.0.1:{port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.request = urllib.request.Request

    def post(self, path, body=b"", content_type="application/json"):
        request = self.request(
            self.url + path, data=body, headers={"Content-Type": content_type}
        )
        with self.opener.open(request, timeout=600) as response:
            return json.loads(response.read())

    def reset(self, instruction):
        return self.post("/reset", json.dumps({"instruction": instruction}).encode())

    def observe(self, observation):
        buffer = io.BytesIO()
        np.savez(buffer, **observation)
        self.post("/observe", buffer.getvalue(), "application/octet-stream")

    def act(self):
        return np.asarray(self.post("/act")["action"], dtype=np.float64)


def run(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map
    from robosuite.utils.transform_utils import quat2axisangle

    def observation(obs, env, step):
        depth = []
        for key in DEPTH_KEYS:
            raw = np.asarray(obs[key], dtype=np.float32).squeeze()
            valid = np.isfinite(raw) & (raw >= 0) & (raw <= 1)
            metres = get_real_depth_map(
                env.sim, np.where(valid, raw, 1.0).astype(np.float32)
            )
            depth.append(np.where(valid, metres, 0.0).astype(np.float32))
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        )
        return {
            "rgb": np.stack([np.ascontiguousarray(obs[key]) for key in RGB_KEYS]),
            "depth": np.stack(depth),
            "state": state.astype(np.float32),
            "step": np.int64(step),
        }

    def episode(env, init_state, instruction, limit):
        env.reset()
        obs = env.set_init_state(init_state)
        client.reset(instruction)
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        steps = 0
        client.observe(observation(obs, env, steps))
        while steps < limit:
            for action in client.act():
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                if done:
                    return True, steps
                if steps >= limit:
                    break
                client.observe(observation(obs, env, steps))
        return False, steps

    client = Client(args.port)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "episodes.jsonl"
    rows = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    done = {(row["suite"], row["task_id"], row["episode"]) for row in rows}
    bddl_root = Path(get_libero_path("bddl_files"))
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task_ids = args.tasks if args.tasks is not None else range(suite.n_tasks)
        for task_id in task_ids:
            pending = [
                e for e in range(args.episodes) if (suite_name, task_id, e) not in done
            ]
            if not pending:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            random.seed(args.seed)
            np.random.seed(args.seed)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_root / task.problem_folder / task.bddl_file),
                camera_heights=RESOLUTION,
                camera_widths=RESOLUTION,
                camera_depths=True,
            )
            env.seed(args.seed)
            try:
                for index in pending:
                    success, steps = episode(
                        env,
                        init_states[index],
                        task.language,
                        MAX_STEPS.get(suite_name, 400),
                    )
                    row = {
                        "suite": suite_name,
                        "task_id": task_id,
                        "task": task.language,
                        "episode": index,
                        "success": success,
                        "steps": steps,
                    }
                    rows.append(row)
                    with log.open("a") as file:
                        file.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
            finally:
                env.close()

    summary = {}
    for suite_name in SUITES:
        results = [row["success"] for row in rows if row["suite"] == suite_name]
        if results:
            summary[suite_name] = {
                "episodes": len(results),
                "success_rate": 100 * sum(results) / len(results),
            }
    if rows:
        summary["all"] = {
            "episodes": len(rows),
            "success_rate": 100 * sum(row["success"] for row in rows) / len(rows),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="model server (PrismWAM environment)")
    server.add_argument("--model", default="libero", help="model folder, or libero")
    server.add_argument("--device", default="cuda:0")
    server.add_argument("--port", type=int, default=8001)
    server.add_argument("--seed", type=int, default=42, help="denoising noise seed")
    client = commands.add_parser("run", help="simulator client (LIBERO environment)")
    client.add_argument("--port", type=int, default=8001)
    client.add_argument("--output", required=True)
    client.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    client.add_argument("--tasks", nargs="+", type=int, help="task ids; default: all")
    client.add_argument("--episodes", type=int, default=50, help="episodes per task")
    client.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    serve(args) if args.command == "serve" else run(args)


if __name__ == "__main__":
    main()
