#!/usr/bin/env python3
"""Serve RoboDojo's ARX X5 kinematics over HTTP, without the simulator: FK, IK and
planned paths through RoboDojo's own cuRobo planner, as the runtime worker answers
them (benches/robodojo/kinematics.py), for policies that run outside the simulator
(the XPolicyLab connector).

    CUDA_VISIBLE_DEVICES=5 python services/servers/serve_kinematics.py \
        --robodojo /path/to/RoboDojo --embodiment embodiments/arx_x5/embodiment.toml --port 8840

POST /fk   {"command": [14 joints]}                         -> {arm: link6 pose}
POST /ik   {"arm", "pose", "current": [6 joints]}           -> {"success", "joints"}
POST /plan {"arm", "current": [6], "pose" | "joints", "raw"} -> {"success", "waypoints"}
GET  /health
Poses are [x, y, z, qw, qx, qy, qz] of link6 in the env frame. One request at a time.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robodojo", required=True, help="RoboDojo checkout (its planner code and robot assets)")
    parser.add_argument("--embodiment", default=str(ROOT / "embodiments" / "arx_x5" / "embodiment.toml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8840)
    args = parser.parse_args()

    from benches.robodojo import kinematics

    with open(args.embodiment, "rb") as handle:
        embodiment = tomllib.load(handle)
    groups = [{"name": name, "joints": group["joints"], "base": embodiment["arm_bases"][name]}
              for name, group in embodiment["groups"].items()]
    interval = int(embodiment["physics_steps_per_tick"])
    dt = 1.0 / (float(embodiment["control_hz"]) * interval)
    planner = kinematics.standalone_planner(args.robodojo, robot=embodiment.get("robot_model", "x5"),
                                            base_z=float(groups[0]["base"][2]), dt=dt)
    by_name = {group["name"]: group for group in groups}
    lock = threading.Lock()

    def answer(route, body):
        if route == "/fk":
            return kinematics.forward(lambda group: planner, groups, body["command"])
        group = by_name[body["arm"]]
        current = body["current"]
        if len(current) != len(group["joints"]):
            current = [current[i] for i in group["joints"]]
        if route == "/ik":
            return kinematics.inverse(planner, group, current, body["pose"])
        if route == "/plan":
            return kinematics.plan(planner, group, current, interval, pose=body.get("pose"),
                                   joints=body.get("joints"), raw=bool(body.get("raw")))
        raise ValueError(f"unknown route {route}")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *arguments):
            pass

        def _reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(200, {"ok": True, "arms": list(by_name)})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                with lock:
                    self._reply(200, {"result": answer(self.path, body)})
            except Exception as error:
                self._reply(400, {"error": f"{type(error).__name__}: {error}"})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"kinematics of {args.embodiment} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
