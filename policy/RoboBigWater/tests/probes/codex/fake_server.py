#!/usr/bin/env python3
"""Fake robo-server for M0: no simulator, synthetic images, scripted feedback.

It speaks the v0 contract so the real robo client can be tested against it,
and it draws images whose content is known exactly:

  * a step counter in large digits (does the agent see refreshed images?)
  * a red disc whose pixel position changes every command (which resolution
    does the model answer in?)

`robo wait <sec>` sleeps sec * --wait-scale real seconds (shell timeout probe).
"""

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont

BUDGET = 60
LOCK = threading.Lock()
STATE = {"step": 0, "budget_left": BUDGET, "over": True, "last": None, "log": [], "task": None}
ARGS = None


def disc_position(step, width, height):
    """Deterministic pseudo-random position, away from the borders."""
    u = int(width * (0.15 + 0.7 * ((step * 37 + 11) % 100) / 100.0))
    v = int(height * (0.35 + 0.5 * ((step * 53 + 29) % 100) / 100.0))
    return u, v


def render(camera, step):
    width, height = ARGS.width, ARGS.height
    colors = {"head": (235, 235, 235), "wrist_l": (225, 235, 245), "wrist_r": (245, 235, 225)}
    image = Image.new("RGB", (width, height), colors[camera])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(24, height // 8))
    small = ImageFont.load_default(size=max(12, height // 24))
    draw.text((width // 40, height // 40), f"{camera} STEP {step}", fill=(0, 0, 0), font=font)
    draw.text((width // 40, height - height // 12), f"{width}x{height}", fill=(90, 90, 90), font=small)
    if camera == "head":
        u, v = disc_position(step, width, height)
        radius = max(6, height // 40)
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(220, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def state_json():
    width, height = ARGS.width, ARGS.height
    focal = 0.9 * width
    camera = {
        "intrinsics": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        "extrinsics_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "size": [width, height],
    }
    arm = {"tcp_pos": [0.0, 0.0, 1.0], "tcp_quat": [1, 0, 0, 0], "tcp_rpy": [0, 0, 0], "gripper": 1.0}
    return {
        "left": arm,
        "right": arm,
        "cameras": {"head": camera, "wrist_l": camera, "wrist_r": camera},
        "sim_time_left_s": 32.0,
        "budget_left": STATE["budget_left"],
        "step": STATE["step"],
    }


def handle_cmd(payload):
    cmd = payload.get("cmd")
    started = time.time()
    with LOCK:
        if cmd == "status":
            return dict(STATE["last"] or {"note": "no command yet"}, exit_code=0)
        if STATE["over"]:
            return {"error": "episode is over", "exit_code": 3}
        if cmd == "obs":
            reply = {"cmd": "obs", "step": STATE["step"], "budget_left": STATE["budget_left"], "obs_updated": True}
        elif cmd == "done":
            STATE["over"] = True
            reply = {"cmd": "done", "success": False}
        else:
            if STATE["budget_left"] <= 0:
                STATE["over"] = True
                return {"error": "command budget exhausted", "exit_code": 3}
            if cmd == "wait":
                time.sleep(max(0.0, float(payload.get("sec", 0))) * ARGS.wait_scale)
            STATE["step"] += 1
            STATE["budget_left"] -= 1
            reply = {
                "cmd": cmd,
                "arm": payload.get("arm"),
                "requested": {k: v for k, v in payload.items() if k not in ("cmd", "arm")},
                "clipped": False,
                "plan_ok": True,
                "plan_fail_reason": None,
                "error_m": 0.002,
                "settled": True,
                "sim_time_left_s": 32.0,
                "budget_left": STATE["budget_left"],
                "step": STATE["step"],
                "obs_updated": True,
            }
        reply["exit_code"] = 0
        STATE["last"] = {k: v for k, v in reply.items() if k not in ("exit_code", "obs_updated")}
        STATE["log"].append({"t": started, "wall_s": round(time.time() - started, 3), "request": payload, "reply": STATE["last"]})
        return reply


class Handler(BaseHTTPRequestHandler):
    admin = False

    def log_message(self, *_):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.admin and self.path == "/admin/result":
            with LOCK:
                truth = {str(s): disc_position(s, ARGS.width, ARGS.height) for s in range(STATE["step"] + 1)}
                return self.send_json({"task": STATE["task"], "steps": STATE["step"], "over": STATE["over"],
                                       "image_size": [ARGS.width, ARGS.height], "disc_truth": truth, "log": STATE["log"]})
        if not self.admin and self.path.startswith("/v1/obs/"):
            name = self.path.rsplit("/", 1)[-1]
            with LOCK:
                step = STATE["step"]
                if name == "state.json":
                    return self.send_json(state_json())
            if name in ("head.png", "wrist_l.png", "wrist_r.png"):
                body = render(name[:-4], step)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        payload = self.read_json()
        if self.admin and self.path == "/admin/reset":
            with LOCK:
                STATE.update(step=0, budget_left=BUDGET, over=False, last=None, log=[], task=payload.get("task"))
            return self.send_json({"instruction": ARGS.instruction})
        if not self.admin and self.path == "/v1/cmd":
            return self.send_json(handle_cmd(payload))
        self.send_json({"error": "not found"}, 404)


class AdminHandler(Handler):
    admin = True


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18700)
    parser.add_argument("--admin-port", type=int, default=18701)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--wait-scale", type=float, default=1.0)
    parser.add_argument("--instruction", default="Stack the three bowls together.")
    ARGS = parser.parse_args()
    agent = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    admin = ThreadingHTTPServer(("127.0.0.1", ARGS.admin_port), AdminHandler)
    threading.Thread(target=admin.serve_forever, daemon=True).start()
    print(f"fake robo-server: agent {ARGS.host}:{ARGS.port}, admin 127.0.0.1:{ARGS.admin_port}, "
          f"image {ARGS.width}x{ARGS.height}", flush=True)
    agent.serve_forever()


if __name__ == "__main__":
    main()
