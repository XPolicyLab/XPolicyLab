#!/usr/bin/env python3
"""robo: the agent-side command line client. Standard library only.

Environment:
  ROBO_SERVER   base URL of robo-server (default http://127.0.0.1:28700)
  ROBO_OBS_DIR  where observations are written (default ./obs)
  ROBO_TIMEOUT  seconds to wait for one command (default 300)
"""

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request

EXIT_OK, EXIT_BAD_ARGS, EXIT_EXEC_FAILED, EXIT_EPISODE_OVER, EXIT_CLIENT_TIMEOUT = 0, 1, 2, 3, 4
OBS_FILES = ("head.png", "wrist_l.png", "wrist_r.png", "state.json")
OPTIONAL_FILES = ("head_depth.npy", "wrist_l_depth.npy", "wrist_r_depth.npy", "head_depth.png", "wrist_l_depth.png", "wrist_r_depth.png")
ARMS = ("left", "right")

SERVER = os.environ.get("ROBO_SERVER", "http://127.0.0.1:28700").rstrip("/")
OBS_DIR = os.environ.get("ROBO_OBS_DIR", "obs")
TIMEOUT = float(os.environ.get("ROBO_TIMEOUT", "300"))
# ROBO_PROFILE=open (generic profile): adds goto, reach, pixels and `robo py` (Python cells in a persistent kernel)
OPEN = os.environ.get("ROBO_PROFILE", "") == "open"
KERNEL_SOCKET = os.environ.get("ROBO_KERNEL_SOCKET", "/tmp/robo_kernel.sock")

# never route robot traffic through an HTTP proxy
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write(f"robo: error: {message}\n")
        sys.exit(EXIT_BAD_ARGS)


def gripper_value(text):
    if text in ("open", "close"):
        return 1.0 if text == "open" else 0.0
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected open, close or a number in [0, 1]")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("expected a number in [0, 1]")
    return value


def build_parser():
    parser = Parser(prog="robo", description="Control the robot. Units: meters, degrees, seconds.")
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=Parser)

    sub.add_parser("obs", help="refresh obs/ (free)")
    sub.add_parser("status", help="feedback of the last command (free)")
    sub.add_parser("done", help="declare the task finished and end the episode")

    move = sub.add_parser("move", help="translate the TCP in the world frame")
    move.add_argument("arm", choices=ARMS)
    for axis in ("dx", "dy", "dz"):
        move.add_argument(f"--{axis}", type=float, default=0.0, metavar="M")

    rotate = sub.add_parser("rotate", help="rotate about the TCP")
    rotate.add_argument("arm", choices=ARMS)
    for axis in ("roll", "pitch", "yaw"):
        rotate.add_argument(f"--{axis}", type=float, default=0.0, metavar="DEG")
    rotate.add_argument("--frame", choices=("world", "tool"), default="world")

    point = sub.add_parser("point", help="set the gripper direction and the axis along which the fingers open")
    point.add_argument("arm", choices=ARMS)
    point.add_argument("preset", choices=("down", "forward", "down45"), help="direction the gripper points to")
    point.add_argument("--open", required=True, choices=("x", "y", "z"), help="world axis along which the fingers open")

    gripper = sub.add_parser("gripper", help="1 is fully open, 0 is fully closed")
    gripper.add_argument("arm", choices=ARMS)
    gripper.add_argument("value", type=gripper_value, metavar="open|close|0..1")

    home = sub.add_parser("home", help="return to the initial pose")
    home.add_argument("arm", choices=ARMS + ("both",))

    wait = sub.add_parser("wait", help="let the simulation run")
    wait.add_argument("sec", type=float)
    if OPEN:
        for name, text in (("goto", "move the TCP to an absolute pose"), ("reach", "check whether a TCP pose is reachable (free, nothing moves)")):
            cmd = sub.add_parser(name, help=text)
            cmd.add_argument("arm", choices=ARMS)
            for axis in ("x", "y", "z"):
                cmd.add_argument(f"--{axis}", type=float, metavar="M", help="world coordinate; omitted: keep the current one")
            cmd.add_argument("--approach", help="down, forward, down45 or a direction a,b,c; omitted: keep")
            cmd.add_argument("--open", help="x, y, z, an angle in degrees about the approach direction, or a direction a,b,c; omitted: keep")
            if name == "goto":
                cmd.add_argument("--path", choices=("line", "joint"), default="line", help="line: straight TCP line (default); joint: joint interpolation")
        pixels = sub.add_parser("pixels", help="write the latest images as numpy arrays obs/<camera>_rgb.npy (free)")
        pixels.add_argument("--camera", choices=("head", "wrist_l", "wrist_r", "all"), default="all")
        py = sub.add_parser("py", help="run Python code in the persistent kernel (from FILE, or from stdin)")
        py.add_argument("file", nargs="?", help="file with the code; omitted: read the code from stdin")
    for command in extra_commands():
        extra = sub.add_parser(command["name"], help=command.get("help", f"tool {command.get('tool', '')}"))
        for arg in command.get("args", []):
            kind = {"float": float, "int": int}.get(arg.get("type", "str"), str)
            options = {"type": kind, "help": arg.get("help")}
            if arg.get("choices"):
                options["choices"] = arg["choices"]
            if arg.get("positional"):
                extra.add_argument(arg["name"], **options)
            else:
                if arg.get("required"):
                    options["required"] = True
                else:
                    options["default"] = arg.get("default")
                extra.add_argument(f"--{arg['name']}", **options)
    return parser


def extra_commands():
    """Commands added by the tools of the running task; the server publishes their argument lists."""
    try:
        with _opener.open(SERVER + "/v1/tools", timeout=10) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return []


def post(payload):
    request = urllib.request.Request(
        SERVER + "/v1/cmd",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _opener.open(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def download_obs():
    os.makedirs(OBS_DIR, exist_ok=True)
    for name in OBS_FILES + OPTIONAL_FILES:
        try:
            with _opener.open(f"{SERVER}/v1/obs/{name}", timeout=60) as response:
                data = response.read()
        except urllib.error.HTTPError as err:
            if err.code == 404 and name in OPTIONAL_FILES:
                continue
            raise
        tmp = os.path.join(OBS_DIR, name + ".part")
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, os.path.join(OBS_DIR, name))


def write_files(reply):
    """Files a command returns (base64 in `files`) go into the observation folder; the reply lists their paths."""
    import base64

    files = reply.pop("files", None)
    if files:
        os.makedirs(OBS_DIR, exist_ok=True)
        reply["files"] = []
        for name, data in files.items():
            path = os.path.join(OBS_DIR, os.path.basename(name))
            with open(path, "wb") as handle:
                handle.write(base64.b64decode(data))
            reply["files"].append(path)


def run_python(file):
    """`robo py`: send the code to the kernel (started on first use) and print its output."""
    import signal
    import subprocess
    import time

    code = open(file).read() if file else sys.stdin.read()
    kernel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")

    def connect():
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(KERNEL_SOCKET)
        return conn

    try:
        conn = connect()
    except OSError:
        log = open(os.path.join(os.path.dirname(KERNEL_SOCKET), "robo_kernel.log"), "a")
        subprocess.Popen([sys.executable, kernel, KERNEL_SOCKET], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True, env=dict(os.environ))
        for _ in range(100):
            time.sleep(0.1)
            try:
                conn = connect()
                break
            except OSError:
                continue
        else:
            print("robo py: the Python kernel did not start (see /tmp/robo_kernel.log)")
            return EXIT_EXEC_FAILED
        print("[Python kernel started: variables from earlier cells, if any, are gone]")

    def interrupt(signum, frame):
        try:
            os.kill(int(open(KERNEL_SOCKET + ".pid").read()), signal.SIGINT)
        except (OSError, ValueError):
            pass
        sys.exit(130)

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with conn:
        conn.sendall(json.dumps({"code": code}).encode())
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = conn.recv(65536)
            if not data:
                break
            chunks.append(data)
    try:
        reply = json.loads(b"".join(chunks))
    except ValueError:
        print("robo py: the kernel stopped while running the cell; the next `robo py` starts a new one")
        return EXIT_EXEC_FAILED
    sys.stdout.write(reply.get("output", ""))
    return EXIT_OK if reply.get("ok") else EXIT_EXEC_FAILED


def main():
    args = vars(build_parser().parse_args())
    if args.get("cmd") == "py":
        return run_python(args.get("file"))
    args = {k: v for k, v in args.items() if v is not None}   # omitted options: the server's defaults
    try:
        reply = post(args)
    except (socket.timeout, TimeoutError):
        print(json.dumps({"error": "client_timeout", "hint": "the command may still be running; use `robo status`"}))
        return EXIT_CLIENT_TIMEOUT
    except urllib.error.HTTPError as err:
        try:
            reply = json.loads(err.read())
        except ValueError:
            print(json.dumps({"error": f"server returned HTTP {err.code}"}))
            return EXIT_EXEC_FAILED
    except (urllib.error.URLError, OSError) as err:
        print(json.dumps({"error": f"cannot reach robo-server: {err}"}))
        return EXIT_EXEC_FAILED

    exit_code = int(reply.pop("exit_code", EXIT_OK))
    write_files(reply)
    if reply.pop("obs_updated", False):
        try:
            download_obs()
        except (urllib.error.URLError, OSError) as err:
            reply["obs_error"] = f"observation download failed: {err}"
    print(json.dumps(reply))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
