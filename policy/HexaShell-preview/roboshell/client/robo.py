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


def main():
    args = vars(build_parser().parse_args())
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
    if reply.pop("obs_updated", False):
        try:
            download_obs()
        except (urllib.error.URLError, OSError) as err:
            reply["obs_error"] = f"observation download failed: {err}"
    print(json.dumps(reply))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
