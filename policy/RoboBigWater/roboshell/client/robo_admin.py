#!/usr/bin/env python3
"""robo-admin: operator-side client. Never installed in the agent container.

Environment:
  ROBO_ADMIN_SERVER  base URL of the admin port (default http://127.0.0.1:28701)
"""

import argparse
import json
import os
import sys
import urllib.request

SERVER = os.environ.get("ROBO_ADMIN_SERVER", "http://127.0.0.1:28701").rstrip("/")
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(route, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(SERVER + route, data=data, headers={"Content-Type": "application/json"})
    with _opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(prog="robo-admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    reset = sub.add_parser("reset", help="start an episode; prints the task instruction")
    reset.add_argument("--task", required=True)
    reset.add_argument("--seed", type=int, default=0)
    reset.add_argument("--layout", type=int, required=True)
    sub.add_parser("result", help="print the result of the current or last episode")
    replay = sub.add_parser("replay", help="replay a finished episode from its joint targets and compare the final state")
    replay.add_argument("directory")
    replay.add_argument("--then-home", action="store_true", help="counterfactual: append `home both` to the recorded episode")
    sub.add_parser("shutdown", help="stop robo-server")
    finish = sub.add_parser("finish", help="the agent stopped without `done`: judge now and close the episode")
    finish.add_argument("--reason", default="agent_exit")
    args = parser.parse_args()

    if args.cmd == "reset":
        reply = call("/admin/reset", {"task": args.task, "seed": args.seed, "layout": args.layout})
        if "error" in reply:
            print(json.dumps(reply), file=sys.stderr)
            return 5
        print(reply["instruction"])
    elif args.cmd == "replay":
        print(json.dumps(call("/admin/replay", {"directory": args.directory, "then": {"cmd": "home", "arm": "both"} if args.then_home else None}), indent=2))
    elif args.cmd == "finish":
        print(json.dumps(call("/admin/finish", {"reason": args.reason}), indent=2))
    elif args.cmd == "shutdown":
        print(json.dumps(call("/admin/shutdown", {})))
    else:
        print(json.dumps(call("/admin/result"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
