"""Submit Python code and read feedback from the current robot session."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete session response; execution may have happened. "
            "Query status before choosing the next action."
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def send(stream, value):
    data = json.dumps(value, ensure_ascii=True).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Session message exceeds 16 MiB")
    stream.write(data)
    stream.flush()


def request(payload, *, socket_path=None):
    address = socket_path or os.environ.get("EMBODIEDEVO_SOCKET")
    if not address:
        raise RuntimeError(
            "Run inside an initialized workspace with EMBODIEDEVO_SOCKET configured"
        )
    # Send each request once; inspect session status after a connection error.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(address)
        with connection.makefile("rwb") as stream:
            send(stream, payload)
            return receive(stream)


def source_snapshot(workspace, entry, includes):
    workspace = Path(workspace).resolve()
    files = []
    for name in [*includes, entry]:
        path = (workspace / name).resolve()
        relative = path.relative_to(workspace)
        if relative.parts[0] != "submission" or path.suffix != ".py":
            raise ValueError("Submit .py files inside submission/")
        files.append({"path": str(relative), "code": path.read_text(encoding="utf-8")})
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("exec", help="execute a snapshot in the existing simulator session")
    execute.add_argument("file", nargs="?", default="submission/solution.py")
    execute.add_argument(
        "--include", action="append", default=[], help="prepend helper code, in order"
    )
    for command in ("status", "observe", "instruction", "finish"):
        sub.add_parser(command)
    args = parser.parse_args(argv)
    payload = {"op": args.command}
    try:
        if args.command == "exec":
            workspace = os.environ.get(
                "EMBODIEDEVO_WORKSPACE", str(Path(__file__).resolve().parents[1])
            )
            payload["files"] = source_snapshot(workspace, args.file, args.include)
        result = request(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok", False) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
