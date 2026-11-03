#!/usr/bin/env python3
"""Persistent Python kernel behind `robo py`. Standard library + numpy only.

    kernel.py SOCKET     serve cells on a Unix socket until the container ends

Each request is one JSON object {"code": "..."}; the reply is {"output": "...", "ok": true|false}. Cells run one at a
time in one namespace that starts with `from robolib import *`, so variables, functions and imports stay defined
from one cell to the next. Like a notebook, the value of a final expression is printed. SIGINT interrupts the
running cell (the `robo py` client sends it when it is stopped).
"""
import ast
import contextlib
import io
import json
import os
import signal
import socket
import sys
import traceback

MAX_OUTPUT = 20000


def run_cell(code, namespace):
    """Execute one cell; returns (output text, ok)."""
    buffer = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), namespace)
            if last is not None:
                value = eval(compile(last, "<cell>", "eval"), namespace)
                if value is not None:
                    print(repr(value))
        except KeyboardInterrupt:
            ok = False
            print("KeyboardInterrupt: the cell was interrupted")
        except SystemExit as stop:
            print(f"SystemExit({stop.code}) ignored: the kernel keeps running")
        except BaseException:
            ok = False
            kind, error, tb = sys.exc_info()
            frames = [f for f in traceback.extract_tb(tb) if f.filename in ("<cell>",) or "robolib" in f.filename]
            print("Traceback (most recent call last):")
            print("".join(traceback.format_list(frames[-6:])).rstrip())
            print("".join(traceback.format_exception_only(kind, error)).rstrip())
    text = buffer.getvalue()
    if len(text) > MAX_OUTPUT:
        text = text[: MAX_OUTPUT // 2] + f"\n... [{len(text) - MAX_OUTPUT} characters cut] ...\n" + text[-MAX_OUTPUT // 2:]
    return text, ok


def receive(conn):
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        chunks.append(data)
    return json.loads(b"".join(chunks) or b"{}")


def main():
    path = sys.argv[1]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    namespace = {"__name__": "__cell__"}
    startup, _ = run_cell("from robolib import *", namespace)
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(4)
    with open(path + ".pid", "w") as handle:
        handle.write(str(os.getpid()))
    signal.signal(signal.SIGINT, signal.default_int_handler)
    while True:
        try:
            conn, _ = server.accept()
        except KeyboardInterrupt:   # an interrupt that came between cells
            continue
        with conn:
            try:
                request = receive(conn)
                output, ok = run_cell(request.get("code", ""), namespace)
                if startup:
                    output, startup = f"[kernel start] {startup}\n{output}", ""
                conn.sendall(json.dumps({"output": output, "ok": ok}).encode())
            except KeyboardInterrupt:
                continue
            except (OSError, ValueError) as error:
                try:
                    conn.sendall(json.dumps({"output": f"kernel error: {error}", "ok": False}).encode())
                except OSError:
                    pass


if __name__ == "__main__":
    main()
