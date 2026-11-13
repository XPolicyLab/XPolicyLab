import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(os.environ['ACTWEAVE_ROOT']).resolve()
xpl_root = Path(__file__).resolve().parents[2]
config = (root / os.environ['EVO_CONFIG']).resolve()
os.environ['EVO_CONFIG'] = str(config)
settings = json.loads(config.read_text())
children = []


def stop(signum, frame):
    raise SystemExit(128 + signum)


def terminate_children():
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for child in reversed(children):
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


for signum in (signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, stop)
try:
    bridge = subprocess.Popen(
        ['node', '--import', 'tsx', root / 'src/main.ts'],
        cwd=root, start_new_session=True,
    )
    children.append(bridge)
    subprocess.run(
        ['bash', xpl_root / 'utils/wait_for_policy_server.sh', '127.0.0.1',
         str(settings['ports']['bridge']), str(bridge.pid), 'ActWeave runtime', '60'],
        check=True, timeout=65,
    )
    server = subprocess.Popen(
        [sys.executable, xpl_root / 'setup_policy_server.py', *sys.argv[1:]],
        cwd=xpl_root, start_new_session=True,
    )
    children.append(server)
    _, status = os.wait()
    code = os.waitstatus_to_exitcode(status)
    raise SystemExit(128 - code if code < 0 else code or 1)
finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    terminate_children()
