"""Read a checkpoint continuation actor using its own catalog code.

Only programmatic policy_source continuation uses this adapter. The current
production catalog remains strict and does not accept old task metadata.
"""
import json
from pathlib import Path
import subprocess
import sys


def read(root, task):
    root = Path(root).resolve()
    code = '''import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from rsi.policy_catalog import resolve_task, runtime_settings
_, route = resolve_task(None, sys.argv[2], root=root)
print(json.dumps(dict(route=route, settings=runtime_settings(root=root))))
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(root), task],
                            capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)
