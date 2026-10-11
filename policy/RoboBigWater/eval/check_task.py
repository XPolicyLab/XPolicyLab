#!/usr/bin/env python3
"""Delivery check for tasks/<task>/: files present, tools load, the manual carries no task words."""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from roboshell.server.tools import base_task, load_tools  # noqa: E402

task = base_task(sys.argv[1])
base = os.path.join(ROOT, "tasks", task)
ok = True
for name in ("enabled_tools.txt", "playbook.md", "skill.md"):
    present = os.path.exists(os.path.join(base, name))
    print(("ok  " if present else "MISSING ") + name)
    ok &= present
try:
    registry = load_tools(task)
    print("ok   tools load:", sorted(registry) or "none")
except Exception as error:
    print("FAIL tools load:", error)
    ok = False
manual = subprocess.run([sys.executable, os.path.join(ROOT, "roboshell", "docs.py"), "check", "--task", task], capture_output=True, text=True)
print(("ok   " if manual.returncode == 0 else "FAIL ") + manual.stdout.strip())
ok &= manual.returncode == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
