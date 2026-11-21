#!/usr/bin/env python3
"""Build the shared robot API manual."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roboshell.server.tools import enabled_tools, find_tool


def manual():
    text = (ROOT / "agent/api/prompts/robodojo/system.md").read_text().rstrip() + "\n"
    text += "\n# Shared robo commands\n"
    for name in enabled_tools():
        path = Path(find_tool(name, None)).with_name("interface.md")
        text += "\n" + path.read_text().strip() + "\n"
    return text


def build(task=None):
    out = ROOT / "agent/build/AGENTS_generic.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".AGENTS_generic.{os.getpid()}.tmp")
    temporary.write_text(manual())
    temporary.replace(out)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check"])
    parser.add_argument("--task", help="Accepted for CLI compatibility; manual is shared.")
    args = parser.parse_args()
    print(build())
