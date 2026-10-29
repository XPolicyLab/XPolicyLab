#!/usr/bin/env python3
"""One line per turn of the newest program episode under a runs directory.

    python3 scripts/episode_status.py /path/to/runs [--code]
"""

import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    show_code = "--code" in sys.argv
    episodes = sorted(root.glob("*/seed-*/events.jsonl"), key=lambda p: p.stat().st_mtime)
    if not episodes:
        print("no episodes")
        return
    path = episodes[-1]
    print(path.parent)
    for line in path.read_text().splitlines():
        event = json.loads(line)
        kind = event["event"]
        if kind == "model_reply":
            text = event["text"].strip().replace("\n", " ")
            print(f"  reply {event['call']:>2} ({event.get('seconds', 0):.0f}s, "
                  f"{event['usage'].get('output_tokens')} tok): {text[:160]}")
        elif kind == "cell":
            result = event["result"]
            detail = ""
            if result.get("escalation"):
                ev = result["escalation"]["event"]
                detail = f" {ev['kind']} from {ev['source']} {json.dumps(ev['detail'])[:160]}"
            if result.get("error"):
                detail = " " + result["error"].strip().splitlines()[-1][:200]
            executions = [f"{e['skill']}:{e['event']['kind']}" for e in result.get("events", [])]
            print(f"    cell {result['status']}{detail} | {', '.join(executions)[:200]}")
            if show_code:
                print("      " + event["code"].replace("\n", "\n      "))
            if result.get("stdout"):
                print("      out: " + result["stdout"].strip().replace("\n", " | ")[:300])
        elif kind in ("verify", "finish", "outcome", "error", "protocol_error", "kernel_died"):
            payload = {k: v for k, v in event.items() if k not in ("event", "time", "seq")}
            print(f"  {kind}: {json.dumps(payload)[:300]}")
    summary = path.parent / "summary.json"
    if summary.exists():
        print("summary:", summary.read_text().replace("\n", " ")[:600])


if __name__ == "__main__":
    main()
